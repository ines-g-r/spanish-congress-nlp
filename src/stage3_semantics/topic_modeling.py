"""
Topic Modeling over Parliamentary Interventions with LDA.

Preprocesses lemmatized sentences into a bag-of-words corpus, searches over
LDA hyperparameters, assigns a readable label to each topic via zero-shot 
classification, and saves the final per-intervention topic assignments.

Usage:
    python src/stage3_semantics/topic_modeling.py --step search   # hyperparameter search
    python src/stage3_semantics/topic_modeling.py --step label    # label + save the chosen model
    python src/stage3_semantics/topic_modeling.py                 # runs both steps in order
"""

import argparse
import gc
from collections import Counter, defaultdict
from random import sample, seed
import gensim
import ijson
import numpy as np
import pandas as pd
import torch
import umap
from nltk.corpus import stopwords
from transformers import AutoTokenizer, pipeline

STANZA_RESULTS_PATH = "results/stanza/stanza_results.json"
LDA_DIR = "results/lda/ldamulticore"
SUMMARY_CSV = "results/lda/summary.csv"

# Model chosen (after inspecting 'summary.csv') to carry forward to labeling.
CHOSEN_MODEL_NAME = "ldamulticore_45asymmetric"

EXTRA_STOPWORDS = [
    "señora", "señor", "presidenta", "presidente", "diputado", "diputada",
    "señoría", "ministro", "ministra", "cámara", "congreso", "senado",
    "pleno", "sesión", "comisión", "subcomisión", "grupo", "parlamentario",
    "gracia", "ser", "estar", "tener", "hacer", "decir", "ir", "poder",
    "venir", "dar", "haber",
]
UPOS_KEEP = {"NOUN", "VERB", "ADJ", "ADV"}

CANDIDATE_LABELS = [
    "sanidad y servicios médicos", "salud y prevención social", "sistema educativo",
    "legislación laboral", "seguridad social y pensiones", "cambio climático",
    "justicia y derecho penal", "defensa y seguridad", "colaboración internacional",
    "ciencia e investigación", "crisis financiera", "gestión del gasto público",
    "competencias autonómicas y locales", "fiscalidad y sistema tributario",
    "producción y consumo", "democracia y dinámica de partidos",
    "explotación de recursos naturales", "crisis migratoria y extranjería",
    "sistema electoral", "historia", "tecnología y sociedad de la información",
    "deporte", "cultura, patrimonio y comunicación",
    "independentismo y política autonómica", "turismo", "política comercial e industria",
    "energía", "infraestructuras y transporte", "vivienda", "igualdad y colectivos",
    "protección social, familia e infancia",
]

HPARAM_GRID = [
    {"n_clusters": 15, "alpha": "symmetric"}, {"n_clusters": 15, "alpha": "asymmetric"},
    {"n_clusters": 35, "alpha": "symmetric"}, {"n_clusters": 35, "alpha": "asymmetric"},
    {"n_clusters": 40, "alpha": "symmetric"}, {"n_clusters": 40, "alpha": "asymmetric"},
    {"n_clusters": 45, "alpha": "symmetric"}, {"n_clusters": 45, "alpha": "asymmetric"},
    {"n_clusters": 50, "alpha": "symmetric"}, {"n_clusters": 50, "alpha": "asymmetric"},
    {"n_clusters": 70, "alpha": "symmetric"}, {"n_clusters": 70, "alpha": "asymmetric"},
]


def get_stopwords() -> list[str]:
    """Spanish stopwords plus a few words specific to parliamentary language."""
    stop_es = stopwords.words("spanish")
    stop_es.extend(EXTRA_STOPWORDS)
    return stop_es


def preprocess(stop_es: list[str], upos: set[str] | None = None, stanza_path: str = STANZA_RESULTS_PATH):
    """Lemmatize and filter each intervention's sentences into a document of tokens.

    If 'upos' is given, only words with that POS tag are kept.
    Returns the list of documents (tokens) and the matching list of IDs.
    """

    documents = []
    ids = []

    with open(stanza_path, "rb") as f:
        parser = ijson.items(f, "item")
        for item in parser:
            words = []
            for sentence in item.get("sentences", []):
                for word in sentence.get("words", []):
                    if upos and word["upos"] not in upos:
                        continue
                    lemma = word["lemma"].lower()
                    if lemma not in stop_es:
                        words.append(lemma)
            if words:
                documents.append(words)
                ids.append(item.get("ID"))

    return documents, ids


def metadata_generator(stanza_path: str = STANZA_RESULTS_PATH):
    """Yields full interventions one by one (including their sentences)."""

    with open(stanza_path, "rb") as f:
        parser = ijson.items(f, "item")
        for item in parser:
            yield item


def build_corpus(documents: list[list[str]], no_below: int = 10, no_above: float = 0.35):
    """Builds the dictionary and bag-of-words corpus gensim needs."""

    id2word = gensim.corpora.Dictionary(documents)
    id2word.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [id2word.doc2bow(doc) for doc in documents]
    return id2word, corpus


def train_lda(hparams: dict, corpus, id2word):
    """Train an LDA model with the given hyperparameters."""
    
    return gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=hparams["n_clusters"],
        alpha=hparams["alpha"],
        iterations=300,
        passes=10,
        chunksize=1000,
        eval_every=0,
        random_state=21,
    )


def hyperparameter_search(corpus, id2word, documents: list[list[str]], hparam_grid: list[dict] = HPARAM_GRID, output_dir: str = LDA_DIR, summary_csv: str = SUMMARY_CSV) -> pd.DataFrame:
    """Train one LDA model per hyperparameter combination and save coherence/
    perplexity results.

    For each entry in 'hparam_grid', trains an LDA model, saves it to
    'output_dir', and evaluates it with log-perplexity and c_v coherence. 
    Results are written to 'summary_csv'.
    """

    results = []
    for hparams in hparam_grid:
        lda_model = train_lda(hparams, corpus, id2word)
        lda_model.save(f"{output_dir}/ldamulticore_{hparams['n_clusters']}{hparams['alpha']}.model")

        log_perplexity = lda_model.log_perplexity(corpus)
        coherence_model = gensim.models.CoherenceModel(
            model=lda_model, texts=documents, dictionary=id2word, coherence="c_v"
        )
        coherence_score = coherence_model.get_coherence()
        coherence_per_topic = coherence_model.get_coherence_per_topic()

        print(f"\nn_clusters={hparams['n_clusters']} alpha={hparams['alpha']}")
        print(f"log-perplexity: {log_perplexity:.4f}")
        for i, score in enumerate(coherence_per_topic):
            print(f"  Topic {i + 1}: {score:.4f}")
        print(f"Overall coherence: {coherence_score:.4f}")

        results.append({
            "n_clusters": hparams["n_clusters"],
            "alpha": hparams["alpha"],
            "log-perplexity": log_perplexity,
            "coherence": coherence_score,
        })

        del lda_model, coherence_model
        gc.collect()

    results_df = pd.DataFrame(results)
    results_df.to_csv(summary_csv, index=False)
    print(f"\nSaved {summary_csv}")
    return results_df


def assign_topic_labels(model, corpus, metadata, candidate_labels: list[str] = CANDIDATE_LABELS, sample_size: int = 50, random_seed: int = 21) -> dict[int, str]:
    """Assign a human-readable label to each LDA topic via zero-shot classification.

    For each topic, samples up to 'sample_size' of its dominant documents,
    classifies their text against 'candidate_labels', and assigns the
    topic its most common predicted label.
    """
    tokenizer = AutoTokenizer.from_pretrained("sileod/deberta-v3-base-tasksource-nli", model_max_length=500)
    classifier = pipeline("zero-shot-classification", model="sileod/deberta-v3-base-tasksource-nli", tokenizer=tokenizer)

    seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    doc_topic_matrix = np.array([
        [prob for _, prob in model.get_document_topics(bow, minimum_probability=0)]
        for bow in corpus
    ])
    dominant_topics = np.argmax(doc_topic_matrix, axis=1)

    indices_by_topic = defaultdict(list)
    for idx, t in enumerate(dominant_topics):
        indices_by_topic[t].append(idx)

    indices_to_topic = {}
    for topic, idxs in indices_by_topic.items():
        sampled_idxs = sample(idxs, min(sample_size, len(idxs)))
        for i in sampled_idxs:
            indices_to_topic[i] = topic

    topic_docs = defaultdict(list)
    for current_idx, doc in enumerate(metadata):
        if current_idx not in indices_to_topic:
            continue
        topic = indices_to_topic[current_idx]
        sens = [
            sentence.get("sentence_text", "")
            for sentence in doc.get("sentences", [])
            if sentence.get("sentence_text", "").lower() not in ("", "no disponible")
        ]
        topic_docs[topic].append(" ".join(sens))

        del indices_to_topic[current_idx]
        if not indices_to_topic:
            break

    all_texts, topic_map = [], []
    for topic_id, docs in topic_docs.items():
        for doc in docs:
            all_texts.append(doc)
            topic_map.append(topic_id)

    print(f"Classifying {len(all_texts)} sampled documents...")
    results = classifier(all_texts, candidate_labels, truncation=True, batch_size=8)
    predicted_labels = [r["labels"][0] for r in results]

    topic_labels = defaultdict(list)
    for topic_id, label in zip(topic_map, predicted_labels):
        topic_labels[topic_id].append(label)

    return {t: Counter(lbls).most_common(1)[0][0] for t, lbls in topic_labels.items()}


def build_document_topics(model, corpus) -> np.ndarray:
    """Return the full document-topic matrix (probability of each topic for each 
    document) for a fitted LDA model."""

    doc_topics = []
    for bow in corpus:
        topic_dist = model.get_document_topics(bow, minimum_probability=0)
        vec = np.array([prob for _, prob in sorted(topic_dist, key=lambda x: x[0])])
        doc_topics.append(vec)
    return np.array(doc_topics)


def process_and_save(
    doc_topics: np.ndarray,
    ids: list,
    metadata,
    topic_labels: dict[int, str],
    filename: str,
):
    """Projects the topics to 2D with UMAP, merges everything with metadata, and saves it.
 
    Saves a .pt file with the final dataframe (dominant topic, speaker,
    party, legislature, etc.) and the topic label dictionary.
    """

    umap_2d = umap.UMAP(n_components=2, random_state=21)
    projections = umap_2d.fit_transform(doc_topics)
    all_topic_ids = np.argmax(doc_topics, axis=1)

    full_meta_dict = {
        str(d.get("ID")): {
            "ID": d.get("ID"), "ORGANO": d.get("ORGANO"), "LEGISLATURA": d.get("LEGISLATURA"),
            "PARTIDO": d.get("PARTIDO"), "PARTIDOIMPUTADO": d.get("PARTIDOIMPUTADO"),
            "ORADOR": d.get("ORADOR"), "CARGOORADOR": d.get("CARGOORADOR"), "SESION": d.get("SESION"),
            "OBJETOINICIATIVA": d.get("OBJETOINICIATIVA"), "TIPOINICIATIVA": d.get("TIPOINICIATIVA"),
            "GENEROIMPUTADO": d.get("GENEROIMPUTADO"),
        }
        for d in metadata
    }

    print("Merging topics with metadata...")
    final_rows = []
    for i, current_id in enumerate(ids):
        m = full_meta_dict.get(str(current_id))
        if not m:
            continue
        final_rows.append({
            "x": projections[i, 0], "y": projections[i, 1], "id": m["ID"], "organo": m["ORGANO"],
            "legislatura": m["LEGISLATURA"], "partido": m["PARTIDO"], "partido_imputado": m["PARTIDOIMPUTADO"],
            "orador": m["ORADOR"], "cargo": m["CARGOORADOR"], "fecha": m["SESION"],
            "objetoiniciativa": m["OBJETOINICIATIVA"], "tipoiniciativa": m["TIPOINICIATIVA"],
            "genero": m["GENEROIMPUTADO"], "topic_id": all_topic_ids[i],
        })

    df = pd.DataFrame(final_rows)
    df["topic_label"] = df["topic_id"].map(topic_labels)
    df["legislatura"] = pd.Categorical(df["legislatura"], categories=["XI", "XII", "XIII", "XIV", "XV"], ordered=True)

    torch.save({"df": df.to_dict(), "topics": topic_labels}, filename)
    print(f"Saved {filename}")


def run_search() -> None:
    """Run the LDA hyperparameter search and save models + summary.csv."""

    stop_es = get_stopwords()
    documents, _ = preprocess(stop_es, UPOS_KEEP)
    id2word, corpus = build_corpus(documents)
    print(f"{len(corpus)} documents, {len(id2word)} unique terms")
    hyperparameter_search(corpus, id2word, documents)


def run_label(model_name: str = CHOSEN_MODEL_NAME) -> None:
    """Load 'model_name', assign topic labels, and save the final topic assignments."""

    stop_es = get_stopwords()
    documents, ids = preprocess(stop_es, UPOS_KEEP)
    id2word, corpus = build_corpus(documents)
    print(f"{len(corpus)} documents, {len(id2word)} unique terms")

    lda_model = gensim.models.LdaMulticore.load(f"{LDA_DIR}/{model_name}.model")

    topic_labels = assign_topic_labels(lda_model, corpus, metadata_generator(), sample_size=150)
    for topic_id, label in sorted(topic_labels.items()):
        print(f"Topic {topic_id + 1}: {label}")

    doc_topics = build_document_topics(lda_model, corpus)
    process_and_save(doc_topics, ids, metadata_generator(), topic_labels, f"{LDA_DIR}/{model_name}.pt")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["search", "label", "all"],
        default="all",
        help="which step to run: search, label, or all (default)",
    )
    parser.add_argument("--model-name", default=CHOSEN_MODEL_NAME, help="model file name (without extension) used by 'label'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    steps = ["search", "label"] if args.step == "all" else [args.step]

    if "search" in steps:
        run_search()
    if "label" in steps:
        run_label(args.model_name)