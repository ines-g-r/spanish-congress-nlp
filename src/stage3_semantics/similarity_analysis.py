"""Speaker and Party Discourse Similarity, per Term.

For each term, builds a similarity network between speakers and
between parties, based on their interventions. Two methods are available:

- 'bm25': lexical similarity, using a representative sample of each
  speaker's/party's interventions and BM25 score.
- 'embeddings': semantic similarity, using multilingual sentence
  embeddings + cosine similarity, with a 2D UMAP projection for plotting.

Usage:
    python src/stage3_semantics/similarity_analysis.py --method bm25
    python src/stage3_semantics/similarity_analysis.py --method embeddings
    python src/stage3_semantics/similarity_analysis.py                      # both methods
"""

import argparse
import json
import os
from collections import Counter, defaultdict
import bm25s
import ijson
import nltk
import numpy as np
import torch
import umap
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

STANZA_RESULTS_PATH = "results/stanza/stanza_results.json"
BM25_OUTPUT_PATH = "results/similarity/similitudes_intralegislatura_bm25.json"
EMBEDDINGS_OUTPUT_PATH = "results/similarity/similitudes_intralegislatura_embeddings.json"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

EXTRA_STOPWORDS = [
    "señora", "señor", "presidenta", "presidente", "diputado", "diputada",
    "señoría", "ministro", "ministra", "cámara", "congreso", "senado",
    "pleno", "sesión", "comisión", "subcomisión", "grupo", "parlamentario",
    "gracia", "ser", "estar", "tener", "hacer", "decir", "ir", "poder",
    "venir", "dar", "haber", "señorías", "ustedes", "usted",
    "enmienda", "proposición", "ley", "voto", "favor", "contra", "abstención",
    "debate", "artículo", "aprobación", "intervención", "portavoz", "orden", "día",
]


def get_stopwords() -> list[str]:
    """Spanish stopwords plus a few words specific to parliamentary language."""
    nltk.download("stopwords", quiet=True)
    stop_es = stopwords.words("spanish")
    stop_es.extend(EXTRA_STOPWORDS)
    return stop_es


def construir_macro_texto_representativo(lista_de_textos: list[str], n_muestras: int = 100, min_palabras: int = 50) -> str:
    """Build one representative "macro-text" out of a speaker's/party's interventions.

    Drops very short interventions, trims outliers above the 95th length
    percentile, and keeps the 'n_muestras' interventions closest in length
    to the median -- meant to avoid a handful of very long speeches
    dominating the BM25 comparison.
    """
    docs = [{"texto": t, "len": len(t.split())} for t in lista_de_textos]
    docs = [d for d in docs if d["len"] >= min_palabras]

    if not docs:
        return ""

    if len(docs) > 10:
        longitudes = [d["len"] for d in docs]
        p95 = np.percentile(longitudes, 95)
        docs = [d for d in docs if d["len"] <= p95]

    longitudes_limpias = [d["len"] for d in docs]
    mediana_len = np.median(longitudes_limpias)
    docs.sort(key=lambda x: abs(x["len"] - mediana_len))
    seleccionados = docs[:n_muestras]

    return " ".join([d["texto"] for d in seleccionados])


def calcular_bloque_bm25_representativo(nombres: list[str], listas_de_textos: list[list[str]], stop_es: list[str]):
    """Pairwise BM25 similarity between speakers/parties, using their macro-texts.

    Each macro-text is used as a query against the whole corpus of
    macro-texts, so the resulting matrix gives, for every pair, how well
    one's text retrieves the other's.

    Returns the list of names that ended up with a valid macro-text, and
    the corresponding similarity matrix.
    """
    corpus = []
    nombres_validos = []

    for nombre, textos in zip(nombres, listas_de_textos):
        macro_texto = construir_macro_texto_representativo(textos, n_muestras=100)
        if macro_texto.strip():
            corpus.append(macro_texto)
            nombres_validos.append(nombre)

    N = len(corpus)
    if N <= 1:
        return nombres_validos, np.zeros((N, N))

    corpus_tokens = bm25s.tokenize(corpus, stopwords=stop_es)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    results = retriever.retrieve(corpus_tokens, k=N)
    docs_idx, scores = results.documents, results.scores

    sim_matrix = np.zeros((N, N))
    for i in range(N):
        for rank in range(N):
            j = int(docs_idx[i, rank])
            sim_matrix[i, j] = float(scores[i, rank])

    return nombres_validos, sim_matrix


def bm25_similarity_analysis(json_path: str = STANZA_RESULTS_PATH, output_path: str = BM25_OUTPUT_PATH) -> None:
    """Compute lexical (BM25) similarity between speakers and between parties, per term."""
    textos_partidos = defaultdict(lambda: defaultdict(list))
    textos_oradores = defaultdict(lambda: defaultdict(list))
    meta_orador_partido = defaultdict(dict)
    stop_es = get_stopwords()

    print(f"Grouping texts from {json_path}")
    with open(json_path, "rb") as f:
        parser = ijson.items(f, "item")
        for item in parser:
            raw_sentences = [s["sentence_text"] for s in item.get("sentences", []) if "sentence_text" in s]
            if not raw_sentences:
                continue

            texto_unido = " ".join(raw_sentences)
            leg = item.get("LEGISLATURA")
            orador = item.get("ORADOR")
            partido = item.get("PARTIDO")

            if leg and orador and partido:
                textos_oradores[leg][orador].append(texto_unido)
                textos_partidos[leg][partido].append(texto_unido)
                meta_orador_partido[leg][orador] = partido

    resultados_finales = {"por_legislatura": {}}

    for leg in textos_oradores.keys():
        print(f"Term: {leg}")

        nombres_o = list(textos_oradores[leg].keys())
        textos_o = [textos_oradores[leg][o] for o in nombres_o]
        nombres_o_validos, sim_matrix_o = calcular_bloque_bm25_representativo(nombres_o, textos_o, stop_es)

        nombres_p = list(textos_partidos[leg].keys())
        textos_p = [textos_partidos[leg][p] for p in nombres_p]
        nombres_p_validos, sim_matrix_p = calcular_bloque_bm25_representativo(nombres_p, textos_p, stop_es)

        resultados_finales["por_legislatura"][leg] = {
            "partidos": [
                {
                    "nombre": nombres_p_validos[i],
                    "similitudes": sorted([
                        {"con": nombres_p_validos[j], "score": float(sim_matrix_p[i, j])}
                        for j in range(len(nombres_p_validos)) if i != j
                    ], key=lambda x: x["score"], reverse=True),
                } for i in range(len(nombres_p_validos))
            ],
            "oradores": [
                {
                    "nombre": nombres_o_validos[i],
                    "partido": meta_orador_partido[leg].get(nombres_o_validos[i], "Desconocido"),
                    "similares": sorted([
                        {
                            "nombre": nombres_o_validos[j],
                            "partido": meta_orador_partido[leg].get(nombres_o_validos[j], "Desconocido"),
                            "score": float(sim_matrix_o[i, j]),
                        }
                        for j in range(len(nombres_o_validos)) if i != j
                    ], key=lambda x: x["score"], reverse=True),
                } for i in range(len(nombres_o_validos))
            ],
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resultados_finales, f, ensure_ascii=False, indent=2)
    print(f"Saved {output_path}")


def get_discourse_embedding(sentences: list[str], model, window_size: int = 3, batch_size: int = 32):
    """Embed a whole discourse (list of sentences) as a single normalized vector.

    Groups sentences into sliding windows to preserve some local context,
    embeds each window, mean-pools across all windows, and L2-normalizes
    the result.
    """
    if not sentences:
        return None

    chunks = [" ".join(sentences[i:i + window_size]) for i in range(len(sentences))]
    chunk_embeddings = model.encode(chunks, batch_size=batch_size, convert_to_tensor=True)
    mean_emb = torch.mean(chunk_embeddings, dim=0)
    norm = torch.linalg.vector_norm(mean_emb, ord=2)

    return mean_emb / norm


def embeddings_similarity_analysis(
    json_path: str = STANZA_RESULTS_PATH,
    output_path: str = EMBEDDINGS_OUTPUT_PATH,
    model_name: str = EMBEDDING_MODEL,
) -> None:
    """Compute semantic (sentence-embedding) similarity between speakers and between parties, per term."""
    model = SentenceTransformer(model_name)

    intervenciones_data = []

    with open(json_path, "rb") as f:
        parser = ijson.items(f, "item")
        for item in parser:
            raw_sentences = [s["sentence_text"] for s in item.get("sentences", []) if "sentence_text" in s]
            if not raw_sentences:
                continue

            emb = get_discourse_embedding(raw_sentences, model)

            if emb is not None:
                intervenciones_data.append({
                    "ORADOR": item.get("ORADOR"),
                    "PARTIDO": item.get("PARTIDO"),
                    "LEGISLATURA": item.get("LEGISLATURA"),
                    "embedding": emb,
                })

    legislaturas = defaultdict(list)
    for d in intervenciones_data:
        legislaturas[d["LEGISLATURA"]].append(d)

    resultados_finales = {"por_legislatura": {}}

    for leg, docs in legislaturas.items():
        print(f"Term: {leg}")

        # speaker vs speaker: average each speaker's interventions
        oradores_dict = defaultdict(list)
        for d in docs:
            oradores_dict[d["ORADOR"]].append(d)

        meta_oradores, embs_oradores = [], []
        for nombre, lista in oradores_dict.items():
            mean_v = torch.mean(torch.stack([x["embedding"] for x in lista]), dim=0)
            mean_v = mean_v / torch.linalg.vector_norm(mean_v, ord=2)

            partidos = [x["PARTIDO"] for x in lista]
            partido = partidos[0] if len(set(partidos)) == 1 else Counter(partidos).most_common(1)[0][0]

            meta_oradores.append({"nombre": nombre, "partido": partido})
            embs_oradores.append(mean_v)

        matrix_o = torch.stack(embs_oradores)

        # party vs party: average all interventions in the party
        partidos_dict = defaultdict(list)
        for d in docs:
            partidos_dict[d["PARTIDO"]].append(d["embedding"])

        meta_partidos, embs_partidos = [], []
        for partido, lista_v in partidos_dict.items():
            mean_p = torch.mean(torch.stack(lista_v), dim=0)
            mean_p = mean_p / torch.linalg.vector_norm(mean_p, ord=2)
            meta_partidos.append(partido)
            embs_partidos.append(mean_p)

        matrix_p = torch.stack(embs_partidos)

        reducer_o = umap.UMAP(n_components=2, metric="cosine")
        coords_o = reducer_o.fit_transform(matrix_o.cpu().numpy())

        reducer_p = umap.UMAP(n_components=2, metric="cosine")
        coords_p = reducer_p.fit_transform(matrix_p.cpu().numpy())

        resultados_finales["por_legislatura"][leg] = {
            "partidos": [
                {
                    "nombre": p,
                    "coords": {"x": float(coords_p[i, 0]), "y": float(coords_p[i, 1])},
                    "similitudes": sorted([
                        {"con": meta_partidos[j], "score": float(cosine_similarity(matrix_p[i].unsqueeze(0), matrix_p[j].unsqueeze(0)))}
                        for j in range(len(meta_partidos)) if i != j
                    ], key=lambda x: x["score"], reverse=True),
                } for i, p in enumerate(meta_partidos)
            ],
            "oradores": [
                {
                    "nombre": o["nombre"],
                    "partido": o["partido"],
                    "coords": {"x": float(coords_o[i, 0]), "y": float(coords_o[i, 1])},
                    "similares": sorted([
                        {"nombre": meta_oradores[j]["nombre"], "partido": meta_oradores[j]["partido"], "score": float(cosine_similarity(matrix_o[i].unsqueeze(0), matrix_o[j].unsqueeze(0)))}
                        for j in range(len(meta_oradores)) if i != j
                    ], key=lambda x: x["score"], reverse=True),
                } for i, o in enumerate(meta_oradores)
            ],
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resultados_finales, f, ensure_ascii=False, indent=2)
    print(f"Saved {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["bm25", "embeddings", "all"],
        default="all",
        help="which similarity method to run: bm25, embeddings, or all (default)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    methods = ["bm25", "embeddings"] if args.method == "all" else [args.method]

    if "bm25" in methods:
        bm25_similarity_analysis()
    if "embeddings" in methods:
        embeddings_similarity_analysis()