"""
Stanza Linguistic Preprocessing and Language Filtering Pipeline

Tokenizes parliamentary interventions into sentences and performs full linguistic 
analysis (PoS tagging, lemmatization, dependency parsing) using Stanza. Filters 
out non-Spanish sentences from parliamentary interventions using language detection.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
import stanza
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_json_data(folder_path: str) -> List[dict]:
    data = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".json"):
                json_path = os.path.join(root, file_name)
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)
                        data.extend([item for item in json_data if "TEXTO" in item])
                except json.JSONDecodeError:
                    print(f"Error al leer el archivo '{file_name}'.")
        print(f"Datos leídos de {root}")
    return data


def idiom_detect_batch(tokenizer, model, sentences: List[str], use_gpu: bool = False, batch_size: int = 32) -> List[str]:
    """Classify the language of each sentence in batches."""

    idiomas = []
    i = 0

    while i < len(sentences):
        batch = sentences[i:i + batch_size]
        try:
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            if use_gpu:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits
                preds = logits.argmax(dim=1).cpu().tolist()
                idiomas.extend([model.config.id2label[p] for p in preds])

            i += batch_size
            torch.cuda.empty_cache()

            if not use_gpu and torch.cuda.is_available():
                try:
                    model.to("cuda")
                    use_gpu = True
                    print("GPU available again.")
                except RuntimeError:
                    pass  # still out of memory, keep running on CPU

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size > 2:
                batch_size = max(2, batch_size // 2)
                print(f"Reducing batch_size to {batch_size}.")
            else:
                print("Falling back to CPU temporarily.")
                use_gpu = False
                model.to("cpu")
            continue

    return idiomas


def _get_sentences(entries: List[dict]) -> List[dict]:
    """Executes Stanza NLP processing (PoS, lemmas, parsing) and filters out 
    non-Spanish sentences.

    Args:
        entries: A batch of raw entries, each expected to have a "TEXTO" field.

    Returns:
        List[dict]: Entries containing at least one Spanish sentence along with 
        structured linguistic annotations.
    """

    try:
        pid = os.getpid()
        print(f"[PID {pid}] Starting processing of {len(entries)} entries.")
        texts = [entry["TEXTO"].strip() for entry in entries if entry.get("TEXTO")]

        if not texts:
            print(f"[PID {pid}] No valid texts.")
            return []

        use_gpu = torch.cuda.is_available()

        pipe = stanza.Pipeline(
            lang="es",
            processors="tokenize,pos,lemma,depparse",
            use_gpu=use_gpu,
            download_method=stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
        )

        pairs = [
            (entry, entry["TEXTO"].strip())
            for entry in entries
            if entry.get("TEXTO") and entry["TEXTO"].strip()
        ]
        entries_filt, texts = zip(*pairs)

        docs = pipe.bulk_process(texts)

        # Language-detection model.
        model_name = "ERCDiDip/40_langdetect_v01"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        if use_gpu:
            model.to("cuda")

        results = []

        for i, (entry, doc) in enumerate(zip(entries_filt, docs)):
            sentences = [s.text.strip() for s in doc.sentences if s.text.strip()]
            if not sentences:
                continue

            idiomas = idiom_detect_batch(tokenizer, model, sentences, use_gpu=use_gpu, batch_size=16)

            text_sens = []
            for sen, lang in zip(doc.sentences, idiomas):
                if lang != "es":
                    continue
                text_sens.append({
                    "sentence_text": sen.text,
                    "tokens": [t.text for t in sen.tokens],
                    "words": [{
                        "token_id": w.id,
                        "token_text": w.text,
                        "head": w.head,
                        "deprel": w.deprel,
                        "upos": w.upos,
                        "lemma": w.lemma,
                    } for w in sen.words],
                })

            if text_sens:
                result_entry = {k: v for k, v in entry.items() if k != "TEXTO"}
                result_entry["sentences"] = text_sens
                results.append(result_entry)

            if (i + 1) % 10 == 0:
                print(f"[PID {pid}] Processed {i + 1}/{len(entries)} texts...")

            torch.cuda.empty_cache()

        print(f"[PID {pid}] Done. Total: {len(results)} entries processed.")
        return results

    except Exception as e:
        print(f"[PID {os.getpid()}] Error during processing: {e}")
        torch.cuda.empty_cache()
        return []


def get_sentences(data: List[dict], num_workers: int = os.cpu_count(), batch_size: str = 15) -> List[dict]:
    """Executes sentence extraction and filtering in parallel across worker processes."""
    
    seen_ids = set()
    sens = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(_get_sentences, data[i:i + batch_size])
            for i in range(0, len(data), batch_size)
        ]

        for f in as_completed(futures):
            try:
                results = f.result()
            except Exception as e:
                print("Batch failed:", e)
                continue

            for r in results:
                rid = r["ID"]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    sens.append(r)

            print("Batch processed successfully.")

    return sens


def data_preparation(data: List[dict], output_dir: str, num_workers: int = os.cpu_count(), stanza_results_path: str = None) -> None:
    """Generates or loads precomputed Stanza processing results for the dataset.

    Loads existing JSON results if available; otherwise triggers sentence
    extraction and writes output to 'stanza_results.json'.
    """

    os.makedirs(output_dir, exist_ok=True)

    if stanza_results_path and os.path.exists(stanza_results_path):
        with open(stanza_results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Loaded existing results from {stanza_results_path}")
    else:
        results = get_sentences(data, num_workers)

    output_file = os.path.join(output_dir, "stanza_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    path = os.path.join(".", "data")
    data = load_json_data(path)

    data_preparation(
        data=data,
        num_workers=2,
        output_dir="results/stanza",
        stanza_results_path="results/stanza/stanza_results.json",
    )