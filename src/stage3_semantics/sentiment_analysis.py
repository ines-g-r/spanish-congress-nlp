"""
Sequence Classification and Embedding Extraction

Embeds sentences and runs them through a pretrained classifier, for three
different tasks: emotion, hate speech and sentiment.

Inputs:
    A JSON file containing parsed textual interventions (e.g., from Stanza).
Outputs:
    PyTorch tensor files (.pt) containing the embeddings and a dictionary
    with logits and label mappings for each entry.

Usage:
    python src/stage3_semantics/sentiment_analysis --task emotion
    python src/stage3_semantics/sentiment_analysis --task hate_speech
    python src/stage3_semantics/sentiment_analysis --task sentiment
    python src/stage3_semantics/sentiment_analysis --task all          # Runs all tasks sequentially
"""

import argparse
import os
from typing import List, Optional
import ijson
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# Pretrained checkpoints and output directories for each task.
# Hate speech and sentiment use device_map='auto' by default.
# Emotion is mapped manually to the available device.
TASK_CONFIGS = {
    "emotion": {
        "pretrained": "finiteautomata/beto-emotion-analysis",
        "device_map": None,
        "output_path1": "results/emotion_analysis/embed",
        "output_path2": "results/emotion_analysis/logits",
    },
    "hate_speech": {
        "pretrained": "pysentimiento/robertuito-hate-speech",
        "device_map": "auto",
        "output_path1": "results/hate_speech_analysis/embed",
        "output_path2": "results/hate_speech_analysis/logits",
    },
    "sentiment": {
        "pretrained": "pysentimiento/robertuito-sentiment-analysis",
        "device_map": "auto",
        "output_path1": "results/sentiment_analysis/embed",
        "output_path2": "results/sentiment_analysis/logits",
    },
}

DEFAULT_JSON_PATH = "results/stanza/stanza_results.json"


class SequenceEmbedder(nn.Module):
    """
    Wraps a pretrained classification model to extract embeddings and logits.
    
    Provides utilities to process text directly from JSON files and save the 
    resulting PyTorch tensors to disk.
    """

    def __init__(self, pretrained: str, device_map: Optional[str] = None):
        super().__init__()
        self.pretrained = pretrained
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

        model_kwargs = {"output_hidden_states": True}
        if device_map is not None:
            model_kwargs["device_map"] = device_map
            
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained, **model_kwargs
        ).eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_map is None:
            self.model.to(self.device)

        config = AutoConfig.from_pretrained(pretrained)
        self.dim = config.hidden_size
        self.max_len = config.max_position_embeddings
        self.num_labels = config.num_labels

    def forward(self, sentences: List[str]) -> torch.Tensor:
        """
        Processes a batch of sentences and extracts embeddings and logits.
        
        Args:
            sentences: List of text strings.  
        Returns:
            A tuple of (embeddings, logits) for the input batch.
        """
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embed = (
            outputs.hidden_states[-1][:, 0]
            if outputs.hidden_states
            else outputs.last_hidden_state[:, 0]
        )
        logits = outputs.logits
        
        return embed, logits

    def process_json_and_save(
        self,
        json_path: str,
        output_path_embed: str,
        output_path_logits: str,
        batch_size: int = 10,
    ) -> None:
        """
        Iterates through the Stanza JSON results, embedding each intervention 
        individually and saving the outputs to disk.
        """
        os.makedirs(output_path_embed, exist_ok=True)
        os.makedirs(output_path_logits, exist_ok=True)

        with open(json_path, "r", encoding="utf-8") as f:
            for entry in ijson.items(f, "item"):
                # Extract valid sentences
                sentences = [
                    sen.get("sentence_text", "").strip()
                    for sen in entry.get("sentences", [])
                    if sen.get("sentence_text", "").strip()
                ]
                if not sentences:
                    continue

                emb_batches, log_batches = [], []
                
                # Process in batches to manage memory
                for i in range(0, len(sentences), batch_size):
                    batch = sentences[i : i + batch_size]
                    batch_emb, batch_log = self.forward(batch)
                    
                    emb_batches.append(batch_emb.cpu())
                    log_batches.append(batch_log.cpu())
                    
                    del batch_emb, batch_log
                    torch.cuda.empty_cache()

                emb = torch.cat(emb_batches, dim=0)
                log = torch.cat(log_batches, dim=0)

                filename = f"{entry['ID']}.pt"

                # Sanity check to ensure tensor dimensions are correct before saving
                is_valid = (
                    emb.ndim == 2
                    and log.ndim == 2
                    and emb.size(0) == log.size(0)
                    and emb.size(1) == self.dim
                    and log.size(1) == self.num_labels
                    and not torch.isnan(emb).any()
                    and not torch.isnan(log).any()
                )
                
                if is_valid:
                    torch.save(emb, os.path.join(output_path_embed, filename))
                    torch.save(
                        {
                            "id": entry["ID"],
                            "logits": log,
                            "labels": {
                                i: self.model.config.id2label[i]
                                for i in range(self.num_labels)
                            },
                        },
                        os.path.join(output_path_logits, filename),
                    )
                    print(f"Saved {filename} with {emb.size(0)} sentences.")
                else:
                    print(
                        f"Warning: Invalid tensor in {filename} | "
                        f"emb_shape={emb.shape}, log_shape={log.shape}"
                    )


def run_task(task: str, json_path: str = DEFAULT_JSON_PATH, batch_size: int = 16) -> None:
    """Initializes the model for a specific task and processes the JSON dataset."""
    cfg = TASK_CONFIGS[task]
    print(f"\n--- Initializing Task: {task} ({cfg['pretrained']}) ---")
    
    embedder = SequenceEmbedder(
        pretrained=cfg["pretrained"], 
        device_map=cfg["device_map"]
    )
    
    embedder.process_json_and_save(
        json_path=json_path,
        output_path_embed=cfg["output_path1"],
        output_path_logits=cfg["output_path2"],
        batch_size=batch_size,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract embeddings and logits using HuggingFace models."
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_CONFIGS.keys()) + ["all"],
        default="all",
        help="Specify the model task to run: 'emotion', 'hate_speech', 'sentiment', or 'all' (default).",
    )
    parser.add_argument(
        "--json-path", 
        default=DEFAULT_JSON_PATH, 
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=16, 
        help="Number of sentences per forward pass."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tasks_to_run = list(TASK_CONFIGS.keys()) if args.task == "all" else [args.task]
    
    for task_name in tasks_to_run:
        run_task(task_name, json_path=args.json_path, batch_size=args.batch_size)