"""
Loads the per-intervention logits produced by 'sentiment_analysis.py',
combines them into a single dataframe, and converts the raw logits into
task-specific probabilities and predicted labels (sentiment, hate speech,
or emotion). The result is merged (by id) with the LDA topic dataframe 
and written to a Parquet file.

Usage:
    python src/stage3_semantics/sentiment_mapping.py --task sentiment
    python src/stage3_semantics/sentiment_mapping.py                   # runs all three tasks
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch

LDA_PATH = "results/lda/ldamulticore/ldamulticore_45asymmetric.pt"

# Label set, logits directory, and output Parquet path for each classification task.
TASK_CONFIGS = {
    "sentiment": {
        "labels": ["NEG", "NEU", "POS"],
        "logits_path": "results/sentiment_analysis/logits",
        "output_path": "results/sentiment_analysis/sentiment_df.parquet",
    },
    "hate_speech": {
        "labels": ["hateful", "targeted", "aggressive"],
        "logits_path": "results/hate_speech_analysis/logits",
        "output_path": "results/hate_speech_analysis/hate_speech_df.parquet",
    },
    "emotion": {
        "labels": ["others", "joy", "sadness", "anger", "surprise", "disgust", "fear"],
        "logits_path": "results/emotion_analysis/logits",
        "output_path": "results/emotion_analysis/emotion_df.parquet",
    },
}


def attitude_analysis(
    df,
    task="sentiment",
    logits_path="results/sentiment_analysis/logits",
    labels=("NEG", "NEU", "POS"),
    output_path="sentiment_df.parquet",
):
    """Convert saved logits into predictions and merge them with a dataframe.
 
    Reads every '.pt' file in 'logits_path' (one file per intervention,
    each containing the logits for all of its sentences), stacks them into
    a single dataframe with one row per sentence, and applies the
    appropriate transformation for the given task:
 
    - 'sentiment' and 'emotion': softmax over the labels, plus the argmax label.
    - 'hate_speech': independent sigmoid per label, plus the set of
        labels whose probability exceeds 0.4."""

    logits_list = []
    ids = []
    sent_indices = []

    for f in sorted(os.listdir(logits_path)):
        data = torch.load(os.path.join(logits_path, f), map_location="cpu")
        logits = data["logits"].numpy()
        entry_id = data["id"]

        logits_list.append(logits)
        ids.extend([entry_id] * logits.shape[0])
        sent_indices.extend(range(logits.shape[0]))

    logits_array = np.vstack(logits_list)
    logits_df = pd.DataFrame(logits_array, columns=labels)
    logits_df["id"] = ids
    logits_df["sent_index"] = sent_indices

    if task == "sentiment":
        logits_tensor = torch.tensor(logits_df[labels].values, dtype=torch.float32)
        logits_df[labels] = torch.nn.functional.softmax(logits_tensor, dim=1).numpy()
        logits_df["sentiment_pred"] = logits_df[labels].idxmax(axis=1)

    elif task == "hate_speech":
        logits_tensor = torch.tensor(logits_df[labels].values, dtype=torch.float32)
        logits_df[labels] = torch.sigmoid(logits_tensor).numpy()
        logits_df["hate_pred"] = logits_df[labels].apply(
            lambda row: [label for label in labels if row[label] > 0.4], axis=1
        )

    elif task == "emotion":
        logits_tensor = torch.tensor(logits_df[labels].values, dtype=torch.float32)
        logits_df[labels] = torch.nn.functional.softmax(logits_tensor, dim=1).numpy()
        logits_df["emotion_pred"] = logits_df[labels].idxmax(axis=1)

    df_full = df.merge(logits_df, on="id", how="left")
    df_full.to_parquet(output_path, index=False)
    return df_full


def run_task(task: str, df, lda_path: str = LDA_PATH) -> None:
    cfg = TASK_CONFIGS[task]
    print(f"\n--- {task} ---")
    df_full = attitude_analysis(
        df,
        task=task,
        logits_path=cfg["logits_path"],
        labels=cfg["labels"],
        output_path=cfg["output_path"],
    )
    print(f"Guardado {cfg['output_path']}.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=list(TASK_CONFIGS.keys()) + ["all"],
        default="all",
        help="which task's logits to process: sentiment, hate_speech, emotion, or all (default)",
    )
    parser.add_argument("--lda-path", default=LDA_PATH, help="path to the LDA .pt file with the base df")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    obj = torch.load(args.lda_path, weights_only=False)
    df = pd.DataFrame(obj["df"])

    tasks = list(TASK_CONFIGS.keys()) if args.task == "all" else [args.task]
    for task in tasks:
        run_task(task, df, lda_path=args.lda_path)