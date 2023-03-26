#!/usr/bin/env python3
"""Train the SetFit model for purpose classification"""

import argparse

import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import precision_recall_fscore_support

PURPOSE_LABELS = [
    "advertising",
    "analytics",
    "legal",
    "security",
    "services",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dataset", help="Training set")
    parser.add_argument("test_dataset", help="Testing set")
    parser.add_argument("output", help="Output model path")
    args = parser.parse_args()

    train_dataset = Dataset.from_json(args.train_dataset, keep_in_memory=True)
    test_dataset = Dataset.from_json(args.test_dataset, keep_in_memory=True)


    # Load SetFit model from Hub
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2",
        multi_target_strategy="one-vs-rest"
    )

    # Create trainer
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss_class=CosineSimilarityLoss,
        batch_size=16,
        num_iterations=20,
        num_epochs=2,
    )

    # Train and evaluate!
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    predictions = model(test_dataset["text"])
    y_pred = predictions.numpy()
    y_true = np.array(test_dataset["label"])
    precisions, recalls, fscores, supports = precision_recall_fscore_support(y_true, y_pred)

    statistic = pd.DataFrame.from_dict({
        "label": PURPOSE_LABELS,
        "precision": precisions,
        "recall": recalls,
        "fscore": fscores,
        "support": supports,
    })
    print(statistic)

    model.save_pretrained(args.output, safe_serialization=True)


if __name__ == "__main__":
    main()
