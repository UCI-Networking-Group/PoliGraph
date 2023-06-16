import argparse

import numpy as np
import pandas as pd
from datasets import Dataset
from setfit import SetFitModel
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
    parser.add_argument("model", help="Model checkpoint")
    parser.add_argument("test_data", help="JSONL file of test data")
    args = parser.parse_args()

    model = SetFitModel.from_pretrained(args.model)
    test_dataset = Dataset.from_json(args.test_data, keep_in_memory=True)

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
    print("Macro precision:", np.mean(precisions))
    print("Macro recall:", np.mean(recalls))


if __name__ == "__main__":
    main()
