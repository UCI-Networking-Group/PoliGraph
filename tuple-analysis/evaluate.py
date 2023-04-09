#!/usr/bin/env python3

import argparse
import csv
import logging


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth_csv", help="CSV containing ground truth annotations")
    parser.add_argument("prediction_csv", help="CSV containing predictions")
    args = parser.parse_args()

    app_tuple_annotations = {}

    with open(args.ground_truth_csv, encoding="utf-8", newline="") as fin:
        for row in csv.DictReader(fin):
            app_tuple = (row["app_id"], row["entity"], row["datatype"])

            if app_tuple in app_tuple_annotations:
                logging.warning("Duplicate ground truth annotation: %r", ",".join(app_tuple))

            app_tuple_annotations[app_tuple] = bool(int(row["manual"]))

    with open(args.prediction_csv, encoding="utf-8", newline="") as fin:
        true_positives = set()
        false_positives = set()

        for row in csv.DictReader(fin):
            app_tuple = (row["app_id"], row["entity"], row["datatype"])

            if app_tuple not in app_tuple_annotations:
                logging.warning("Missing annotation: %r", ",".join(app_tuple))
                continue

            if app_tuple_annotations[app_tuple]:
                true_positives.add(app_tuple)
            else:
                false_positives.add(app_tuple)
                print("FP:", ",".join(app_tuple))

        false_negatives = set()

        for app_tuple, state in app_tuple_annotations.items():
            if state and app_tuple not in true_positives:
                false_negatives.add(app_tuple)

    recall = len(true_positives) / (len(true_positives) + len(false_negatives))
    precision = len(true_positives) / (len(true_positives) + len(false_positives))

    print(f"precision = {precision * 100:5.2f}%")
    print(f"   recall = {recall * 100:5.2f}%")

if __name__ == "__main__":
    main()
