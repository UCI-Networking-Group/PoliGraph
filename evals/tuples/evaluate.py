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

    tuple_count = 0

    with open(args.prediction_csv, encoding="utf-8", newline="") as fin:
        true_positives = set()
        false_positives = set()

        for row in csv.DictReader(fin):
            app_tuple = (row["app_id"], row["entity"], row["datatype"])
            tuple_count += 1

            if app_tuple not in app_tuple_annotations:
                logging.warning("Missing annotation: %r", ",".join(app_tuple))
                false_positives.add(app_tuple)
                continue

            if app_tuple_annotations[app_tuple]:
                true_positives.add(app_tuple)
            else:
                false_positives.add(app_tuple)

        false_negatives = set()

        for app_tuple, state in app_tuple_annotations.items():
            if state and app_tuple not in true_positives:
                false_negatives.add(app_tuple)

    print("# tuples:", tuple_count)

    for entities in (("we", "3rd-party"), ("we",), ("3rd-party",)):
        TP = sum(1 for _ in filter(lambda t: t[1] in entities, true_positives))
        FP = sum(1 for _ in filter(lambda t: t[1] in entities, false_positives))
        FN = sum(1 for _ in filter(lambda t: t[1] in entities, false_negatives))

        recall = TP / (TP + FN)
        precision = TP / (TP + FP)

        print(entities)
        print(f"precision = {precision * 100:4.1f}%")
        print(f"   recall = {recall * 100:4.1f}%")


if __name__ == "__main__":
    main()
