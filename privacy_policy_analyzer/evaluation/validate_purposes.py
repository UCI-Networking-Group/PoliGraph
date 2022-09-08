import argparse
import random
import os
from collections import Counter
import networkx as nx
import csv

from privacy_policy_analyzer.graph_utils import gml_destringizer


def load_purposes(workdir):
    kgraph_path = os.path.join(workdir, 'graph_trimmed.gml')
    graph = nx.read_gml(kgraph_path, destringizer=gml_destringizer)

    phrase2purposes =  {}

    for _, _, rel, edge_data in graph.edges(keys=True, data=True):
        if rel == "COLLECT":
            for label, text in edge_data["purposes"]:
                if text not in phrase2purposes:
                    phrase2purposes[text] = set()

                phrase2purposes[text].add(label)

    return phrase2purposes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Input CSV file")
    parser.add_argument("rootdir", help="Root of workdirs")
    args = parser.parse_args()

    kgraph_purposes = {}
    manual_purposes = {}

    with open(args.input_csv, encoding="utf-8") as fin:
        for row in csv.DictReader(fin):
            label = row["purpose_label"]
            phrase = row["text"]

            if label == "":
                continue

            policy_id = row["policy_id"]

            if policy_id not in kgraph_purposes:
                manual_purposes[policy_id] = {}
                kgraph_purposes[policy_id] = load_purposes(os.path.join(args.rootdir, policy_id))

            if phrase not in kgraph_purposes[policy_id]:
                print(f"WARNING: purpose phrase {phrase} not found in policy {policy_id}")
                continue

            if phrase not in manual_purposes[policy_id]:
                manual_purposes[policy_id][phrase] = set()

            if label != "none":
                manual_purposes[policy_id][phrase].add(label)

    tp = Counter()
    fp = Counter()
    fn = Counter()

    for policy_id in manual_purposes:
        for phrase in manual_purposes[policy_id].keys():
            ground_truth = manual_purposes[policy_id][phrase]
            prediction = kgraph_purposes[policy_id][phrase]

            tp.update(prediction & ground_truth)
            fp.update(prediction - ground_truth)
            fn.update(ground_truth - prediction)

    print(tp)
    print(fp)
    print(fn)

    tp_total = sum(tp.values())
    fp_total = sum(fp.values())
    fn_total = sum(fn.values())

    for label in "services", "advertising", "security", "analytics", "legal":
        class_precision = tp[label] / (tp[label] + fp[label])
        class_recall = tp[label] / (tp[label] + fn[label])
        print(f"{label}: precision={class_precision:.3}, recall={class_recall:.3}")

    micro_precision = tp_total / (tp_total + fp_total)
    micro_recall = tp_total / (tp_total + fn_total)
    print("micro precision =", micro_precision)
    print("micro recall =", micro_recall)


if __name__ == "__main__":
    main()
