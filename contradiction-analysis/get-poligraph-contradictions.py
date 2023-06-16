#!/usr/bin/env python3

import argparse
import csv
import logging
import os

from poligrapher.annotators import CollectionAnnotator
from poligrapher.graph_utils import KGraph


def check_subsum_conflict(kgraph: KGraph, pos_term, neg_term):
    if pos_term == neg_term:
        return True
    elif pos_term != neg_term and kgraph.subsum(pos_term, neg_term):
        # A narrow definition as defined in PolicyLint
        return False

    neg_term_children = set(kgraph.descendants(neg_term))
    neg_term_children.add(neg_term)

    pos_term_children = set(kgraph.descendants(pos_term))
    pos_term_children.add(pos_term)

    return bool(neg_term_children & pos_term_children)


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    with open(args.output, "w", encoding="utf-8", newline="") as fout:
        csv_writer = csv.DictWriter(fout, ["app_id", "pos_edge", "neg_edge", "pos_text", "neg_text"])
        csv_writer.writeheader()

        for workdir in args.workdirs:
            app_id = os.path.basename(os.path.realpath(workdir))
            kgraph = KGraph(os.path.join(workdir, "graph.yml"))

            negative_edges = []
            positive_edges = []

            for entity, dtype, key in kgraph.positive_kgraph.edges(keys=True):
                if key in CollectionAnnotator.POSITIVE_EDGE_TYPES:
                    positive_edges.append((entity, dtype, key))

            for entity, dtype, key in kgraph.negative_kgraph.edges(keys=True):
                if key in CollectionAnnotator.NEGATIVE_EDGE_TYPES:
                    negative_edges.append((entity, dtype, key))

            for neg_e, neg_d, neg_action in negative_edges:
                for pos_e, pos_d, pos_action in positive_edges:
                    # Check if action conflict
                    if not pos_action == neg_action.split("_", 1)[1]:
                        continue

                    pos_data = kgraph.positive_kgraph.get_edge_data(pos_e, pos_d, pos_action)
                    neg_data = kgraph.negative_kgraph.get_edge_data(neg_e, neg_d, neg_action)

                    pos_purposes = set(pos_data['purposes'])
                    neg_purposes = set(neg_data['purposes'])

                    # Check if purposes conflict
                    if neg_purposes and not(neg_purposes & pos_purposes):
                        continue

                    # Check if data types / entities conflict
                    dtype_conflict = check_subsum_conflict(kgraph, pos_d, neg_d)
                    entity_conflict = check_subsum_conflict(kgraph, pos_e, neg_e)

                    if not(dtype_conflict and entity_conflict):
                        continue

                    csv_writer.writerow({
                        "app_id": app_id,
                        "pos_edge": f"{pos_action}({pos_e}, {pos_d})",
                        "neg_edge": f"{neg_action}({neg_e}, {neg_d})",
                        "pos_text": "\n".join(pos_data["text"]),
                        "neg_text": "\n".join(neg_data["text"]),
                    })



if __name__ == "__main__":
    main()
