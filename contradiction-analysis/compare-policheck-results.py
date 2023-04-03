#!/usr/bin/env python3

import argparse
import csv
import itertools
import logging
import os

from rapidfuzz import fuzz, process

from privacy_policy_analyzer.annotators import CollectionAnnotator
from privacy_policy_analyzer.graph_utils import KGraph, yaml_load_graph


def get_purposes_in_sentence(graph, text, edge_tuple):
    edge_data = graph.edges[edge_tuple]

    for purpose, phrase_list in edge_data['purposes'].items():
        if any(phrase in text for phrase in phrase_list):
            yield purpose
            break


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("-r", "--policheck-root", help="PoliCheck root dir")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    workdir_map = {}
    policylint_app_contradiction_pairs: dict[str, dict[str, list]] = {}

    for d in args.workdirs:
        app_id = os.path.basename(os.path.realpath(d))
        workdir_map[app_id] = d
        policylint_app_contradiction_pairs[app_id] = {}

    with open(os.path.join(args.policheck_root, "policylint_results.csv"), encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)

        for row in reader:
            app_id = row["packageName"]

            if app_id not in workdir_map or row["contradictionNum"][0] != "C":
                continue

            neg_tuple = (row["policyEntity"], row["policyData"])
            neg_sentences = row["policySentences"].split("||")
            pos_tuple = (row["contradictoryEntity"], row["contradictoryData"])
            pos_sentences = row["contradictionSentences"].split("||")

            if row["policyAction"] == "collect":
                neg_tuple, pos_tuple = pos_tuple, neg_tuple
                neg_sentences, pos_sentences = pos_sentences, neg_sentences

            for pos_sent, neg_sent in itertools.product(pos_sentences, neg_sentences):
                contradiction_pair = (pos_sent, neg_sent)
                tuple_pair = (pos_tuple, neg_tuple)
                contradiction_info = policylint_app_contradiction_pairs[app_id].setdefault(contradiction_pair, [])

                if tuple_pair not in contradiction_info:
                    contradiction_info.append(tuple_pair)


    with open(args.output, "w", encoding="utf-8", newline="") as fout:
        csv_writer = csv.DictWriter(fout, ["app_id", "pos_text", "neg_text", "tuples", "labels"])
        csv_writer.writeheader()

        for app_id, workdir in workdir_map.items():
            contradiction_info = policylint_app_contradiction_pairs[app_id]

            if len(contradiction_info) == 0:
                continue

            with open(os.path.join(workdir, "graph.yml"), encoding="utf-8") as fin:
                graph = yaml_load_graph(fin)

            kgraph = KGraph(os.path.join(workdir, "graph.yml"))

            # Map PoliGraph text to edges
            poligraph_text_to_edges = {}

            for u, v, relationship, data in graph.edges(keys=True, data=True):
                if relationship in CollectionAnnotator.EDGE_TYPES:
                    for text in data["text"]:
                        poligraph_text_to_edges.setdefault(text, [])
                        poligraph_text_to_edges[text].append((u, v, relationship))

            # Map PolicyLint text to PoliGraph text
            text_map = {}

            for sent in set(itertools.chain(*contradiction_info.keys())):
                patial_matches = process.extract(sent, poligraph_text_to_edges.keys(),
                                                scorer=fuzz.partial_ratio, score_cutoff=90.0)

                if len(patial_matches) > 0:
                    text_map[sent], *_ = process.extractOne(sent, [m[0] for m in patial_matches], scorer=fuzz.WRatio)
                else:
                    text_map[sent] = None

            for (pos_sent, neg_sent), tuple_pairs in contradiction_info.items():
                poligraph_pos_sent = text_map[pos_sent]
                poligraph_neg_sent = text_map[neg_sent]
                results = []

                if poligraph_pos_sent is None or poligraph_neg_sent is None:
                    pos_edges = neg_edges = []
                else:
                    pos_edges = poligraph_text_to_edges[poligraph_pos_sent]
                    neg_edges = poligraph_text_to_edges[poligraph_neg_sent]

                    # Discard invalid tuples
                    pos_edges = list(filter(lambda t: not t[2].startswith("NOT_"), pos_edges))
                    neg_edges = list(filter(lambda t: t[2].startswith("NOT_"), neg_edges))

                if len(pos_edges) == 0 or len(neg_edges) == 0:
                    results.append("INVALID")

                if "INVALID" not in results:
                    # Check action
                    diff_action_flags = []

                    for (pos_e, pos_d, pos_a), (neg_e, neg_d, neg_a) in itertools.product(pos_edges, neg_edges):
                        # Do not compare data types here to work around likely different normalization
                        if pos_e == neg_e or ('UNSPECIFIC_ACTOR' in (pos_e, neg_e) and 'we' not in (pos_e, neg_e)):
                            diff_action_flags.append(pos_a != neg_a.split("_", 1)[-1])

                    if len(diff_action_flags) > 0 and all(diff_action_flags):
                        results.append("DIFF_ACTIONS")

                    # Check subject
                    pos_subjects = {(d.split("@", 1) + [""])[1] for _, d, _ in pos_edges}
                    neg_subjects = {(d.split("@", 1) + [""])[1] for _, d, _ in neg_edges}

                    if len(pos_subjects & neg_subjects) == 0:
                        results.append("DIFF_SUBJECTS")

                    # Check purposes
                    pos_purposes = set()
                    for edge_tuple in pos_edges:
                        pos_purposes.update(get_purposes_in_sentence(graph, poligraph_pos_sent, edge_tuple))

                    neg_purposes = set()
                    for edge_tuple in neg_edges:
                        neg_purposes.update(get_purposes_in_sentence(graph, poligraph_neg_sent, edge_tuple))

                    if len(neg_purposes) > 0 and len(pos_purposes & neg_purposes) == 0:
                        results.append("DIFF_PURPOSES")

                if len(results) == 0 and any(pos_tuple != neg_tuple for pos_tuple, neg_tuple in tuple_pairs):
                    for (pos_e, pos_d, _), (neg_e, neg_d, _) in itertools.product(pos_edges, neg_edges):
                        if kgraph.subsum(neg_e, pos_e) and kgraph.subsum(neg_d, pos_d):
                            break
                    else:
                        results.append("ONTOLOGY")

                tuples_text = []
                for (pos_e, pos_d), (neg_e, neg_d) in tuple_pairs:
                    tuples_text.append(f"{pos_e} collect {pos_d}\n{neg_e} not_collect {neg_d}")

                csv_writer.writerow({
                    "app_id": app_id,
                    "pos_text": pos_sent,
                    "neg_text": neg_sent,
                    "tuples": "\n".join(tuples_text),
                    "labels": "\n".join(results)
                })


if __name__ == "__main__":
    main()
