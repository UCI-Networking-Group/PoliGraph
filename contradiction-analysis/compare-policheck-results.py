#!/usr/bin/env python3

import argparse
import csv
import itertools
import logging
import functools
import os
import re
import networkx as nx

from rapidfuzz import fuzz, process

from privacy_policy_analyzer.annotators import CollectionAnnotator
from privacy_policy_analyzer.graph_utils import KGraph, yaml_load_graph


def get_purposes_in_sentence(graph, text, edge_tuple):
    edge_data = graph.edges[edge_tuple]

    for purpose, phrase_list in edge_data['purposes'].items():
        if any(phrase in text for phrase in phrase_list):
            yield purpose
            break


class EdgeMatcher:
    def __init__(self, graph):
        self.graph = graph
        self.text_to_collect_edges = text_to_collect_edges = {}
        self.text_to_subsum_edges = text_to_subsum_edges = {}

        for u, v, relationship, data in graph.edges(keys=True, data=True):
            for text in data["text"]:
                if relationship in CollectionAnnotator.EDGE_TYPES:
                    text_to_collect_edges.setdefault(text, [])
                    text_to_collect_edges[text].append((u, v, relationship))
                elif relationship == "SUBSUM":
                    text_to_subsum_edges.setdefault(text, [])
                    text_to_subsum_edges[text].append((u, v))

    @functools.cache
    def get_edges(self, sentence):
        def _get_edges(text_to_edges):
            patial_matches = process.extract(sentence, text_to_edges.keys(),
                                             scorer=fuzz.partial_ratio, score_cutoff=90.0)

            if len(patial_matches) > 0:
                poligraph_sent, *_ = process.extractOne(sentence, [m[0] for m in patial_matches], scorer=fuzz.WRatio)
                return text_to_edges[poligraph_sent]
            else:
                return []

        collect_edges = _get_edges(self.text_to_collect_edges)
        subsum_edges = _get_edges(self.text_to_subsum_edges)

        mini_ontology = nx.DiGraph()
        mini_ontology.add_edges_from(subsum_edges)

        expanded_collect_edges = set()

        for u, v, rel in collect_edges:
            edge_data = self.graph.edges[u, v, rel]
            purposes_set = set()

            for purpose, phrase_list in edge_data['purposes'].items():
                if any(phrase in sentence for phrase in phrase_list):
                    purposes_set.add(purpose)

            purposes = frozenset(purposes_set)

            mini_ontology.add_node(u)
            mini_ontology.add_node(v)
            expanded_collect_edges.add((u, v, rel, purposes))

            for su, sv in itertools.product(nx.descendants(mini_ontology, u), nx.descendants(mini_ontology, v)):
                expanded_collect_edges.add((su, sv, rel, purposes))

        return sorted(expanded_collect_edges)




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

    term_regex = re.compile(r'^(?P<term>.+?)(?: @(?P<subject>\S+))?(?: (?P<src>\(\d+, \d+\)))?$')

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
            edge_matcher = EdgeMatcher(graph)

            for (pos_sent, neg_sent), tuple_pairs in contradiction_info.items():
                pos_edges = edge_matcher.get_edges(pos_sent)
                neg_edges = edge_matcher.get_edges(neg_sent)

                pos_edges = list(filter(lambda t: not t[2].startswith("NOT_"), pos_edges))
                neg_edges = list(filter(lambda t: t[2].startswith("NOT_"), neg_edges))

                results = []
                conflicted_edge_pairs = []
                weak_conflicted_edge_pairs = []

                # Find possible PoliGraph edges that match the contradicted tuples
                for pos_tuple, neg_tuple in itertools.product(pos_edges, neg_edges):
                    pos_e, pos_d, *_ = pos_tuple
                    neg_e, neg_d, *_ = neg_tuple

                    pos_d = term_regex.match(pos_d)['term']
                    neg_d = term_regex.match(neg_d)['term']
                    pos_e = term_regex.match(pos_e)['term']
                    neg_e = term_regex.match(neg_e)['term']

                    pos_d_is_first_party = bool(kgraph.subsum("we", pos_e))
                    neg_d_is_first_party = bool(kgraph.subsum("we", neg_e))

                    if pos_d_is_first_party and neg_d_is_first_party:
                        e_conflict = any(t1[0] == t2[0] == "we" for t1, t2 in tuple_pairs)
                    elif pos_d_is_first_party or neg_d_is_first_party:
                        e_conflict = False
                    else:
                        e_conflict = (
                            any(not(t1[0] == t2[0] == "we") for t1, t2 in tuple_pairs)
                            and (neg_e == pos_e or kgraph.subsum(neg_e, pos_e) or 'UNSPECIFIC_ACTOR' in (pos_e, neg_e))
                        )

                    d_conflict = neg_d == pos_d or kgraph.subsum(neg_d, pos_d) or 'UNSPECIFIC_DATA' in (pos_d, neg_d)

                    if e_conflict and d_conflict:
                        conflicted_edge_pairs.append((pos_tuple, neg_tuple))
                    elif e_conflict:
                        weak_conflicted_edge_pairs.append((pos_tuple, neg_tuple))

                if len(conflicted_edge_pairs) == 0:
                    if len(weak_conflicted_edge_pairs) == 0:
                        results.append("INVALID")
                    else:
                        # May fail to match because of different term normalization.
                        # Fallback to match entity only
                        conflicted_edge_pairs = weak_conflicted_edge_pairs

                if len(results) == 0:
                    # Check action / subjects / purposes
                    diff_action_flags = []
                    diff_subject_flags = []
                    diff_purpose_flags = []

                    for pos_tuple, neg_tuple in conflicted_edge_pairs:
                        pos_e, pos_d, pos_a, pos_purposes = pos_tuple
                        neg_e, neg_d, neg_a, neg_purposes = neg_tuple

                        diff_action_flags.append(pos_a != neg_a.split("_", 1)[-1])
                        diff_subject_flags.append((pos_d.split("@", 1) + [""])[1] != (neg_d.split("@", 1) + [""])[1])
                        diff_purpose_flags.append(len(neg_purposes) > 0 and len(pos_purposes & neg_purposes) == 0)

                    if len(diff_action_flags) > 0 and all(diff_action_flags):
                        results.append("DIFF_ACTIONS")

                    if len(diff_subject_flags) > 0 and all(diff_subject_flags):
                        results.append("DIFF_SUBJECTS")

                    if len(diff_purpose_flags) > 0 and all(diff_purpose_flags):
                        results.append("DIFF_PURPOSES")

                if len(results) == 0:
                    for (pos_e, pos_d, *_), (neg_e, neg_d, *_) in conflicted_edge_pairs:
                        if kgraph.subsum(neg_e, pos_e) and kgraph.subsum(neg_d, pos_d):
                            break
                    else:
                        if all(pos_e != neg_e or pos_d != neg_d for (pos_e, pos_d), (neg_e, neg_d) in tuple_pairs):
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
