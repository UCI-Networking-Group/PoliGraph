#!/usr/bin/env python3

import argparse
import csv
import itertools
import logging
import os
import re

from rapidfuzz import fuzz, process

from poligrapher.annotators import CollectionAnnotator
from poligrapher.graph_utils import KGraph, yaml_load_graph


def check_subsum_conflict(kgraph: KGraph, pos_term, neg_term):
    if pos_term == neg_term:
        return True

    neg_term_children = set(kgraph.descendants(neg_term))
    neg_term_children.add(neg_term)

    pos_term_children = set(kgraph.descendants(pos_term))
    pos_term_children.add(pos_term)

    return bool(neg_term_children & pos_term_children)


class EdgeMapper:
    _DATA_TERM_MAP = (
        (
            r'^UNSPECIFIED_DATA$',
            {'pii', 'information about you', 'user information'},
        ),
        (
            r'\b(personal|user)\b.*\b(data|datum|information|identifier)\b',
            {'pii', 'information about you', 'user information'},
        ),
        (
            r' \(\d+, \d+\)$',
            {'pii', 'information about you', 'user information'},
        ),
        (
            r'\badvertising id\b',
            {'advertising identifier'},
        ),
        (
            r'\bidentifier\b',
            {'identifier', 'device identifier'},
        ),
        (
            r'\btechnical\b.*\b(data|datum|information)\b',
            {'technical information'}
        ),
        (
            r'\baccount\b.*\b(data|datum|information)\b',
            {'account information'}
        ),
        (
            r'\b(data|datum|information)\b.*\baccount',
            {'account information'}
        ),
        (
            r'\bcustomer\b.*\b(data|datum|information)\b',
            {'customer information'}
        ),
        (
            r'\b(payment|billing)\b',
            {'billing information'}
        ),
        (
            r'\bprofile\b',
            {'profile information'}
        ),
        (
            r'\bdemographic\b',
            {'demographic information'}
        ),
        (
            r'\b(geo)?location\b',
            {'geographical location'}
        ),
        (
            r'\b(usage|activity)\b',
            {'usage information'}
        ),
        (
            r'\b(internet|network)\b.*\b(data|datum|information)\b',
            {'network information'}
        ),
        (
            r'\bdevice\b.*\b(data|datum|information)\b',
            {'device information'}
        ),
        (
            r'\b(data|datum|information)\b.*\b(device|computer)',
            {'device information'}
        ),
        (
            r'\b(non-personal|deidentified)\b.*\b(data|datum|information)\b',
            {'non-pii'}
        ),
    )

    _ENTITY_TERM_MAP = (
        (
            r'^(UNSPECIFIED_ACTOR|service provider|contractor|platform)$',
            {'third party'}
        ),
        (
            r'\b(advertis(er|ing|ment))\b',
            {'advertising network'}
        ),
        (
            r'\bsocial\b.*\b(media|network)\b',
            {'social network'}
        ),
        (
            r'\bpayment\b',
            {'third party payment processor'}
        ),
        (
            r'\bgoogle\b',
            {'google', 'google ads', 'google analytic', 'firebase', 'youtube'}
        )
    )

    DATA_TERM_MAP = [(re.compile(r, re.I), frozenset(s)) for r, s in _DATA_TERM_MAP]
    ENTITY_TERM_MAP = [(re.compile(r, re.I), frozenset(s)) for r, s in _ENTITY_TERM_MAP]

    def __init__(self, workdir):
        with open(os.path.join(workdir, "graph-extended.yml"), encoding="utf-8") as fin:
            self.graph = yaml_load_graph(fin)

        self.kgraph = KGraph(os.path.join(workdir, "graph-extended.yml"))

        self.text_to_collect_edges = text_to_collect_edges = {}

        for u, v, relationship, data in self.graph.edges(keys=True, data=True):
            for text in data["text"]:
                if relationship in CollectionAnnotator.EDGE_TYPES:
                    text_to_collect_edges.setdefault(text, [])
                    text_to_collect_edges[text].append((u, v, relationship))

    def match(self, policylint_tuple, policylint_sentences):
        def test_term_match(term_map, our_term, pl_term):
            if our_term.lower() == pl_term:
                return True

            for regex, term_set in term_map:
                if regex.search(our_term) and pl_term in term_set:
                    return True

            return False

        candidate_edges = set()
        candidate_edge_purposes = {}
        pl_entity, pl_data_type = policylint_tuple

        for s in policylint_sentences:
            patial_matches = process.extract(s, self.text_to_collect_edges.keys(),
                                             scorer=fuzz.partial_ratio, score_cutoff=90.0)
            for key, *_ in patial_matches:
                for u, v, rel in self.text_to_collect_edges[key]:
                    candidate_edge_purposes.setdefault((u, v, rel), set())

                    edge_data = self.graph.get_edge_data(u, v, rel)
                    for purpose, phrases in edge_data["purposes"].items():
                        if any(s in key for s in phrases):
                            candidate_edge_purposes[(u, v, rel)].add(purpose)

                    candidate_edges.add((u, v, rel))

        matched_edges = []

        for edge in candidate_edges:
            root_entity, root_data_type, *_ = edge

            our_data_types = [root_data_type, *self.kgraph.descendants(root_data_type)]
            our_entities = [root_entity, *self.kgraph.descendants(root_entity)]

            for our_e, our_d in itertools.product(our_entities, our_data_types):
                if (test_term_match(self.DATA_TERM_MAP, our_d, pl_data_type) and
                    test_term_match(self.ENTITY_TERM_MAP, our_e, pl_entity)):
                    matched_edges.append((edge, our_e == root_entity, our_d == root_data_type))
                    break

        if matched_edges:
            selected_edges = set()
            matched_edges.sort(key=lambda i: (i[1], i[2]), reverse=True)
            _, *best_status = matched_edges[0]

            for edge, *status in matched_edges:
                if status == best_status and edge not in selected_edges:
                    selected_edges.add(edge)
                    yield *edge, candidate_edge_purposes[edge]

    def classify_contradiction(self, pos_edge, neg_edge):
        pos_e, pos_d, pos_action, pos_purposes = pos_edge
        neg_e, neg_d, neg_action, neg_purposes = neg_edge
        reasons = []

        if pos_action.startswith("NOT_") or not neg_action.startswith("NOT_"):
            return ["INVALID"]

        # Check if action conflict
        if pos_action != neg_action.split("_", 1)[1]:
            reasons.append("DIFF_ACTIONS")

        # Check if purposes conflict
        if neg_purposes and not(neg_purposes & pos_purposes):
            reasons.append("DIFF_PURPOSES")

        # Check if subjects conflict
        if (pos_d.split("@", 1) + [""])[1] != (neg_d.split("@", 1) + [""])[1]:
            reasons.append("DIFF_SUBJECTS")

        # Check if data types / entities conflict
        if len(reasons) == 0:
            dtype_conflict = check_subsum_conflict(self.kgraph, pos_d, neg_d)
            entity_conflict = check_subsum_conflict(self.kgraph, pos_e, neg_e)

            if not(dtype_conflict and entity_conflict):
                reasons.append("ONTOLOGY")

        return reasons


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

    current_app_id = None
    seen_policylint_contradictions = set()

    with (open(os.path.join(args.policheck_root, "policylint_results.csv"), encoding="utf-8", newline="") as fin,
          open(args.output, "w", encoding="utf-8", newline="") as fout):
        csv_writer = csv.DictWriter(fout, ["app_id",
                                           "pos_text", "neg_text",
                                           "pos_tuple", "neg_tuple",
                                           "pos_edge", "neg_edge",
                                           "labels"])
        csv_writer.writeheader()

        for row in csv.DictReader(fin):
            app_id = row["packageName"]

            if app_id not in workdir_map or row["contradictionNum"][0] != "C":
                continue

            if app_id != current_app_id:
                edge_mapper = EdgeMapper(workdir_map[app_id])
                current_app_id = app_id

            neg_tuple = (row["policyEntity"], row["policyData"])
            neg_sentences = row["policySentences"].split("||")
            pos_tuple = (row["contradictoryEntity"], row["contradictoryData"])
            pos_sentences = row["contradictionSentences"].split("||")

            if row["policyAction"] == "collect":
                neg_tuple, pos_tuple = pos_tuple, neg_tuple
                neg_sentences, pos_sentences = pos_sentences, neg_sentences

            # PolicyLint output has duplicates like T1<->T2 and T2<->T1
            if (app_id, neg_tuple, pos_tuple) in seen_policylint_contradictions:
                continue
            else:
                seen_policylint_contradictions.add((app_id, neg_tuple, pos_tuple))

            pos_edges = list(edge_mapper.match(pos_tuple, pos_sentences))
            neg_edges = list(edge_mapper.match(neg_tuple, neg_sentences))

            conflicting_edges = []
            ontology_conflicting_edges = []
            valid_edges = []
            invalid_edges = []

            for pos_edge, neg_edge in itertools.product(pos_edges, neg_edges):
                reasons = edge_mapper.classify_contradiction(pos_edge, neg_edge)

                if not reasons:
                    conflicting_edges.append((pos_edge, neg_edge, {"CONFLICT"}))
                elif "INVALID" in reasons:
                    invalid_edges.append((pos_edge, neg_edge, frozenset(reasons)))
                elif "ONTOLOGY" in reasons:
                    ontology_conflicting_edges.append((pos_edge, neg_edge, frozenset(reasons)))
                else:
                    valid_edges.append((pos_edge, neg_edge, frozenset(reasons)))

            selected_edge_pairs = conflicting_edges or ontology_conflicting_edges or valid_edges or invalid_edges

            if len(selected_edge_pairs) == 0:
                print(app_id)
                print(neg_sentences, neg_tuple, neg_edges)
                print(pos_sentences, pos_tuple, pos_edges)

            for pos_edge, neg_edge, reasons in selected_edge_pairs:
                pos_e, pos_d, pos_action, pos_purposes = pos_edge
                neg_e, neg_d, neg_action, neg_purposes = neg_edge

                csv_writer.writerow({
                    "app_id": app_id,
                    "pos_text": '\n'.join(pos_sentences),
                    "neg_text": '\n'.join(neg_sentences),
                    "pos_tuple": str(pos_tuple),
                    "neg_tuple": str(neg_tuple),
                    "pos_edge": f'{pos_e} {pos_action} {pos_d}\n{sorted(pos_purposes)}',
                    "neg_edge": f'{neg_e} {neg_action} {neg_d}\n{sorted(neg_purposes)}',
                    "labels": "\n".join(sorted(reasons)),
                })

    print("Number of unique PolicyLint tuples:", len(seen_policylint_contradictions))


if __name__ == "__main__":
    main()
