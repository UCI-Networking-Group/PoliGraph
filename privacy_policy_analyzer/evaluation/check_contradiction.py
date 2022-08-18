#!/usr/bin/env python3

import argparse
import json
import os
import re
from privacy_policy_analyzer.graph_utils import load_ontologies

import networkx as nx


def extract_subsum_subgraph(kgraph: nx.MultiDiGraph):
    subgraph = nx.DiGraph()

    for u in kgraph.nodes():
        subgraph.add_node(u)

        for _, v, rel in kgraph.out_edges(u, keys=True):
            if rel == "SUBSUM":
                subgraph.add_edge(u, v)

    return subgraph


def expand_weak_subsum(kgraph: nx.MultiDiGraph, data_ontology, entity_ontology):
    new_graph = kgraph.copy()

    for n in data_ontology.nodes():
        if n in new_graph and new_graph.nodes[n]["type"] != "DATA":
            return None

        new_graph.add_node(n, type="DATA")

    for n in entity_ontology.nodes():
        if n in new_graph and new_graph.nodes[n]["type"] != "ACTOR":
            return None

        new_graph.add_node(n, type="ACTOR")

    for u, v in data_ontology.edges():
        if not new_graph.has_edge(u, v, "SUBSUM"):
            new_graph.add_edge(u, v, "SUBSUM")

    for u, v in entity_ontology.edges():
        if entity_ontology.nodes[v].get("is_precise") == 1:
            # skip a lot of precise entity names
            continue

        if not new_graph.has_edge(u, v, "SUBSUM"):
            new_graph.add_edge(u, v, "SUBSUM")

    return new_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("-y", "--ontology", required=True, help="Ontology directory")
    parser.add_argument("-e", "--entity-info", required=True, help="Path to entity_info.json")
    args = parser.parse_args()

    data_ontology, entity_ontology = load_ontologies(args.ontology, args.entity_info)

    contradiction_types = {
        "=>": "N1",
        "<>": "N2",
        ">=": "N3",
        ">>": "N4",
        "==": "C1",
        "=<": "C2",
        "<=": "C3",
        "<<": "C4",
        "><": "C5",
    }

    for d in args.workdirs:
        kgraph_path = os.path.join(d, 'graph_trimmed.gml')

        if not os.path.isfile(kgraph_path):
            continue

        print(f"Processing {d} ...")

        kgraph = nx.read_gml(kgraph_path)

        collect_tuples = []
        not_collect_tuples = []

        for ent, dt, rel in kgraph.edges(keys=True):
            if rel == "COLLECT":
                collect_tuples.append((ent, dt))
            elif rel == "NOT_COLLECT":
                not_collect_tuples.append((ent, dt))

        if len(not_collect_tuples) == 0 or len(collect_tuples) == 0:
            # avoid expensive operations if no tuple is found
            continue

        kgraph_subsum = extract_subsum_subgraph(kgraph)

        ext_kgraph = expand_weak_subsum(kgraph, data_ontology, entity_ontology)

        if ext_kgraph is None:
            print("Bad extended graph!!!")
            ext_kgraph_subsum = None
        else:
            ext_kgraph_subsum = extract_subsum_subgraph(ext_kgraph)

        contradictions = []

        for e1, d1 in collect_tuples:
            for e2, d2 in not_collect_tuples:
                status = []

                for p1, p2 in (e1, e2), (d1, d2):
                    if p1 == p2:
                        status.append(("internal", "="))
                    elif nx.has_path(kgraph_subsum, p1, p2):
                        status.append(("internal", ">"))
                    elif nx.has_path(kgraph_subsum, p2, p1):
                        status.append(("internal", "<"))
                    elif ext_kgraph_subsum and nx.has_path(ext_kgraph_subsum, p1, p2):
                        status.append(("ontology", ">"))
                    elif ext_kgraph_subsum and nx.has_path(ext_kgraph_subsum, p2, p1):
                        status.append(("ontology", "<"))
                    else:
                        break

                if len(status) != 2:
                    # no contradiction
                    continue

                e_state, d_state = status
                positive_text = kgraph[e1][d1]["COLLECT"]["text"]
                negative_text = kgraph[e2][d2]["NOT_COLLECT"]["text"]

                if d_state[0] == e_state[0] == "internal":
                    cause = {"internal"}
                else:
                    cause = {"ontology"}

                policylint_contradiction_type = contradiction_types[e_state[1] + d_state[1]]

                for sentence in negative_text:
                    if re.search(r"\b((under|in|below|between)\s*((the|that|this)? ages?|\d+|sixteen|eighteen))\b", sentence, re.I):
                        cause.add("age_group")
                    elif re.search(r"\b((from|of) (children|a child|minors?))\b", sentence, re.I):
                        cause.add("age_group")

                    if re.search(r"\b(sell|sold|rent|rented|lease|leased|trade|traded)\b", sentence, re.I):
                        cause.add("for_revenue")

                    if re.search(r"\b(unless|without.*(consent|approval))\b", sentence, re.I):
                        cause.add("exception")

                contradiction_info = {
                    "pos_tuple": (e1, d1),
                    "neg_tuple": (e2, d2),
                    "pos_text": positive_text,
                    "neg_text": negative_text,
                    "causes": sorted(cause),
                    "policylint": policylint_contradiction_type,
                }

                print(json.dumps(contradiction_info, indent=2))
                contradictions.append(contradiction_info)

        if len(contradictions) > 0:
            with open(os.path.join(d, "contradictions.json"), "w") as fout:
                json.dump(contradictions, fout)


if __name__ == "__main__":
    main()
