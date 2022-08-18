#!/usr/bin/env python3

import argparse
import csv
import os

import networkx as nx
from privacy_policy_analyzer.graph_utils import load_ontologies


def check_definitions(kgraph, data_ontology, entity_ontology):
    def dfs(node, parent_ontology_nodes=()):
        nonlocal target_ontology, result

        if node in target_ontology:
            for parent in parent_ontology_nodes:
                if not nx.has_path(target_ontology, parent, node):
                    result.add((parent, node))

            if target_ontology.nodes[node]["is_precise"] == 0:
                parent_ontology_nodes = parent_ontology_nodes + (node,)

        for _, child, relationship in kgraph.out_edges(node, keys=True):
            if relationship == "SUBSUM":
                dfs(child, parent_ontology_nodes)

    bad_datatype_links = set()
    bad_entity_links = set()

    for node in kgraph.nodes():
        match kgraph.nodes[node]["type"]:
            case "DATA":
                target_ontology = data_ontology
                result = bad_datatype_links
            case "ACTOR":
                target_ontology = entity_ontology
                result = bad_entity_links
            case _:
                raise ValueError()

        dfs(node)

    return bad_datatype_links, bad_entity_links


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("-y", "--ontology", required=True, help="Ontology directory")
    parser.add_argument("-e", "--entity-info", required=True, help="Path to entity_info.json")
    args = parser.parse_args()

    data_ontology, entity_ontology = load_ontologies(args.ontology, args.entity_info)

    for d in args.workdirs:
        kgraph_path = os.path.join(d, 'graph_trimmed.gml')

        if not os.path.isfile(kgraph_path):
            continue

        print(f"Processing {d} ...")

        kgraph = nx.read_gml(kgraph_path)

        bad_datatype_links, bad_entity_links = check_definitions(kgraph, data_ontology, entity_ontology)

        with open(os.path.join(d, "misleading_definitions.csv"), "w") as fout:
            writer = csv.DictWriter(fout, fieldnames=["type", "parent", "child"])
            writer.writeheader()

            for u, v in bad_datatype_links:
                writer.writerow(dict(type="data", parent=u, child=v))

            for u, v in bad_entity_links:
                writer.writerow(dict(type="entity", parent=u, child=v))

if __name__ == "__main__":
    main()
