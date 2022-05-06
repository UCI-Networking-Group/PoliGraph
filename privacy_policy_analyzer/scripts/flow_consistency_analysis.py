#!/usr/bin/env python3

import argparse
import csv
import itertools
import json
from pathlib import Path

import networkx as nx
import tldextract


def iter_all_hypernyms(graph, first_node):
    seen_nodes = set()

    def dfs(node):
        yield node
        seen_nodes.add(node)

        for parent, _, data in graph.in_edges(node, data=True):
            if parent not in seen_nodes and data["label"] == "SUBSUM":
                yield from dfs(parent)

    yield from dfs(first_node)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--flow-json", required=True, help="Flow JSON file")
    parser.add_argument("-p", "--privacy-policy-root", required=True, help="Input privacy policy directories")
    parser.add_argument("-d", "--entity-info", required=True, help="Path to entity_info.json")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    args = parser.parse_args()

    policy_root = Path(args.privacy_policy_root)

    with open(args.flow_json, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    with open(args.entity_info, "r", encoding="utf-8") as fin:
        domain_map = dict()

        for entity, info in json.load(fin).items():
            for d in info["domains"]:
                domain_map[d] = entity

    fout = open(args.output, "w", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=[
        "package_id", "flow_entity", "flow_data", "policy_entity", "policy_data",
        "consistency_result"])
    writer.writeheader()

    for package_id, info in data.items():
        kgraph_path = policy_root / info["privacy_policy_id"] / 'graph.gml'

        if not kgraph_path.exists():
            continue

        kgraph = nx.read_gml(kgraph_path)
        unique_tuples = set()

        for flow in info["flows"]:
            data_type = flow["data_type"]
            endpoints = []
            datatypes = []

            result = {
                "package_id": package_id,
                "flow_data": data_type,
                "policy_entity": "",
                "policy_data": "",
                "consistency_result": "omitted"
            }

            base_domain = tldextract.extract(flow["dest_domain"]).registered_domain
            recipient_entity = base_domain or "IP_ENDPOINT"

            if base_domain in domain_map:
                recipient_entity = domain_map[base_domain]

            if flow["party"] == "first party":
                if "first party" in kgraph:
                    endpoints.append("first party")

                result["flow_entity"] = "first party"
            else:
                result["flow_entity"] = recipient_entity

            if (flow_tuple := (recipient_entity, data_type)) not in unique_tuples:
                unique_tuples.add(flow_tuple)
            else:
                continue

            if data_type in kgraph:
                datatypes.extend(iter_all_hypernyms(kgraph, data_type))

            if recipient_entity in kgraph:
                endpoints.extend(iter_all_hypernyms(kgraph, recipient_entity))

            for e, d in itertools.product(endpoints, datatypes):
                edge_data = kgraph.get_edge_data(e, d)

                if edge_data and edge_data["label"] == "COLLECT":
                    result["policy_entity"] = e
                    result["policy_data"] = d
                    result["consistency_result"] = "clear"
                    break

            writer.writerow(result)

    fout.close()


if __name__ == "__main__":
    main()
