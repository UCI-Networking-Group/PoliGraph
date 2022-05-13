#!/usr/bin/env python3

import argparse
import csv
import itertools
import json
from pathlib import Path

import networkx as nx
import tldextract

DATA_ONTOLOGY = {
    "pii": [
        "advertising id", "android id", "serial number", "router ssid", "mac address", "imei", "sim serial number",
        "phone number", "email address", "person name"
    ],
    "non-pii": [],
    "contact information": [
        "phone number", "email address", "person name"
    ],
    "unique personal identifier": [
        "advertising id", "android id", "serial number", "router ssid", "mac address", "imei", "sim serial number",
        "phone number"
    ],
    "online identifier": ["advertising id"],
    "device identifier": [
        "advertising id", "android id", "serial number", "router ssid", "mac address", "imei", "sim serial number"
    ],
    "probabilistic identifier": ["advertising id"],
}

ENTITY_ONTOLOGY = {
    "advertising network": [
        "Google", "Facebook", "Unity", "AppsFlyer", "Verizon Media", "Chartboost", "Amazon.com",
        "Start.io", "Tapjoy", "AppLovin", "Liftoff", "ironSource", "AdColony"
    ],
    "analytic provider": [
        "Google", "Facebook", "Unity", "AppsFlyer", "Verizon Media", "ironSource"
    ],
    "social network": [
        "Facebook", "Twitter", "Google"
    ]
}


def iter_all_hypernyms(graph, first_node):
    seen_nodes = set()

    def dfs(node):
        yield node
        seen_nodes.add(node)

        for parent, _, data in graph.in_edges(node, data=True):
            if parent not in seen_nodes and data["label"] == "SUBSUM":
                yield from dfs(parent)

    yield from dfs(first_node)


def reverse_dict(d):
    result = dict()

    for key, li in d.items():
        for val in li:
            if val not in result:
                result[val] = [key]
            else:
                result[val].append(key)

    return result


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

    reversed_data_ontology = reverse_dict(DATA_ONTOLOGY)
    reversed_entity_ontology = reverse_dict(ENTITY_ONTOLOGY)

    for package_id, info in data.items():
        kgraph_path = policy_root / info["privacy_policy_id"] / 'graph.gml'

        if not kgraph_path.exists():
            continue

        kgraph = nx.read_gml(kgraph_path)
        unique_tuples = set()

        for flow in info["flows"]:
            data_type = flow["data_type"]
            clear_endpoints = []
            clear_datatypes = []

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
                    clear_endpoints.append("first party")

                result["flow_entity"] = "first party"
            else:
                result["flow_entity"] = recipient_entity

            if (flow_tuple := (recipient_entity, data_type)) not in unique_tuples:
                unique_tuples.add(flow_tuple)
            else:
                continue

            if data_type in kgraph:
                clear_datatypes.extend(iter_all_hypernyms(kgraph, data_type))

            if recipient_entity in kgraph:
                clear_endpoints.extend(iter_all_hypernyms(kgraph, recipient_entity))

            for e, d in itertools.product(clear_endpoints, clear_datatypes):
                edge_data = kgraph.get_edge_data(e, d)

                if edge_data and edge_data["label"] == "COLLECT":
                    result["policy_entity"] = e
                    result["policy_data"] = d
                    result["consistency_result"] = "clear"
                    break

            if result["consistency_result"] == "omitted":
                vague_endpoints = []
                vague_datatypes = []

                for d in reversed_data_ontology.get(data_type, []):
                    if d in kgraph:
                        vague_datatypes.extend(iter_all_hypernyms(kgraph, d))

                for e in reversed_entity_ontology.get(recipient_entity, []):
                    if e in kgraph:
                        vague_endpoints.extend(iter_all_hypernyms(kgraph, e))

                # both are vague
                for e, d in itertools.product(vague_endpoints, vague_datatypes):
                    edge_data = kgraph.get_edge_data(e, d)

                    if edge_data and edge_data["label"] == "COLLECT":
                        result["policy_entity"] = e
                        result["policy_data"] = d
                        result["consistency_result"] = "vague_both"
                        break

                # data type being vague
                for e, d in itertools.product(clear_endpoints, vague_datatypes):
                    edge_data = kgraph.get_edge_data(e, d)

                    if edge_data and edge_data["label"] == "COLLECT":
                        result["policy_entity"] = e
                        result["policy_data"] = d
                        result["consistency_result"] = "vague_d"
                        break

                # entity being vague
                for e, d in itertools.product(vague_endpoints, clear_datatypes):
                    edge_data = kgraph.get_edge_data(e, d)

                    if edge_data and edge_data["label"] == "COLLECT":
                        result["policy_entity"] = e
                        result["policy_data"] = d
                        result["consistency_result"] = "vague_e"
                        break

            writer.writerow(result)

    fout.close()


if __name__ == "__main__":
    main()
