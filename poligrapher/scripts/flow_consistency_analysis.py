#!/usr/bin/env python3

import argparse
import itertools
import json
from collections import deque
from pathlib import Path

import networkx as nx
import tldextract


def iter_hypernyms(graph, start_node):
    bfs_queue = deque()
    bfs_queue.append((start_node,))
    visited_nodes = {start_node}

    while len(bfs_queue) > 0:
        path = bfs_queue.popleft()
        yield path

        for parent, _, rel in graph.in_edges(path[-1], keys=True):
            if parent not in visited_nodes and rel == "SUBSUM":
                visited_nodes.add(parent)
                bfs_queue.append((parent,) + path)


def cache_parents(graph):
    precise_nodes = set()
    parent_node_dict = dict()

    for n, data in graph.nodes(data=True):
        parent_node_dict[n] = set()

        if data["is_precise"]:
            precise_nodes.add(n)

    for n1, n2 in nx.topological_sort(nx.line_graph(graph)):
        parent_node_dict[n2].add(n1)
        parent_node_dict[n2].update(parent_node_dict[n1])

    return {k: parent_node_dict[k] for k in precise_nodes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("flow_json", help="Flow JSON file")
    parser.add_argument("output", help="Output JSON file")
    parser.add_argument("-y", "--ontology", required=True, help="Ontology directory")
    parser.add_argument("-p", "--privacy-policy-root", required=True, help="Input privacy policy directories")
    parser.add_argument("-e", "--entity-info", required=True, help="Path to entity_info.json")
    args = parser.parse_args()

    policy_root = Path(args.privacy_policy_root)
    ontology_root = Path(args.ontology)

    data_ontology = nx.read_gml(ontology_root / "data.gml")
    entity_ontology = nx.read_gml(ontology_root / "entity.gml")

    with open(args.flow_json, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    with open(args.entity_info, "r", encoding="utf-8") as fin:
        domain_map = dict()

        for entity, info in json.load(fin).items():
            if entity in entity_ontology:
                raise ValueError(f"Duplicated entity: {entity}")

            for domain in info["domains"]:
                domain_map[domain] = entity

            entity_ontology.add_node(entity, is_precise=1)

            for cat in info["categories"] or ["third party"]:
                if entity_ontology.nodes[cat]["is_precise"]:
                    raise ValueError(f"Invalid entity category: {cat}")

                entity_ontology.add_edge(cat, entity)

    reversed_data_ontology = cache_parents(data_ontology)
    reversed_entity_ontology = cache_parents(entity_ontology)
    all_results = {}

    for package_id, info in data.items():
        kgraph_path = policy_root / info["privacy_policy_id"] / 'graph.gml'

        if not kgraph_path.exists():
            continue

        kgraph = nx.read_gml(kgraph_path)
        unique_tuples = set()

        all_results[package_id] = flow_results = []

        # Assign party labels to entities (instead of domains) based on both package name and PoliCheck's labels
        first_party_entities = set()

        ## From package name
        rev_package_name = ".".join(reversed(package_id.split(".")))
        base_domain = tldextract.extract(rev_package_name).registered_domain

        if base_domain in domain_map:
            first_party_entities.add(domain_map[base_domain])

        ## From PoliCheck's label
        for flow in info["flows"]:
            base_domain = tldextract.extract(flow["dest_domain"]).registered_domain

            if base_domain in domain_map:
                flow["entity"] = domain_map[base_domain]
            else:
                flow["entity"] = base_domain or "IP_ENDPOINT"  # base domain can be empty if dest is IP

            if flow["party"] == "first party" and flow["entity"] != "IP_ENDPOINT":
                first_party_entities.add(flow["entity"])

        for flow in info["flows"]:
            data_type = flow["data_type"]
            recipient_entity = flow["entity"]

            result = {"flow_data": data_type, "flow_entity": recipient_entity}

            if base_domain in domain_map:
                recipient_entity = domain_map[base_domain]

            hypernym_paths = dict()
            clear_datatypes = list()
            clear_endpoints = list()
            vague_datatypes = list()
            vague_endpoints = list()

            if recipient_entity in first_party_entities:
                if "we" in kgraph:
                    clear_endpoints.append("we")
                    hypernym_paths["we"] = ("we",)

                result["party"] = "first party"
            else:
                result["party"] = "third party"

            if (flow_tuple := (recipient_entity, data_type)) not in unique_tuples:
                unique_tuples.add(flow_tuple)
            else:
                continue

            if data_type in kgraph:
                for path in iter_hypernyms(kgraph, data_type):
                    clear_datatypes.append(path[0])
                    hypernym_paths[path[0]] = path

            if recipient_entity in kgraph:
                for path in iter_hypernyms(kgraph, recipient_entity):
                    clear_endpoints.append(path[0])
                    hypernym_paths[path[0]] = path

            for d in reversed_data_ontology.get(data_type, []):
                if d in kgraph:
                    for path in iter_hypernyms(kgraph, d):
                        if path[0] not in hypernym_paths:
                            vague_datatypes.append(path[0])
                            hypernym_paths[path[0]] = path

            for e in reversed_entity_ontology.get(recipient_entity, []):
                if e in kgraph:
                    for path in iter_hypernyms(kgraph, e):
                        if path[0] not in hypernym_paths:
                            vague_endpoints.append(path[0])
                            hypernym_paths[path[0]] = path

            all_to_check = [
                ("clear", clear_datatypes, clear_endpoints),
                ("vague_e", clear_datatypes, vague_endpoints),
                ("vague_d", vague_datatypes, clear_endpoints),
                ("vague_both", vague_datatypes, vague_endpoints),
            ]

            result["policies"] = policies = []

            for disclosure_type, datatypes, endpoints in all_to_check:
                for d, e in itertools.product(datatypes, endpoints):
                    for rel, consistency in (("COLLECT", "consistent"), ("NOT_COLLECT", "inconsistent")):
                        if edge_data := kgraph.get_edge_data(e, d, key=rel):
                            all_sentences = set(edge_data["text"])

                            for item in d, e:
                                hpath = hypernym_paths[item]
                                for parent, child in zip(hpath[:-1], hpath[1:]):
                                    all_sentences.update(kgraph[parent][child]["SUBSUM"]["text"])

                            policies.append({
                                "consistency": consistency,
                                "disclosure_type": disclosure_type,
                                "entity": e,
                                "data": d,
                                "text": list(all_sentences),
                            })

                # Ignore vague ones if clear statements are found
                if disclosure_type == "clear" and len(policies) > 0:
                    break

            flow_results.append(result)

    with open(args.output, "w", encoding="utf-8") as fout:
        json.dump(all_results, fout)


if __name__ == "__main__":
    main()
