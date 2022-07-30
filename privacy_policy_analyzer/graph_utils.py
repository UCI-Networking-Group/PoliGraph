import json
from pathlib import Path

import networkx as nx


def load_ontologies(ontology_root, entity_info_json_path):
    ontology_root = Path(ontology_root)

    data_ontology = nx.read_gml(ontology_root / "data.gml")
    entity_ontology = nx.read_gml(ontology_root / "entity.gml")

    with open(entity_info_json_path, "r", encoding="utf-8") as fin:
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

    return data_ontology, entity_ontology


def gml_destringizer(s):
    if s.startswith('[') or s.startswith('{'):
        return json.loads(s)
    else:
        return s


def gml_stringizer(obj):
    if isinstance(obj, str):
        return obj
    else:
        return json.dumps(obj)


class KGraph:
    def __init__(self, path):
        self.kgraph = kgraph = nx.read_gml(path, destringizer=gml_destringizer)

        # NOT_COLLECT is not used
        edges_to_remove = []
        for u, v, k in kgraph.edges(keys=True):
            if k == "NOT_COLLECT":
                edges_to_remove.append((u, v, k))
        kgraph.remove_edges_from(edges_to_remove)

        # For convenience, reverse all entity subsumption
        for u, v, rel, edge_data in list(kgraph.edges(keys=True, data=True)):
            if rel == "SUBSUM" and kgraph.nodes[u]["type"] == "ACTOR":
                kgraph.remove_edge(u, v, rel)
                kgraph.add_edge(v, u, "SUBSUM_BY", **edge_data)

    @property
    def datatypes(self):
        for node, data in self.kgraph.nodes(data=True):
            if data["type"] == "DATA":
                yield node

    @property
    def entities(self):
        for node, data in self.kgraph.nodes(data=True):
            if data["type"] == "ACTOR":
                yield node

    def who_collect(self, datatype, strict_datatype=False):
        if datatype not in self.kgraph.nodes:
            return

        for node in nx.ancestors(self.kgraph, datatype):
            if self.kgraph.nodes[node]["type"] == "ACTOR":
                if strict_datatype:
                    try:
                        self.kgraph.edges[node, datatype, "COLLECT"]
                    except KeyError:
                        continue

                yield node

    def ancestors(self, node):
        if node not in self.kgraph:
            return

        node_type = self.kgraph.nodes[node]["type"]
        if node_type == "ACTOR":
            li = nx.descendants(self.kgraph, node)
        else:
            li = nx.ancestors(self.kgraph, node)

        for node in li:
            if self.kgraph.nodes[node]["type"] == node_type:
                yield node

    def subsum(self, node1, node2):
        if node1 not in self.kgraph or node2 not in self.kgraph:
            return False

        node_type = self.kgraph.nodes[node1]["type"]

        if node_type != self.kgraph.nodes[node2]["type"]:
            return False

        if node_type == "ACTOR":
            return nx.has_path(self.kgraph, node2, node1)
        else:
            return nx.has_path(self.kgraph, node1, node2)

    def purposes(self, entity, datatype, strict_data=False):
        purposes = set()

        for path in nx.all_simple_paths(self.kgraph, entity, datatype):
            for u, v in zip(path[:-1], path[1:]):
                try:
                    edge_view = self.kgraph.edges[u, v, "COLLECT"]
                except KeyError:
                    continue

                if strict_data and v != datatype:
                    break

                for p, _ in edge_view["purposes"]:
                    if p is not None:
                        purposes.add(p)

                break

        return sorted(purposes)
