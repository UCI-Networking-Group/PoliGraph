import contextlib
import functools
import json
from contextlib import contextmanager

import networkx as nx
import yaml

# Hardcode the barebone of the entity ontology
ENTITY_ONTOLOGY = [
    "analytic provider",
    "content provider",
    "auth provider",
    "social media",
    "advertiser",
    "email service provider",
    "UNKNOWN",
]


def load_ontology_from_graphml(path):
    yed_graph = nx.read_graphml(path)
    ontology = nx.DiGraph()

    for _, data in yed_graph.nodes(data=True):
        name = data["label"]
        is_precise = data.get("is_precise", False)
        ontology.add_node(name, is_precise=is_precise)

    for node1, node2 in yed_graph.edges():
        name1 = yed_graph.nodes[node1]["label"]
        name2 = yed_graph.nodes[node2]["label"]
        ontology.add_edge(name1, name2)

    return ontology


def load_ontologies(data_ontology_path, entity_info_json_path):
    data_ontology = load_ontology_from_graphml(data_ontology_path)
    entity_ontology = nx.DiGraph()

    for entity_category in ENTITY_ONTOLOGY:
        entity_ontology.add_node(entity_category, is_precise=0)

    entity_ontology.add_node("we", is_precise=1)

    with open(entity_info_json_path, "r", encoding="utf-8") as fin:
        # domain_map = dict()

        for entity, info in json.load(fin).items():
            assert entity not in entity_ontology

            # for domain in info["domains"]:
            #     domain_map[domain] = entity

            entity_ontology.add_node(entity, is_precise=1)

            for entity_category in info["categories"] or ["UNKNOWN"]:
                assert entity_ontology.nodes[entity_category]["is_precise"] == 0
                entity_ontology.add_edge(entity_category, entity)

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


def yaml_dump_graph(G: nx.Graph, stream=None):
    graph_data = nx.node_link_data(G)

    for idx, link_dict in enumerate(graph_data['links']):
        new_link_dict = {
            'source': link_dict['source'],
            'target': link_dict['target'],
            'key': link_dict['key'],
            'text': sorted(set(link_dict['text'])),
        }

        if 'purposes' in link_dict:
            purposes = {}

            for label, text_list in sorted(link_dict['purposes'].items()):
                purposes[label] = sorted(set(text_list))

            new_link_dict['purposes'] = purposes

        graph_data['links'][idx] = new_link_dict

    graph_data['links'].sort(key=lambda d: (d['key'], d['source'] != 'we', d['source'], d['target']))
    graph_data['nodes'] = sorted(graph_data.pop('nodes'), key=lambda d: (d['type'], d['id']))

    return yaml.dump(graph_data, stream=stream, sort_keys=False, Dumper=yaml.CSafeDumper)


def yaml_load_graph(stream) -> nx.Graph:
    graph_data = yaml.load(stream, Loader=yaml.CSafeLoader)
    return nx.node_link_graph(graph_data)


def _all_shortest_paths_wrap(*args, **kwargs):
    try:
        yield from nx.all_shortest_paths(*args, **kwargs)
    except nx.NetworkXNoPath:
        return


class KGraph:
    def __init__(self, path, merge_geolocation=False):
        with open(path, "r", encoding="utf-8") as fin:
            kgraph = yaml_load_graph(fin)

        self.kgraph :nx.MultiDiGraph = kgraph

        # NOT_COLLECT is not used
        edges_to_remove = []
        for u, v, k in kgraph.edges(keys=True):
            if k == "NOT_COLLECT":
                edges_to_remove.append((u, v, k))
        kgraph.remove_edges_from(edges_to_remove)

        if merge_geolocation:
            # Merge geolocation / precise geolocation / coarse geolocation to align with PoliCheck
            if "geolocation" not in kgraph:
                kgraph.add_node("geolocation", type="DATA")

            for node in "precise geolocation", "coarse geolocation":
                if node in kgraph:
                    for parent, _, rel, data in kgraph.in_edges(node, keys=True, data=True):
                        if parent != "geolocation":
                            kgraph.add_edge(parent, "geolocation", key=rel, **data)

                    for _, child, rel, data in kgraph.out_edges(node, keys=True, data=True):
                        if child != "geolocation":
                            kgraph.add_edge("geolocation", child, key=rel, **data)

                    kgraph.remove_node(node)

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

    def who_collect(self, datatype):
        if datatype not in self.kgraph.nodes or self.kgraph.nodes[datatype]["type"] != "DATA":
            return

        for node in nx.ancestors(self.kgraph, datatype):
            if self.kgraph.nodes[node]["type"] == "ACTOR":
                yield node

    def ancestors(self, node):
        if node not in self.kgraph:
            return

        match node_type := self.kgraph.nodes[node]["type"]:
            case "DATA":
                li = nx.ancestors(self.kgraph, node)
            case "ACTOR":
                li = nx.descendants(self.kgraph, node)
            case _:
                assert False

        for node in li:
            if self.kgraph.nodes[node]["type"] == node_type:
                yield node

    def descendants(self, node):
        if node not in self.kgraph:
            return

        match node_type := self.kgraph.nodes[node]["type"]:
            case "DATA":
                li = nx.descendants(self.kgraph, node)
            case "ACTOR":
                li = nx.ancestors(self.kgraph, node)
            case _:
                assert False

        for node in li:
            if self.kgraph.nodes[node]["type"] == node_type:
                yield node

    def subsum(self, node1, node2):
        if node1 not in self.kgraph or node2 not in self.kgraph:
            return False

        node_type = self.kgraph.nodes[node1]["type"]

        if node_type != self.kgraph.nodes[node2]["type"]:
            return False

        match node_type:
            case "DATA":
                return nx.has_path(self.kgraph, node1, node2)
            case "ACTOR":
                return nx.has_path(self.kgraph, node2, node1)
            case _:
                assert False

    def purposes(self, entity, datatype):
        purposes = set()

        # nx.all_simple_paths or nx.all_shortest_paths?
        # I feel shortest_paths would be less affected by vague language
        for path in _all_shortest_paths_wrap(self.kgraph, entity, datatype):
            for u, v in zip(path[:-1], path[1:]):
                try:
                    edge_view = self.kgraph.edges[u, v, "COLLECT"]
                except KeyError:
                    continue

                for p, _ in edge_view["purposes"]:
                    if p is not None and p not in purposes:
                        purposes.add(p)
                        yield p

                break

    def get_text(self, node1, node2):
        """Get the policy texts that lead to the relation between two nodes:

        Case 1: node1 COLLECT node2, the path is node1 -> node2
        Case 2: node1 SUBSUME node2 (data types), the path is node1 -> node2
        Case 3: node1 COLLECT node2 (entities), the path is node2 -> node1
        """
        all_text = set()

        if self.kgraph.nodes[node2]["type"] == "ACTOR":
            node1, node2 = node2, node1

        for path in _all_shortest_paths_wrap(self.kgraph, node1, node2):
            for u, v in zip(path[:-1], path[1:]):
                for edge_type, data in self.kgraph.get_edge_data(u, v).items():
                    for text in data["text"]:
                        if text not in all_text:
                            all_text.add(text)

        return sorted(all_text)

    @functools.lru_cache
    def validate_collection(self, datatype):
        for _ in self.who_collect(datatype):
            return True

        return False

    @functools.lru_cache
    def validate_sharing(self, entity, datatype):
        for collection_entity in self.who_collect(datatype):
            if collection_entity == entity:
                return True

        return False


class ExtKGraph(KGraph):
    def __init__(self, path, data_ontology, entity_ontology, **kwargs):
        super().__init__(path, **kwargs)

        # Clone ontology edges to the KGraph
        # For performance reason, precise nodes are not added unless it's already in the KGraph
        self.data_ontology = data_ontology
        self.entity_ontology = entity_ontology

        def try_add_node(node, node_type):
            if node in self.kgraph:
                return node_type == self.kgraph.nodes[node]["type"]
            else:
                self.kgraph.add_node(node, type=node_type)
                return True

        try_add_node("UNSPECIFIC_DATA", "DATA")
        try_add_node("UNSPECIFIC_ACTOR", "ENTITY")

        for u, v in data_ontology.edges():
            try_add_node(u, "DATA")

            if data_ontology.nodes[v]["is_precise"] == 0 or v in self.kgraph.nodes:
                if try_add_node(v, "DATA"):
                    self.kgraph.add_edge(u, v, key="SUBSUM", text=["ONTOLOGY"])

        for u, v in entity_ontology.edges():
            try_add_node(u, "ACTOR")

            if entity_ontology.nodes[v]["is_precise"] == 0 or v in self.kgraph.nodes:
                if try_add_node(v, "ACTOR"):
                    self.kgraph.add_edge(v, u, key="SUBSUM_BY", text=["ONTOLOGY"])

    @contextmanager
    def attach_node(self, node, node_type):
        # Temperarily attach a node (usually a precise term)

        ontology = self.data_ontology if node_type == "DATA" else self.entity_ontology

        if node in self.kgraph.nodes or node not in ontology.nodes:
            yield
            return

        try:
            self.kgraph.add_node(node, type=node_type)
            for u, _ in ontology.in_edges(node):
                if self.kgraph.nodes[u]["type"] == node_type:
                    if node_type == "DATA":
                        self.kgraph.add_edge(u, node, key="SUBSUM", text=["ONTOLOGY"])
                    else:
                        self.kgraph.add_edge(node, u, key="SUBSUM_BY", text=["ONTOLOGY"])

            yield
        finally:
            self.kgraph.remove_node(node)

    @contextmanager
    def accept_unspecific_data(self):
        tmp_edges = []

        for data_type in self.datatypes:
            if data_type != "UNSPECIFIC_DATA":
                tmp_edges.append(("UNSPECIFIC_DATA", data_type, "SUBSUM"))

        self.kgraph.add_edges_from(tmp_edges)
        yield
        self.kgraph.remove_edges_from(tmp_edges)

    @contextmanager
    def accept_unspecific_actor(self):
        tmp_edges = []

        for entity in self.entities:
            if entity != "UNSPECIFIC_ACTOR":
                tmp_edges.append((entity, "UNSPECIFIC_ACTOR", "SUBSUM_BY"))

        self.kgraph.add_edges_from(tmp_edges)
        yield
        self.kgraph.remove_edges_from(tmp_edges)

    def who_collect(self, datatype):
        # Limitation: Precise company names are not returned unless already in the KGraph

        with self.attach_node(datatype, "DATA"):
            yield from super().who_collect(datatype)

    def purposes(self, entity, datatype):
        with self.attach_node(datatype, "DATA"):
            with self.attach_node(entity, "ACTOR"):
                yield from super().purposes(entity, datatype)

    @functools.lru_cache
    def validate_collection(self, datatype):
        return super().validate_collection(datatype)

    @functools.lru_cache
    def validate_sharing(self, entity, datatype, accept_unspecific_data=True):
        context = self.accept_unspecific_data if accept_unspecific_data else contextlib.nullcontext

        with context(), self.attach_node(entity, "ACTOR"):
            return super().validate_sharing(entity, datatype)

    @functools.lru_cache
    def validate_purpose(self, entity, datatype, purpose, accept_unspecific=True):
        if accept_unspecific:
            contexts = [self.accept_unspecific_data, self.accept_unspecific_actor]
        else:
            contexts = [contextlib.nullcontext, contextlib.nullcontext]

        with contexts[0](), contexts[1]():
            for disclosed_purpose in self.purposes(entity, datatype):
                if disclosed_purpose == purpose:
                    return True

        return False
