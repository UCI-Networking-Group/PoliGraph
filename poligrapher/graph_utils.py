import contextlib
from contextlib import contextmanager
import functools
import importlib.resources as pkg_resources
import itertools
import json

import networkx as nx
import yaml

import poligrapher
from poligrapher.annotators import CollectionAnnotator

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


def load_ontologies(data_ontology_path="", entity_info_json_path=""):
    with pkg_resources.path(poligrapher, "extra-data") as extra_data:
        if not data_ontology_path:
            data_ontology_path = extra_data / "data_ontology_readable.graphml"
        if not entity_info_json_path:
            entity_info_json_path = extra_data / "entity_info.json"

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


def contracted_nodes(G: nx.Graph, u, v, ensure_dag=False):
    """Contract node v into u in the graph G

    If G is a multi graph, we implement contraction ourselves because
    nx.contracted_nodes does not preserve key for multi graphs."""

    if ensure_dag:
        # DAG work around: If contracting causes circle, give up.
        while G.has_edge(u, v):
            G.remove_edge(u, v)

        if u in G and u in G and (nx.has_path(G, u, v) or nx.has_path(G, v, u)):
            G.remove_node(v)
            return

    if not G.is_multigraph():
        nx.contracted_nodes(G, u, v, self_loops=False, copy=False)
    else:
        edges_to_remap = itertools.chain(G.in_edges(v, keys=True, data=True), G.out_edges(v, keys=True, data=True))

        for (prev_w, prev_x, key, data) in edges_to_remap:
            w = prev_w if prev_w != v else u
            x = prev_x if prev_x != v else u

            if w != x and not G.has_edge(w, x, key):
                G.add_edge(w, x, key, **data)

        G.remove_node(v)


class KGraphRelation:
    def __init__(self, graph: nx.MultiDiGraph, relation: str, src_node: str, dst_node: str,
                 path: list[tuple[str, str, str]]):
        self._graph = graph
        self._relation = relation
        self._src = src_node
        self._dst = dst_node
        self._path = path

    def get_text(self):
        all_text = set()

        for u, v, key in self._path:
            edge_data = self._graph.get_edge_data(u, v, key)

            for text in edge_data['text']:
                if text not in all_text:
                    yield text
                    all_text.add(text)

    def __bool__(self):
        return True


class KGraph:
    def __init__(self, path, merge_geolocation=False):
        with open(path, "r", encoding="utf-8") as fin:
            kgraph: nx.MultiDiGraph = yaml_load_graph(fin)

        if merge_geolocation:
            # Merge geolocation / precise geolocation / coarse geolocation to align with PoliCheck
            for node in "precise geolocation", "coarse geolocation":
                if node in kgraph:
                    kgraph.add_node("geolocation", type="DATA")
                    contracted_nodes(kgraph, "geolocation", node)

        # For convenience, reverse all entity subsumption
        for u, v, key, data in list(kgraph.edges(keys=True, data=True)):
            if key == "SUBSUM" and kgraph.nodes[u]["type"] == "ACTOR":
                kgraph.remove_edge(u, v, key)
                kgraph.add_edge(v, u, "SUBSUM_BY", **data)

        # Separate positive and negative COLLECT edges to two graphs
        self.negative_kgraph = negative_kgraph = nx.MultiDiGraph()
        negative_kgraph.add_nodes_from(kgraph.nodes(data=True))
        negative_edges = []

        for u, v, key, data in kgraph.edges(keys=True, data=True):
            if key in ("SUBSUM", "SUBSUM_BY"):
                negative_kgraph.add_edge(u, v, key, **data)
            elif key in CollectionAnnotator.NEGATIVE_EDGE_TYPES:
                negative_kgraph.add_edge(u, v, key, **data)
                negative_edges.append((u, v, key))

        self.positive_kgraph = positive_kgraph = kgraph
        positive_kgraph.remove_edges_from(negative_edges)

    @property
    def datatypes(self):
        for node, data in self.positive_kgraph.nodes(data=True):
            if data["type"] == "DATA":
                yield node

    @property
    def entities(self):
        for node, data in self.positive_kgraph.nodes(data=True):
            if data["type"] == "ACTOR":
                yield node

    def can_collect(self, entity, datatype):
        G = self.positive_kgraph

        if G.nodes[entity]["type"] != "ACTOR" or G.nodes[entity]["datatype"] != "DATA":
            return False
        else:
            return nx.has_path(G, entity, datatype)

    def cannot_collect(self, entity, datatype):
        G = self.negative_kgraph

        if G.nodes[entity]["type"] != "ACTOR" or G.nodes[entity]["datatype"] != "DATA":
            return False
        else:
            return nx.has_path(G, entity, datatype)

    def who_collect(self, datatype):
        G = self.positive_kgraph

        if G.has_node(datatype) and G.nodes[datatype]["type"] == "DATA":
            for node in nx.ancestors(G, datatype):
                if G.nodes[node]["type"] == "ACTOR":
                    yield node

    def ancestors(self, anchor_node):
        G = self.positive_kgraph

        if anchor_node not in G:
            return

        match node_type := G.nodes[anchor_node]["type"]:
            case "DATA":
                li = nx.ancestors(G, anchor_node)
            case "ACTOR":
                li = nx.descendants(G, anchor_node)

        for node in li:
            if G.nodes[node]["type"] == node_type:
                yield node

    def descendants(self, anchor_node):
        G = self.positive_kgraph

        if anchor_node not in G:
            return

        match node_type := G.nodes[anchor_node]["type"]:
            case "DATA":
                li = nx.descendants(G, anchor_node)
            case "ACTOR":
                li = nx.ancestors(G, anchor_node)

        for node in li:
            if G.nodes[node]["type"] == node_type:
                yield node

    def subsum(self, node1, node2):
        G = self.positive_kgraph

        if not(G.has_node(node1) and G.has_node(node2)):
            return None

        node_type = G.nodes[node1]["type"]

        if node_type != G.nodes[node2]["type"]:
            return None

        match node_type:
            case "DATA":
                src, dst = node1, node2
                edge_type = "SUBSUM"
            case "ACTOR":
                src, dst = node2, node1
                edge_type = "SUBSUM_BY"

        try:
            shortest_path = nx.shortest_path(G, src, dst)
        except nx.exception.NetworkXNoPath:
            return None

        keyed_path = [(u, v, edge_type) for u, v in zip(shortest_path[:-1], shortest_path[1:])]
        return KGraphRelation(G, "SUBSUM", node1, node2, keyed_path)

    def purposes(self, entity, datatype):
        purposes = set()
        G = self.positive_kgraph

        # nx.all_simple_paths or nx.all_shortest_paths?
        # I feel shortest_paths would be less affected by vague language
        for path in _all_shortest_paths_wrap(G, entity, datatype):
            for u, v in zip(path[:-1], path[1:]):
                for key, data in G.get_edge_data(u, v).items():
                    if key in CollectionAnnotator.POSITIVE_EDGE_TYPES:
                        for p in data["purposes"]:
                            if p is not None and p not in purposes:
                                purposes.add(p)
                                yield p

    def get_text(self, node1, node2):
        """Get the policy texts that lead to the relation between two nodes:

        Case 1: node1 COLLECT node2, the path is node1 -> node2
        Case 2: node1 SUBSUME node2 (data types), the path is node1 -> node2
        Case 3: node1 COLLECT node2 (entities), the path is node2 -> node1
        """
        all_text = set()

        if self.positive_kgraph.nodes[node2]["type"] == "ACTOR":
            node1, node2 = node2, node1

        for path in _all_shortest_paths_wrap(self.positive_kgraph, node1, node2):
            for u, v in zip(path[:-1], path[1:]):
                for _, data in self.positive_kgraph.get_edge_data(u, v).items():
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
            if node in self.positive_kgraph:
                return node_type == self.positive_kgraph.nodes[node]["type"]
            else:
                self.positive_kgraph.add_node(node, type=node_type)
                return True

        try_add_node("UNSPECIFIED_DATA", "DATA")
        try_add_node("UNSPECIFIED_ACTOR", "ACTOR")

        for u, v in data_ontology.edges():
            try_add_node(u, "DATA")

            if data_ontology.nodes[v]["is_precise"] == 0 or v in self.positive_kgraph.nodes:
                if try_add_node(v, "DATA"):
                    self.positive_kgraph.add_edge(u, v, key="SUBSUM", text=["ONTOLOGY"])

        for u, v in entity_ontology.edges():
            try_add_node(u, "ACTOR")

            if entity_ontology.nodes[v]["is_precise"] == 0 or v in self.positive_kgraph.nodes:
                if try_add_node(v, "ACTOR"):
                    self.positive_kgraph.add_edge(v, u, key="SUBSUM_BY", text=["ONTOLOGY"])

    @contextmanager
    def attach_node(self, node, node_type):
        # Temperarily attach a node (usually a precise term)

        ontology = self.data_ontology if node_type == "DATA" else self.entity_ontology

        if node in self.positive_kgraph.nodes or node not in ontology.nodes:
            yield
            return

        try:
            self.positive_kgraph.add_node(node, type=node_type)
            for u, _ in ontology.in_edges(node):
                if self.positive_kgraph.nodes[u]["type"] == node_type:
                    if node_type == "DATA":
                        self.positive_kgraph.add_edge(u, node, key="SUBSUM", text=["ONTOLOGY"])
                    else:
                        self.positive_kgraph.add_edge(node, u, key="SUBSUM_BY", text=["ONTOLOGY"])

            yield
        finally:
            self.positive_kgraph.remove_node(node)

    @contextmanager
    def accept_unspecific_data(self):
        tmp_edges = []

        for data_type in self.datatypes:
            if data_type != "UNSPECIFIED_DATA":
                tmp_edges.append(("UNSPECIFIED_DATA", data_type, "SUBSUM"))

        self.positive_kgraph.add_edges_from(tmp_edges)
        yield
        self.positive_kgraph.remove_edges_from(tmp_edges)

    @contextmanager
    def accept_unspecific_actor(self):
        tmp_edges = []

        for entity in self.entities:
            if entity != "UNSPECIFIED_ACTOR":
                tmp_edges.append((entity, "UNSPECIFIED_ACTOR", "SUBSUM_BY"))

        self.positive_kgraph.add_edges_from(tmp_edges)
        yield
        self.positive_kgraph.remove_edges_from(tmp_edges)

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
