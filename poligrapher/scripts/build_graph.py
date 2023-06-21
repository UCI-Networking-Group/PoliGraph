#!/usr/bin/env python3

import argparse
from collections import defaultdict, deque
import importlib.resources as pkg_resources
import itertools
import logging
import os

import networkx as nx
import yaml

import poligrapher
from poligrapher.annotators import CollectionAnnotator
from poligrapher.document import PolicyDocument, SegmentType
from poligrapher.graph_utils import contracted_nodes, yaml_dump_graph
from poligrapher.phrase_normalization import EntityMatcher, RuleBasedPhraseNormalizer
from poligrapher.purpose_classification import PurposeClassifier
from poligrapher.utils import setup_nlp_pipeline


def dag_add_edge(G, n1, n2, *args, **kwargs):
    if (n1 == n2) or (G.has_node(n1) and n2 in nx.ancestors(G, n1)):
        return False
    else:
        G.add_edge(n1, n2, *args, **kwargs)
        return True


class GraphBuilder:
    def __init__(self, phrase_map, entity_map, purpose_classification_model_path, variant):
        with open(phrase_map, "r", encoding="utf-8") as fin:
            phrase_map_rules = yaml.safe_load(fin)

        self.entity_mapper = EntityMatcher(entity_map)
        self.data_phrase_normalizer = RuleBasedPhraseNormalizer(phrase_map_rules["DATA"])
        self.actor_phrase_normalizer = RuleBasedPhraseNormalizer(phrase_map_rules["ACTOR"])
        self.purpose_classifier = PurposeClassifier(purpose_classification_model_path)
        self.variant = variant

    def build_graph(self, document: PolicyDocument):
        token_type_map = {}

        G_collect = nx.MultiDiGraph()
        G_subsum = nx.DiGraph()
        G_coref = nx.DiGraph()

        normalized_terms_map = {}
        G_final = nx.MultiDiGraph()

        def build_phrase_map_from_ent_labels():
            """Step 1: Populate token_type_map using NER labels."""

            for src in document.token_relationship.nodes:
                token = document.get_token_with_src(src)

                if (ent_type := token.ent_type_) in ("DATA", "ACTOR"):
                    token_type_map[src] = ent_type

        def build_collect_graph():
            """Step 2: Infer phrase types using COLLECT-like edges and copy
            non-conflict edges to G_collect. Classify purpose phrases and save
            them as edge attributes in G_collect.
            """

            data_type_purposes: dict[tuple, list[str]] = {}

            for src1, src2, relationship in document.token_relationship.edges(keys=True):
                if relationship in CollectionAnnotator.EDGE_TYPES:
                    for src, expected_type in (src1, "ACTOR"), (src2, "DATA"):
                        if token_type_map.setdefault(src, expected_type) != expected_type:
                            break
                    else:
                        match self.variant:
                            case "original" | "per_sentence" | "per_section":
                                # Only "COLLECT" edges
                                if not relationship.startswith("NOT_"):
                                    G_collect.add_edge(src1, src2, key="COLLECT")
                                    data_type_purposes.setdefault(src2, [])
                            case "extended":
                                # Extension: keep all edges
                                G_collect.add_edge(src1, src2, key=relationship)
                                data_type_purposes.setdefault(src2, [])

            # Annotate purposes
            purpose_text_to_labels: dict[str, list[str]] = {}

            for data_type_src, purposes in data_type_purposes.items():
                for _, purpose_src, relationship in document.token_relationship.edges(data_type_src, keys=True):
                    if relationship == "PURPOSE":
                        purpose_root = document.get_token_with_src(purpose_src)

                        left = purpose_root.left_edge.i
                        right = purpose_root.right_edge.i + 1
                        purpose_text = purpose_root.doc[left:right].text.strip()

                        purposes.append(purpose_text)
                        purpose_text_to_labels[purpose_text] = []

            for (text, labels), predictions in zip(purpose_text_to_labels.items(),
                                                   self.purpose_classifier(list(purpose_text_to_labels))):
                logging.info("Purpose %r -> %s", text, predictions)
                labels.extend(predictions)

            for data_type_src, purpose_text_list in data_type_purposes.items():
                edge_purposes = set()

                for purpose_text in purpose_text_list:
                    for purpose in purpose_text_to_labels[purpose_text]:
                        edge_purposes.add((purpose, purpose_text))

                for _, _, _, edge_data in G_collect.in_edges(data_type_src, keys=True, data=True):
                    edge_data["purposes"] = edge_purposes

        def build_subsum_and_coref_graphs():
            """Step 3: Infer phrase types using SUBSUM / COREF edges and copy
            non-conflict edges to G_subsum / G_coref.
            """

            # Run a BFS starting from nodes with known types.
            bfs_queue = deque(token_type_map.keys())
            visited_nodes = set(token_type_map.keys())

            while len(bfs_queue) > 0:
                src1 = bfs_queue.popleft()
                token_type = token_type_map[src1]

                in_edge_view = document.token_relationship.in_edges(src1, keys=True)
                out_edge_view = document.token_relationship.out_edges(src1, keys=True)

                for edge_from, edge_to, relationship in itertools.chain(in_edge_view, out_edge_view):
                    if relationship in ["SUBSUM", "COREF"]:
                        src2 = edge_to if src1 == edge_from else edge_from

                        if token_type_map.setdefault(src2, token_type) == token_type:
                            if src2 not in visited_nodes:
                                visited_nodes.add(src2)
                                bfs_queue.append(src2)

                            # Call dag_add_edge to safely add an edge without creating a circle
                            match relationship:
                                case "SUBSUM": dag_add_edge(G_subsum, edge_from, edge_to)
                                case "COREF": dag_add_edge(G_coref, edge_from, edge_to)

        def contract_coref_nodes():
            """Step 4: Follow G_coref to contract coreferences in G_subsum and
            G_collect into their main mentions.
            """

            if self.variant == "per_sentence":
                return

            for src1 in nx.topological_sort(G_coref):
                match G_coref.out_degree(src1):
                    case 0:
                        continue
                    case 1:
                        (_, src2), = G_coref.out_edges(src1)

                        if src1 in G_collect:
                            contracted_nodes(G_collect, src2, src1)

                        if src1 in G_subsum:
                            contracted_nodes(G_subsum, src2, src1, ensure_dag=True)
                    case _:
                        # A phrase graph allows one token to have multiple COREF edges
                        # Turn them into SUBSUM edges
                        for _, src2 in G_coref.out_edges(src1):
                            dag_add_edge(G_subsum, src1, src2)

        def reduce_graph():
            """Step 5: Reduce G_subsum / G_collect to have fewer edges without
            changing connectivity, so the output graph can be simpler."""
            nonlocal G_subsum, G_collect

            # Reduce G_subsum
            G_subsum = nx.transitive_reduction(G_subsum)

            # Reduce G_collect
            subsum_topo_order = {node: i for i, node in enumerate(nx.topological_sort(G_subsum))}
            edges_to_remove = set()

            for u in G_collect.nodes:
                match token_type_map[u]:
                    case "DATA":
                        edges = sorted(G_collect.in_edges(u, keys=True), key=lambda k: subsum_topo_order.get(k, 0))
                        other_idx = 0
                    case "ACTOR":
                        edges = sorted(G_collect.out_edges(u, keys=True), key=lambda k: subsum_topo_order.get(k, 0))
                        other_idx = 1

                for edge_tuple1, edge_tuple2 in itertools.combinations(edges, 2):
                    v1 = edge_tuple1[other_idx]
                    v2 = edge_tuple2[other_idx]

                    try:
                        if edge_tuple1[-1] != edge_tuple2[-1] or not nx.has_path(G_subsum, v1, v2):
                            continue
                    except nx.exception.NodeNotFound:
                        continue

                    purposes1 = G_collect.get_edge_data(*edge_tuple1)['purposes']
                    purposes2 = G_collect.get_edge_data(*edge_tuple2)['purposes']

                    if not purposes2.difference(purposes1):
                        edges_to_remove.add(edge_tuple2)

            G_collect.remove_edges_from(edges_to_remove)

        def _expand_phrase(src):
            root_token = document.get_token_with_src(src)

            if (base_phrase := root_token._.ent) is None:
                return None

            right_boundary = base_phrase.end

            for child in sorted(filter(lambda t: t.i >= right_boundary, root_token.rights)):
                child_indices = sorted(t.i for t in child.subtree)

                if (
                    child.dep_ not in {'punct', "dep", "meta", "cc", "preconj", "conj"}
                    and all(t._.src not in token_type_map for t in child.subtree)      # No overlap w/ other nodes
                    and right_boundary == child_indices[0]                             # Continuous
                    and child_indices[-1] - child_indices[0] + 1 == len(child_indices) # Continuous subtree
                ):
                    right_boundary = child_indices[-1] + 1
                else:
                    break

            return root_token.doc[base_phrase.start:right_boundary]

        def _eliminate_intermediate_node(src):
            if G_collect.has_node(src):
                if token_type_map[src] == "DATA":
                    for u, _, rel, data in G_collect.in_edges(src, keys=True, data=True):
                        for _, v in G_subsum.out_edges(src):
                            G_collect.add_edge(u, v, rel, **data)
                elif token_type_map[src] == "ACTOR":
                    for _, v, rel, data in G_collect.out_edges(src, keys=True, data=True):
                        for _, u in G_subsum.out_edges(src):
                            G_collect.add_edge(u, v, rel, **data)

                G_collect.remove_node(src)

            for u, _ in G_subsum.in_edges(src):
                for _, v in G_subsum.out_edges(src):
                    dag_add_edge(G_subsum, u, v)

            G_subsum.remove_node(src)

        def _get_section_id(src):
            """Get the section ID for per_section ablative configuration"""
            segment = document.segments[src[0]]

            while segment is not None and segment.segment_type != SegmentType.HEADING:
                segment = segment.parent

            return segment.segment_id if segment is not None else 0

        def normalize_terms():
            """Step 6: Run phrase normalization."""

            for src, token_type in token_type_map.items():
                if (phrase := _expand_phrase(src)) is None:
                    normalized_terms_map[src] = []
                    continue

                # Fallback to lemmatization only if token types from NER and graph relation agrees
                flag_use_stem = phrase.root.ent_type_ == token_type

                # All candidate terms
                terms = set()

                match token_type:
                    case "DATA":
                        terms.update(self.data_phrase_normalizer.normalize(phrase, flag_use_stem))
                    case "ACTOR":
                        # If there is any proper noun, run entity_mapper to find company names
                        if any(t.pos_ == "PROPN" for t in phrase):
                            terms.update(self.entity_mapper.match_name(phrase.text))
                            flag_use_stem = flag_use_stem and not terms

                        # Try rule-based normalizer
                        terms.update(self.actor_phrase_normalizer.normalize(phrase, flag_use_stem))

                has_subsum = G_subsum.has_node(src) and G_subsum.out_degree(src) > 0
                be_subsumed = G_subsum.has_node(src) and G_subsum.in_degree(src) > 0

                if "UNSPECIFIED" in terms and len(terms) == 1 and not has_subsum and not be_subsumed:
                    # UNSPECIFIED node cannot subsume or be subsumed by any other node
                    terms.remove("UNSPECIFIED")
                    terms.add(f"UNSPECIFIED_{token_type}")
                else:
                    terms.discard("UNSPECIFIED")

                    if has_subsum and not terms:
                        # If the phrase subsumes anything, at least fallback to an intermediate node
                        _eliminate_intermediate_node(src)

                match self.variant:
                    case "extended":
                        # Extension: include data subject (if no subject info then this is no-op)
                        if token_type == "DATA":
                            if subject := document.token_relationship.nodes[src].get('subject'):
                                replaced_terms = [f"{term} @{subject}" for term in terms]
                                terms.clear()
                                terms.update(replaced_terms)
                    case "per_sentence":
                        # PolicyLint simulation -- Make every term unique
                        replaced_terms = [f"{term} {src}" for term in terms]
                        terms.clear()
                        terms.update(replaced_terms)
                    case "per_section":
                        # "per_section" simulation -- Limit relations within a section
                        section_id = _get_section_id(src)
                        replaced_terms = [f"{term} {(section_id, 0)}" for term in terms]
                        terms.clear()
                        terms.update(replaced_terms)

                G_final.add_nodes_from(terms, type=token_type)

                normalized_terms_map[src] = sorted(terms, reverse=True)  # Stablize set order for reproducibility
                logging.info("Phrase %r (%s) -> %r", phrase.text, token_type, ", ".join(normalized_terms_map[src]))

        def merge_subsum_graph():
            """Step 7: Populate SUBSUM edges in G_final from G_subsum."""

            relationship = "SUBSUM"

            for src1, src2 in G_subsum.edges():
                src1_terms = normalized_terms_map[src1]
                src2_terms = normalized_terms_map[src2]

                token_type = token_type_map[src1]

                for n1, n2 in itertools.product(src1_terms, src2_terms):
                    if not G_final.nodes[n1]['type'] == G_final.nodes[n2]['type'] == token_type:
                        continue

                    # Some sentences lead to subsumption relationship between 1st/3rd parties.
                    # Workaround: Simply ignore all subsumption edges to "we"
                    if n2 == "we":
                        logging.warning("Invalid subsumption: %r -> %r", n1, n2)
                        continue

                    if not G_final.has_edge(n1, n2, key=relationship):
                        dag_add_edge(G_final, n1, n2, key=relationship, ref=[])

                    if G_final.has_edge(n1, n2, key=relationship):
                        G_final[n1][n2][relationship]["ref"].extend((src1, src2))

        def merge_collect_graph():
            """Step 8: Populate COLLECT edges in G_final from G_collect."""

            for src1, src2, relationship, edge_data in G_collect.edges(keys=True, data=True):
                src1_terms = normalized_terms_map[src1]
                src2_terms = normalized_terms_map[src2]
                edge_purposes = sorted(edge_data["purposes"])

                for n1, n2 in itertools.product(src1_terms, src2_terms):
                    if G_final.nodes[n1]['type'] == "ACTOR" and G_final.nodes[n2]['type'] == "DATA":
                        if not G_final.has_edge(n1, n2, key=relationship):
                            dag_add_edge(G_final, n1, n2, key=relationship, ref=[], purposes=defaultdict(list))

                        if G_final.has_edge(n1, n2, key=relationship):
                            G_final[n1][n2][relationship]["ref"].extend((src1, src2))
                            purpose_dict = G_final[n1][n2][relationship]["purposes"]

                            for purpose, text in edge_purposes:
                                purpose_dict[purpose].append(text)

        def finalize():
            """Step 9: Finalize"""

            # Clean-up zero degree nodes
            for node in list(G_final.nodes()):
                if G_final.degree(node) == 0:
                    G_final.remove_node(node)

            # Turn src on edges into text, remove duplications while keep the order
            for _, _, _, data in G_final.edges(keys=True, data=True):
                all_text = []

                for src in sorted(set(data.pop("ref"))):
                    sentence = document.get_token_with_src(src).sent.text.strip()
                    all_text.append(sentence)

                dedup_text = set()

                for text in sorted(all_text, key=len, reverse=True):
                    for existing_text in dedup_text:
                        if text in existing_text:
                            break
                    else:
                        dedup_text.add(text)

                data["text"] = list(filter(dedup_text.__contains__, all_text))

        build_phrase_map_from_ent_labels()
        build_collect_graph()
        build_subsum_and_coref_graphs()
        contract_coref_nodes()
        reduce_graph()
        normalize_terms()
        merge_subsum_graph()
        merge_collect_graph()
        finalize()

        return G_final


def trim_graph(graph):
    """Remove SUBSUM edges that do not generate any collect(n, d) relations"""
    important_nodes = set()

    for d1, d2, rel in graph.edges(keys=True):
        if rel in CollectionAnnotator.EDGE_TYPES:
            important_nodes.add(d1)
            important_nodes.add(d2)

    if important_nodes:
        dijkstra_results = nx.multi_source_dijkstra_path_length(graph, important_nodes)
    else:
        dijkstra_results = {}

    important_nodes.update(dijkstra_results.keys())

    useless_nodes = set(graph.nodes) - important_nodes
    logging.info("Nodes to remove: %s", ", ".join(sorted(useless_nodes)))
    returned_graph = graph.copy()
    returned_graph.remove_nodes_from(useless_nodes)

    return returned_graph


def colorize_graph(graph):
    new_graph = nx.MultiDiGraph()

    for label, data in graph.nodes(data=True):
        new_graph.add_node(label, label=label, type=data["type"])

    for i, (u, v, rel, data) in enumerate(graph.edges(keys=True, data=True)):
        new_graph.add_edge(u, v, key=f"e{i}", relationship=rel, text="\n".join(data["text"]))

    return new_graph


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp", default="", help="NLP model directory")
    parser.add_argument("--purpose-classification", default="", help="Purpose classification model directory")
    parser.add_argument("-p", "--phrase-map", default="", help="Path to phrase_map.yml")
    parser.add_argument("-e", "--entity-info", default="", help="Path to entity_info.json")
    parser.add_argument("-v", "--variant", choices=["original", "extended", "per_sentence", "per_section"],
                        default="original", help="Variant of the graph")
    parser.add_argument("--pretty", action="store_true", help="Generate pretty GraphML graph for visualization")
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    args = parser.parse_args()

    # Load resources from extra-data folder unless overridden in the args
    with pkg_resources.path(poligrapher, "extra-data") as extra_data:
        if not args.purpose_classification:
            args.purpose_classification = extra_data / "purpose_classification"

        if not args.phrase_map:
            args.phrase_map = extra_data / "phrase_map.yml"

        if not args.entity_info:
            args.entity_info = extra_data / "entity_info.json"

    nlp = setup_nlp_pipeline(args.nlp)
    graph_builder = GraphBuilder(args.phrase_map, args.entity_info, args.purpose_classification, args.variant)

    for d in args.workdirs:
        logging.info("Processing %s ...", d)

        document = PolicyDocument.load(d, nlp)

        # Full graph includes unused SUBSUM edges -- For debug use
        full_graph = graph_builder.build_graph(document)

        with open(os.path.join(d, f"graph-{args.variant}.full.yml"), "w", encoding="utf-8") as fout:
            yaml_dump_graph(full_graph, fout)

        # Trimmed graph excludes unused SUBSUM edges
        trimmed_graph = trim_graph(full_graph)

        with open(os.path.join(d, f"graph-{args.variant}.yml"), "w", encoding="utf-8") as fout:
            yaml_dump_graph(trimmed_graph, fout)

        # GraphML version for visualization
        if args.pretty:
            colored_graph = colorize_graph(trimmed_graph)
            nx.write_graphml(colored_graph, os.path.join(d, f"graph-{args.variant}.graphml"))


if __name__ == "__main__":
    main()
