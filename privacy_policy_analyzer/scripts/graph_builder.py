#!/usr/bin/env python3

import argparse
import os

import networkx as nx
import spacy
from privacy_policy_analyzer.document import PolicyDocument
from privacy_policy_analyzer.phrase_normalization import EntityMatcher, RuleBasedPhraseNormalizer


def build_graph(document, nlp, phrase_normalizer, entity_matcher):
    full_doc = document.get_full_doc(nlp, True)
    normalized_terms = {}

    unspecific_terms = {
        "info", "data",
        "third party", "service", "provider",
        "whom", "who", "which", "this", "that", "these", "those", "they", "other",
    }
    ignored_terms = {"you", "user", "other", "people"}
    unspec_term_count = 0

    # Find the normalized term of each noun phrase
    coref_dependency = nx.DiGraph()
    for ent in full_doc.ents:
        idx = ent.start
        term = None

        #if ent._.ent_type in ["ACTOR", "NN"]:
        #    term = entity_matcher.match_name(ent.text)

        if term is None:
            if (term := phrase_normalizer.normalize(ent)) in unspecific_terms:
                term = term + f"_{unspec_term_count}"
                unspec_term_count += 1

        for _, dest_token, relationship in document.get_all_links(ent[0]):
            if relationship == "COREF":
                coref_dependency.add_edge(dest_token.i, idx)
                break
        else:
            normalized_terms[idx] = term
            print(ent, normalized_terms[idx], sep=" -> ")

    # Normalize coreferences to the terms they refer to
    for i in nx.topological_sort(coref_dependency):
        if i not in normalized_terms:
            dest_i, _ = next(iter(coref_dependency.in_edges(i)))
            normalized_terms[i] = normalized_terms[dest_i]
            print(full_doc[i]._.ent, normalized_terms[idx])

    terms_to_remove = []
    for k, v in normalized_terms.items():
        if v in ignored_terms:
            terms_to_remove.append(k)

    for k in terms_to_remove:
        normalized_terms.pop(k)

    knowledge_graph = nx.DiGraph()

    for idx, term in normalized_terms.items():
        token = full_doc[idx]

        if term not in knowledge_graph.nodes:
            knowledge_graph.add_node(term, source=[])

        knowledge_graph.nodes[term]["source"].append(token._.ent.text)


    for idx, term in normalized_terms.items():
        token = full_doc[idx]

        for _, dest_token, relationship in document.get_all_links(token):
            if dest_token.i not in normalized_terms:
                continue
            elif relationship not in ["SUBSUM", "COLLECT"]:
                continue

            dest_term = normalized_terms[dest_token.i]
            if dest_term != term:
                knowledge_graph.add_edge(term, dest_term, label=relationship,
                                         sentences=[token.sent.text, dest_token.sent.text])

    # Remove unspecific terms by connecting nodes at two sides
    for term in filter(lambda t: "_" in t, knowledge_graph.nodes):
        for v1, _, data1 in knowledge_graph.in_edges(term, data=True):
            for _, v2, data2 in knowledge_graph.out_edges(term, data=True):
                if data1["label"] == data2["label"] == "COLLECT":
                    # NLP gliches
                    continue
                elif data1["label"] == data2["label"] == "SUBSUM":
                    knowledge_graph.add_edge(v1, v2, label="SUBSUM")
                elif {data1["label"], data2["label"]} == {"COLLECT", "SUBSUM"}:
                    knowledge_graph.add_edge(v1, v2, label="COLLECT")

    knowledge_graph.remove_nodes_from([n for n in knowledge_graph.nodes if "_" in n])

    # Only keep the connected subgraph that has COLLECT relationships
    critical_nodes = set()
    for v1, v2, data in knowledge_graph.edges(data=True):
        if data["label"] == "COLLECT":
            critical_nodes.add(v1)
            critical_nodes.add(v2)

    if len(critical_nodes) > 0:
        shortest_paths = nx.multi_source_dijkstra_path_length(knowledge_graph, critical_nodes)
        trim_graph = knowledge_graph.subgraph(list(shortest_paths.keys()))
        return trim_graph
    else:
        return nx.DiGraph()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--entity-map", required=True, help="Path to entity_map.json")
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    args = parser.parse_args()

    phrase_normalizer = RuleBasedPhraseNormalizer()
    entity_matcher = EntityMatcher(args.entity_map)
    nlp = spacy.load("en_core_web_sm")

    for d in args.workdirs:
        print(f"Processing {d} ...")

        document = PolicyDocument(d)
        knowledge_graph = build_graph(document, nlp, phrase_normalizer, entity_matcher)
        nx.write_gml(knowledge_graph, os.path.join(d, "graph.gml"))


if __name__ == "__main__":
    main()
