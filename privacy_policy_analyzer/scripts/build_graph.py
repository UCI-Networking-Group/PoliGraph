#!/usr/bin/env python3

import argparse
import itertools
import os
import re
from collections import deque

import networkx as nx
import spacy
import yaml
from privacy_policy_analyzer.document import PolicyDocument
from privacy_policy_analyzer.named_entity_recognition import ACTOR_KEYWORDS, DATATYPE_KEYWORDS
from privacy_policy_analyzer.phrase_normalization import EntityMatcher, RuleBasedPhraseNormalizer


def expand_phrase(token):
    doc = token.doc

    def has_alpha(s):
        return re.search(r"[a-zA-Z]", s, flags=re.ASCII) is not None

    def should_include(current_token, in_left_modifier=False):
        if current_token._.src is None:
            return False

        match current_token.dep_:
            case "dep" | "meta":
                # NLP glitches
                return current_token.ent_type_ != ""
            case "punct":
                # Keep punct only it's inside a word (e.g. health-related info) or it's a quotation mark
                flag = (current_token.i > 0 and current_token.i + 1 < len(doc)
                        and doc[current_token.i - 1].whitespace_ == "" and has_alpha(doc[current_token.i - 1].text)
                        and current_token.whitespace_ == "" and has_alpha(doc[current_token.i + 1].text))
                flag |= current_token.lemma_ in "'\""
                flag |= current_token.ent_type_ != ""
                flag |= in_left_modifier

                if not flag:
                    return False

        if current_token.dep_ in ["amod", "nmod", "nummod", "compound"] and current_token.i < current_token.head.i:
            in_left_modifier = True

        for c in current_token.children:
            if not should_include(c, in_left_modifier):
                return False

        return True

    # Noun phrase, try to include simple children
    left_idx = token.i
    for child in sorted(token.lefts, reverse=True):
        if should_include(child):
            child_indices = sorted(t.i for t in child.subtree)

            if (child_indices[-1] - child_indices[0] == len(child_indices) - 1
                and left_idx - 1 == child_indices[-1]):
                left_idx = child_indices[0]
            else:
                break
        else:
            break

    right_idx = token.i + 1
    for child in sorted(token.rights):
        if should_include(child):
            child_indices = sorted(t.i for t in child.subtree)

            if (child_indices[-1] - child_indices[0] == len(child_indices) - 1
                and right_idx == child_indices[0]):
                right_idx = child_indices[-1] + 1
            else:
                break
        else:
            break

    if doc[right_idx - 1].dep_ == "cc":
        right_idx -= 1

    return doc[left_idx:right_idx]


def simplify_phrase(phrase):
    trivial_words = frozenset([
        "some", "all", "any", "type", "variety", "category", "example",
        "more", "such", "other", "following", "additional",
        "certain", "similar", "limited", "various", "detailed",
        "further", "enough", "e.g.", "i.e.",
    ])

    def dfs(token, state):
        if token.dep_ in trivial_words:
            return

        match state:
            case "compound" | "nmod" | "pobj" | "nsubj" | "dobj":
                next_states = {"compound", "amod", "nmod", "prep", "relcl", "acl"}
            case "prep":
                next_states = {"pobj"}
            case "amod":
                next_states = {"advmod", "npadvmod"}
            case "relcl" | "acl":
                next_states = {"nsubj", "dobj"}
            case _:
                yield token
                return

        for t in token.children:
            if t in phrase and t.dep_ in next_states:
                yield from dfs(t, t.dep_)

        yield token

    return sorted(dfs(phrase.root, "compound"))


class GraphBuilder:
    def __init__(self, phrase_map, entity_map):
        with open(phrase_map, "r", encoding="utf-8") as fin:
            phrase_map_rules = yaml.safe_load(fin)

        self.entity_mapper = EntityMatcher(entity_map)
        self.data_phrase_normalizer = RuleBasedPhraseNormalizer(phrase_map_rules["DATA"])
        self.actor_phrase_normalizer = RuleBasedPhraseNormalizer(phrase_map_rules["ACTOR"])

    def build_graph(self, document: PolicyDocument):
        stage1_graph = self.__build_graph_stage1(document)
        stage2_graph = self.__build_graph_stage2(stage1_graph)
        return stage2_graph

    def __build_graph_stage1(self, document: PolicyDocument):
        stage1_graph = nx.DiGraph()

        # Step 1: Infer phrase type using NER label
        for src in document.token_relationship.nodes:
            token = document.get_token_with_src(src)

            match token._.ent_type:
                case "NN":
                    continue
                case "DATA":
                    stage1_graph.add_node(src, token=token, type="DATA")
                case "ACTOR":
                    stage1_graph.add_node(src, token=token, type="ACTOR")
                case "OTHER":
                    # Will eventually remove these nodes
                    stage1_graph.add_node(src, token=token, type="OTHER")

        # Step 2: Infer phrase type using COLLECT relationship
        for src1, src2, data in document.token_relationship.edges(data=True):
            if data["relationship"] == "COLLECT":
                token1 = document.get_token_with_src(src1)
                token2 = document.get_token_with_src(src2)

                if src1 not in stage1_graph:
                    stage1_graph.add_node(src1, token=token1, type="ACTOR")

                if src2 not in stage1_graph:
                    stage1_graph.add_node(src2, token=token2, type="DATA")

                if [stage1_graph.nodes[s]["type"] for s in (src1, src2)] == ["ACTOR", "DATA"]:
                    stage1_graph.add_edge(src1, src2, relationship="COLLECT")

        # Step 3: Infer phrase type using SUBSUM / COREF relationship
        bfs_queue = deque(stage1_graph.nodes)

        while len(bfs_queue) > 0:
            src1 = bfs_queue.popleft()
            the_type = stage1_graph.nodes[src1]["type"]

            in_edge_view = document.token_relationship.in_edges(src1, data=True)
            out_edge_view = document.token_relationship.out_edges(src1, data=True)

            for edge_from, edge_to, data in itertools.chain(in_edge_view, out_edge_view):
                if data["relationship"] in ["SUBSUM", "COREF"] and edge_from != edge_to:
                    src2 = edge_to if src1 == edge_from else edge_from

                    if src2 not in stage1_graph:
                        token2 = document.get_token_with_src(src2)
                        stage1_graph.add_node(src2, token=token2, type=the_type)
                        bfs_queue.append(src2)

                    if stage1_graph.has_edge(edge_to, edge_from):
                        # example: third parties, such as those parties who...
                        continue

                    if stage1_graph.nodes[src2]["type"] == the_type:
                        stage1_graph.add_edge(edge_from, edge_to, relationship=data["relationship"])

        # Step 4: Remove phrases that are not ACTOR / DATA
        to_remove = []
        for src, data in stage1_graph.nodes(data=True):
            if data["type"] == "OTHER":
                to_remove.append(src)

        stage1_graph.remove_nodes_from(to_remove)

        # Step 5: Infer meaning of each phrase using normalizers
        for src, data in stage1_graph.nodes(data=True):
            token = data["token"]
            data["phrase"] = phrase = expand_phrase(token)
            data["normalized_terms"] = set()

            has_coref = False

            for _, _, edge_data in stage1_graph.out_edges(src, data=True):
                if edge_data["relationship"] == "COREF":
                    has_coref = True
                    break

            if has_coref:
                continue

            match data["type"]:
                case "DATA":
                    normalized_terms = list(self.data_phrase_normalizer.normalize(phrase))

                    if len(normalized_terms) == 0:
                        simplified = " ".join(t.lemma_ for t in simplify_phrase(phrase))
                        if simplified and simplified not in DATATYPE_KEYWORDS:
                            normalized_terms.append(simplified)
                case "ACTOR":
                    normalized_terms = list(self.actor_phrase_normalizer.normalize(phrase))

                    if any(t.pos_ == "PROPN" for t in phrase):
                        if entity_name := self.entity_mapper.match_name(phrase.text):
                            normalized_terms.append(entity_name)

                    if len(normalized_terms) == 0:
                        simplified = " ".join(t.lemma_ for t in simplify_phrase(phrase))
                        if simplified and simplified != "you" and simplified not in ACTOR_KEYWORDS:
                            normalized_terms.append(simplified)
                case _:
                    raise ValueError("Invalid type")

            print(phrase, normalized_terms)
            data["normalized_terms"].update(normalized_terms)

        return stage1_graph

    def __build_graph_stage2(self, stage1_graph: nx.DiGraph):
        stage2_graph = nx.DiGraph()
        s1s2_node_map = dict()

        for stage1_node in reversed(list(nx.topological_sort(stage1_graph))):
            stage1_node_data = stage1_graph.nodes[stage1_node]
            s1s2_node_map[stage1_node] = mapped_nodes = []
            stage2_edges = set()

            for _, stage1_endpoint, edge_data in stage1_graph.out_edges(stage1_node, data=True):
                match edge_data["relationship"]:
                    case "COREF":
                        for n in s1s2_node_map[stage1_endpoint]:
                            stage2_graph.nodes[n]["source_tokens"].append(stage1_node)
                            mapped_nodes.append(n)
                    case _ as relationship:
                        for n in s1s2_node_map[stage1_endpoint]:
                            stage2_edges.add((n, relationship))

            if len(mapped_nodes) == 0:
                for term in stage1_node_data["normalized_terms"]:
                    mapped_nodes.append(term)

                    if term not in stage2_graph:
                        stage2_graph.add_node(term, source_tokens=[stage1_node])

            for n1 in mapped_nodes:
                for n2, relationship in stage2_edges:
                    stage2_graph.add_edge(n1, n2, label=relationship)

        return stage2_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp", required=True, help="NLP model directory")
    parser.add_argument("-p", "--phrase-map", required=True, help="Path to phrase_map.yml")
    parser.add_argument("-e", "--entity-info", required=True, help="Path to entity_info.json")
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    args = parser.parse_args()

    nlp = spacy.load(args.nlp)
    graph_builder = GraphBuilder(args.phrase_map, args.entity_info)


    for d in args.workdirs:
        print(f"Processing {d} ...")

        document = PolicyDocument.load(d, nlp)
        knowledge_graph = graph_builder.build_graph(document)

        nx.write_gml(knowledge_graph, os.path.join(d, "graph.gml"), stringizer=str)


if __name__ == "__main__":
    main()
