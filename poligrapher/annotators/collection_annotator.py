import importlib.resources as pkg_resources
import itertools
from collections import defaultdict

import networkx as nx
import yaml
from spacy.tokens import Token

import poligrapher

from .base import BaseAnnotator

IMPORTANT_DEPS_OF_POS = {
    "VERB": {
        "nsubj", "dobj", "nsubjpass", "dative", "agent", "prep",
        "ccomp", "xcomp",
        "pobj",  "pcomp", # information _regarding_ ...
    },
    "NOUN": {"prep", "relcl", "appos"},
    "ADP": {"pobj", "pcomp"},
}
IMPORTANT_DEPS_OF_POS["PRON"] = IMPORTANT_DEPS_OF_POS["PROPN"] = IMPORTANT_DEPS_OF_POS["NOUN"]


def build_dependency_graph(root_token: Token):
    """Transform dependency tree into a dependency graph for easier parsing.

    The original dependency tree has many trivial details and unwanted features
    that make parsing difficult. This function does:
      - Flatten conjunct tokens to become children of the conjunct head.
      - Transform passive voices to the same structure as active voices.
      - Avoid interrogative sentences (or clauses).
      - Avoid uninteresting dependencies.
      - ......
    """

    def is_interrogative(token: Token):
        while token.dep_ == "conj":
            token = token.head

        aux_pos = 0xFFFF
        subj_pos = 0

        for child in token.lefts:
            if child.tag_ in ('WRB', 'WP'):
                return True
            elif child.dep_ == "nsubj":
                subj_pos = child.i
            elif child.dep_ == "aux":
                aux_pos = child.i

        return aux_pos < subj_pos

    def is_negative(token: Token):
        if any(c.dep_ == "neg" for c in token.lefts):
            return True

        while token.dep_ == "conj":
            token = token.head

            if any((c.dep_, c.lemma_) == ("cc", "but") for c in token.rights):
                return False
            elif any(c.dep_ == "neg" for c in token.lefts):
                return True

        return False

    def handle_agent(parent_token: Token, agent_token: Token):
        negation_flag = is_negative(agent_token)  # "not by"?

        try:
            first_grand_child = next(filter(lambda t: t.dep_ == "pobj", agent_token.children))
        except StopIteration:
            return

        for grand_child in itertools.chain([first_grand_child], first_grand_child.conjuncts):
            modified_dep_tree.add_edge(parent_token, grand_child, dep="subj")
            modified_dep_tree.nodes[grand_child]["negation"] = negation_flag
            dfs(grand_child)

    def handle_dative(parent_token: Token, dative_token: Token):
        if dative_token.pos_ == "ADP":
            # be given to us (to: ADP) => link "us" to "given"
            try:
                first_grand_child = next(filter(lambda t: t.dep_ == "pobj", dative_token.children))
            except StopIteration:
                return

            negation_flag = is_negative(dative_token)  # "not to"?

            for grand_child in itertools.chain([first_grand_child], first_grand_child.conjuncts):
                modified_dep_tree.add_edge(parent_token, grand_child, dep="dative")
                modified_dep_tree.nodes[grand_child]["negation"] = negation_flag
                dfs(grand_child)
        elif dative_token.pos_ in ('NOUN', 'PRON', 'PROPN'):
            # give us X (us: PRON)
            modified_dep_tree.add_edge(parent_token, dative_token, dep="dative")
            dfs(dative_token)

    def handle_xcomp(parent_token: Token, xcomp_root_token: Token):
        modified_dep_tree.add_edge(parent_token, xcomp_root_token, dep="xcomp")
        dfs(xcomp_root_token)

        for _, node, data in modified_dep_tree.out_edges(xcomp_root_token, data=True):
            if data["dep"] == "subj":
                break
        else:
            # If xcomp_root has no subject, assign any object of parent node as its subject
            for _, node, data in modified_dep_tree.out_edges(parent_token, data=True):
                if data["dep"] == "obj":
                    modified_dep_tree.add_edge(xcomp_root_token, node, dep="subj")

    def handle_ccomp(parent_token: Token, ccomp_root_token: Token):
        modified_dep_tree.add_edge(parent_token, ccomp_root_token, dep="ccomp")
        dfs(ccomp_root_token)

    def handle_appos(parent_token: Token, appos_token: Token):
        modified_dep_tree.add_node(appos_token, negation=is_negative(parent_token))

        for grand_parent_token, _, data in modified_dep_tree.in_edges(parent_token, data=True):
            modified_dep_tree.add_edge(grand_parent_token, appos_token, **data)

    def handler_factory(graph_dep: str):
        def func(parent_token: Token, child_token: Token):
            modified_dep_tree.add_edge(parent_token, child_token, dep=graph_dep)
            dfs(child_token)

        return func

    def find_all_children(current_token: Token):
        children = list(current_token.children)
        existing_deps = {t.dep_ for t in current_token.children}
        conjuncts = current_token.conjuncts

        if (leftist_conj := min(conjuncts, default=current_token)) != current_token:
            # Copy children from the first left conjunct
            # e.g. "We collect and share ..." => Link "we" as subj of "share"
            for child in leftist_conj.children:
                if child.i < leftist_conj.i and child.dep_ not in existing_deps:
                    existing_deps.add(child.dep_)
                    children.append(child)

        for right_conj in filter(current_token.__lt__, conjuncts):
            # Copy children from right conjuncts
            # e.g. "We collect and share X" => Link "X" as obj of "collect"
            for child in right_conj.children:
                if child.dep_ not in existing_deps:
                    existing_deps.add(child.dep_)
                    children.append(child)

        important_deps = IMPORTANT_DEPS_OF_POS.get(current_token.pos_, [])
        return sorted(filter(lambda t: t.dep_ in important_deps, children))

    dep_handlers = {
        "nsubj" : handler_factory("subj"),
        "agent" : handle_agent,
        "dative": handle_dative,
        "appos" : handle_appos,
        "xcomp" : handle_xcomp,
        "ccomp" : handle_ccomp,
    }
    dep_handlers["dobj"] = dep_handlers["nsubjpass"] = dep_handlers["pobj"] = handler_factory("obj")

    def dfs(current_token: Token):
        # Check negation
        modified_dep_tree.nodes[current_token]["negation"] = is_negative(current_token)

        # Link children
        for immediate_child in find_all_children(current_token):
            conjuncts = immediate_child.conjuncts
            dependency = immediate_child.dep_

            if dependency not in dep_handlers:
                dep_handlers[dependency] = handler_factory(dependency)

            for child in itertools.chain([immediate_child], conjuncts):
                if not is_interrogative(child):
                    dep_handlers[dependency](current_token, child)

    modified_dep_tree = nx.DiGraph()
    modified_dep_tree.add_node("")

    if root_token.sent[-1].lemma_ == "?":
        return modified_dep_tree

    for token in itertools.chain([root_token], root_token.conjuncts):
        if not is_interrogative(token):
            modified_dep_tree.add_edge("", token, dep="root")
            dfs(token)

    return modified_dep_tree


class TokenMatcher:
    """Matcher to match one token, used by DependencyPatternMatcher."""

    def __init__(self, rule: str, token_map: dict[str, list[str]]):
        lemma_spec, *res = rule.split(":", 1)
        dep_spec = res[0] if res else ""

        self.match_lemmas = set()

        for item in filter(None, lemma_spec.split('|')):
            if item.startswith('*'):
                self.match_lemmas.update(token_map[item[1:]])
            else:
                self.match_lemmas.add(item)

        self.match_deps = set(filter(None, dep_spec.split('|')))

    def match(self, token: Token, dependency: str):
        return (
            len(self.match_deps) == 0
            or dependency in self.match_deps
        ) and (
            len(self.match_lemmas) == 0
            or token.lemma_.lower() in self.match_lemmas
        )


class DependencyPatternMatcher:
    counter = itertools.count()

    def __init__(self, pattern_spec: dict, token_map: dict[str, list[str]]):
        # For logging / debugging
        self.original_pattern: list[str] = pattern_spec
        self.id: int = next(DependencyPatternMatcher.counter)

        # Token matcher for the root
        self.root_matcher = TokenMatcher(pattern_spec["root"], token_map)

        # Patterns to match chains (paths) in the dependency tree that start
        # from the root token.
        self.all_chains: list[list[tuple[TokenMatcher, str]]] = []

        # Chains that must be matched
        self.required_indices = []

        for match_rule in pattern_spec["match"]:
            if match_rule.startswith("!"):
                self.required_indices.append(len(self.all_chains))
                match_rule = match_rule[1:]

            chain = []
            self.all_chains.append(chain)

            for item in match_rule.split(","):
                token_rule, *res = item.split('@', 1)
                save_to = res[0] if res else None
                chain.append((TokenMatcher(token_rule, token_map), save_to))

        self.transform_rule: list[str] = pattern_spec["transform"]

    def match(self, dependency_graph: nx.DiGraph):
        named_tokens: dict[str, list[Token]] = defaultdict(list)
        unmatched_required_indices: set[int] = set()

        def on_success(chain_idx: int, progress: list[Token]):
            for (_, save_to), node in zip(self.all_chains[chain_idx], progress):
                if dependency_graph.nodes[node]['negation']:
                    # Only handle negation at the root and give up on negations
                    # in the chain. Cases like following will be missed:
                    #   "We share ... with X but _not with_ Y"
                    break

                named_tokens[save_to].append(node)
            else:
                unmatched_required_indices.discard(chain_idx)

        def dfs_match_chain(node: Token, dependency: str, match_progress: dict[int, list[Token]]):
            """DFS to find all chains (paths) that match chain patterns"""
            continue_chains = []

            for chain_idx, progress in match_progress.items():
                chain = self.all_chains[chain_idx]

                if chain[len(progress)][0].match(node, dependency):
                    progress.append(node)

                    if len(progress) == len(chain):
                        on_success(chain_idx, progress)
                    else:
                        continue_chains.append(chain_idx)
                else:
                    progress.append(None)

            if len(continue_chains) > 0:
                for _, child, data in dependency_graph.out_edges(node, data=True):
                    dfs_match_chain(child, data["dep"], {i: match_progress[i] for i in continue_chains})

            for progress in match_progress.values():
                progress.pop()

        for _, v, data in dependency_graph.edges(data=True):
            if self.root_matcher.match(v, data["dep"]):
                named_tokens.clear()
                unmatched_required_indices.clear()
                unmatched_required_indices.update(self.required_indices)

                for _, child, data in dependency_graph.out_edges(v, data=True):
                    dfs_match_chain(child, data["dep"], {i: [] for i in range(len(self.all_chains))})

                if not unmatched_required_indices:
                    action, *arg_names = self.transform_rule
                    arg_list = []

                    for item in arg_names:
                        for arg_name in item.split("|"):
                            if arg_name in named_tokens:
                                arg_list.append(named_tokens[arg_name])
                                break
                        else:
                            arg_list.append([])

                    negation_flag = dependency_graph.nodes[v]["negation"]
                    yield action, negation_flag, arg_list


class CollectionAnnotator(BaseAnnotator):
    """The collection annotator"""

    ACTION_MAP = {
        ("COLLECT", False): [(0, 1, "COLLECT")],
        ("COLLECT", True):  [(0, 1, "NOT_COLLECT")],
        ("SHARE", False):   [(2, 1, "BE_SHARED"),
                             (0, 1, "COLLECT")],
        ("SHARE", True):    [(2, 1, "NOT_BE_SHARED")],
        ("SELL", False):    [(2, 1, "BE_SOLD"),
                             (0, 1, "COLLECT")],
        ("SELL", True):     [(2, 1, "NOT_BE_SOLD")],
        ("USE", False):     [(0, 1, "USE")],
        ("USE", True):      [(0, 1, "NOT_USE")],
        ("STORE", False):   [(0, 1, "STORE")],
        ("STORE", True):    [(0, 1, "NOT_STORE")],
    }

    EDGE_TYPES = frozenset(edge_type for li in ACTION_MAP.values() for _, _, edge_type in li)
    NEGATIVE_EDGE_TYPES = frozenset(filter(lambda t: t.startswith("NOT_"), EDGE_TYPES))
    POSITIVE_EDGE_TYPES = EDGE_TYPES - NEGATIVE_EDGE_TYPES

    def __init__(self, nlp):
        super().__init__(nlp)

        with pkg_resources.open_text(poligrapher, "verb_patterns.yml") as fin:
            config = yaml.safe_load(fin)

        token_map = config["token_map"]
        self.pattern_matchers: list[DependencyPatternMatcher] = []

        for p in config["patterns"]:
            self.pattern_matchers.append(DependencyPatternMatcher(p, token_map))

    def annotate(self, document):
        def validate_type(token_list, target_type):
            """Use existing SUBSUM / COREF edges to check if the tokens to be
            linked has compatible NER type"""

            # Use BFS here to avoid any loop.
            bfs_queue = list(token_list)
            visited_tokens = set(token_list)
            i = 0

            while i < len(bfs_queue):
                tok = bfs_queue[i]
                i += 1

                if tok.ent_type_ == target_type:
                    return True
                elif tok.ent_type_ == "NN":
                    for _, linked_token, relationship in document.get_all_links(tok):
                        if relationship in ("SUBSUM", "COREF") and linked_token not in visited_tokens:
                            bfs_queue.append(linked_token)
                            visited_tokens.add(linked_token)

            return False

        def make_links(action, neg_flag, args):
            nonlocal matcher, sent

            for entity_idx, data_idx, relation in self.ACTION_MAP[action, neg_flag]:
                entity_tokens = args[entity_idx]
                data_tokens = args[data_idx]

                if validate_type(entity_tokens, "ACTOR") and validate_type(data_tokens, "DATA"):
                    self.logger.info("Collection statement (rule #%d): %r", matcher.id, sent.text)

                    for entity in entity_tokens:
                        for dtype in data_tokens:
                            self.logger.info("Edge %s (%s): %r -> %r", relation, action, entity._.ent, dtype._.ent)
                            document.link(entity, dtype, relation)

        for doc in document.iter_docs():
            for sent in doc.sents:
                dependency_graph = build_dependency_graph(sent.root)

                for matcher in self.pattern_matchers:
                    for action, neg_flag, arg_list in matcher.match(dependency_graph):
                        make_links(action, neg_flag, arg_list)
