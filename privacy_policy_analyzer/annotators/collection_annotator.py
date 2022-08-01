import importlib.resources as pkg_resources
import itertools
from typing import Optional

import yaml
from spacy.tokens import Span, Token

import privacy_policy_analyzer
from ..utils import get_conjuncts
from .base import BaseAnnotator


class TokenMatcher:
    def __init__(self, allow_lemmas=None, allow_pos=None, allow_dep=None, save_to_variable=None):
        self.lemmas = allow_lemmas
        self.pos = allow_pos
        self.dep = allow_dep
        self.variable = save_to_variable

    def match(self, token: Token, overrided_dep: Optional[str]=None):
        if self.lemmas:
            if token.lemma_ not in self.lemmas:
                return None

        if self.pos:
            if token.pos_ != self.pos:
                return None

        if self.dep:
            if (overrided_dep or token.dep_) != self.dep:
                return None

        return (token, self.variable)


class ChainMatcher:
    CONTINUE = 1
    SUCCEEDED = 2
    FAILED = 3

    def __init__(self, chain: list[TokenMatcher], on_success):
        self.chain = chain
        self.current_token_is_matched = True
        self.cursor = -1
        self.matched_tokens = []
        self.previous_status = []

        self.on_success = on_success

    def match(self, token: Optional[Token], inherited_dep: Optional[str]=None):
        self.previous_status.append((self.cursor, self.current_token_is_matched, len(self.matched_tokens)))

        if token and token.dep_ in ["conj", "appos"]:
            # Stay in the current state. Cursor == -1 means conjuncts of the root verb
            if self.cursor == -1:
                return ChainMatcher.CONTINUE
        else:
            # Next state. if the old one hasn't been matched, the matching fails
            if not self.current_token_is_matched:
                return ChainMatcher.FAILED

            self.cursor += 1
            self.current_token_is_matched = False
            assert self.cursor < len(self.chain)

        if token and (m := self.chain[self.cursor].match(token, inherited_dep)) is not None:
            self.matched_tokens.append(m)
            self.current_token_is_matched = True

            # Successfully matched the full chain
            if self.cursor + 1 == len(self.chain):
                self.on_success(self.matched_tokens)
                return ChainMatcher.SUCCEEDED

        return ChainMatcher.CONTINUE

    def rollback(self):
        self.cursor, self.current_token_is_matched, matched_token_pos = self.previous_status.pop()

        while len(self.matched_tokens) > matched_token_pos:
            self.matched_tokens.pop()


class SentencePattern:
    counter = itertools.count()

    def __init__(self, pattern, token_map):
        # For logging / debugging
        self.original_pattern = pattern
        self.rule_id = next(SentencePattern.counter)

        self.root_lemmas = set()

        for k in pattern["root"]:
            if k.isupper():
                self.root_lemmas.update(token_map[k])
            else:
                self.root_lemmas.add(k)

        self.all_chains = []

        for p in pattern["match"]:
            if p.startswith("!"):
                is_important = True
                p = p[1:]
            else:
                is_important = False

            chain = []
            self.all_chains.append((chain, is_important))

            for item in p.split(","):
                lemma_spec, *properties = item.split(":")

                allow_lemmas = set()
                kwargs = dict(allow_lemmas=allow_lemmas)

                for lemma in lemma_spec.split("/"):
                    if lemma.startswith("@"):
                        kwargs["save_to_variable"] = lemma[1:]
                    elif lemma.isupper():
                        allow_lemmas.update(token_map[lemma])
                    else:
                        allow_lemmas.add(lemma)

                if len(allow_lemmas) == 0 or "*" in allow_lemmas:
                    kwargs.pop("allow_lemmas")

                for prop in properties:
                    if prop.islower():
                        kwargs["allow_dep"] = prop
                    elif prop.isupper():
                        kwargs["allow_pos"] = prop
                    else:
                        assert False

                chain.append(TokenMatcher(**kwargs))

        self.transform_rule = pattern["transform"]

    def __repr__(self):
        return f"SentencePattern: {self.original_pattern}"

    def match_root(self, root_token):
        if root_token.lemma_ not in self.root_lemmas:
            return None

        sentence_matcher = SentenceMatcher(self, root_token)

        for chain, is_important in self.all_chains:
            sentence_matcher.add_chain(chain, is_important)

        return sentence_matcher


class SentenceMatcher:
    def __init__(self, parent_pattern: SentencePattern, root_token):
        self.pattern = parent_pattern
        self.root_token = root_token
        self.chain_matchers: list[ChainMatcher] = []
        self.unmatched: list[bool] = []
        self.matched_data: dict[str, tuple(Token, int)] = {}

    def add_chain(self, chain, is_important):
        idx = len(self.chain_matchers)
        self.chain_matchers.append(ChainMatcher(chain, lambda x: self.on_success(x, idx)))
        self.unmatched.append(is_important)

    @property
    def fully_matched(self):
        return len(self.matched_data) > 0 and not any(self.unmatched)

    def on_success(self, matched_chain, idx):
        for token, variable in matched_chain:
            if any(t.dep_ == "neg" for t in token.children):
                break

            if variable:
                conj_list = [token]
                self.matched_data[variable] = conj_list

                for conj in sorted(get_conjuncts(token), key=lambda t: t.i):
                    if conj.i <= token.i:
                        continue
                    elif any(t.dep_ == "neg" for t in conj.children):
                        break
                    else:
                        conj_list.append(conj)

        self.unmatched[idx] = False

    def get_result(self):
        action, *args = self.pattern.transform_rule
        real_args = []

        for item in args:
            token = None

            if isinstance(item, str):
                variable_list = [item]
            else:
                variable_list = list(item)

            for var in variable_list:
                if var in self.matched_data:
                    token = self.matched_data[var]
                    break

            real_args.append(token)

        return self.root_token, action, real_args


class CollectionAnnotator(BaseAnnotator):
    def __init__(self, nlp):
        super().__init__(nlp)

        with pkg_resources.open_text(privacy_policy_analyzer, "verb_patterns.yml") as fin:
            config = yaml.safe_load(fin)

        self.patterns: list[SentencePattern] = []

        for p in config["patterns"]:
            self.patterns.append(SentencePattern(p, config["token_map"]))

    def match_sentence(self, sent: Span):
        def dfs(token, chain_matchers: list[ChainMatcher], inherited_dep, inherited_left_children):
            to_continue: list[ChainMatcher] = []
            to_return: list[ChainMatcher] = []

            for matcher in chain_matchers:
                match matcher.match(token, inherited_dep):
                    case ChainMatcher.CONTINUE:
                        to_continue.append(matcher)
                    case ChainMatcher.FAILED:
                        to_return.append(matcher)
                    case ChainMatcher.SUCCEEDED:
                        pass

            # Start new matchers
            sentence_matcher_list: list[SentenceMatcher] = []
            new_matchers = []

            if ((inherited_dep or token.dep_) in ["ROOT", "ccomp"] and
                all(c.tag_ != "WRB" for c in token.children)):
                # Limit new matchings to ROOT verbs + ccomp
                # ccomp e.g. We inform you that we collect ...
                # Plus: no when/how/whether... adverb
                for p in self.patterns:
                    if sentence_matcher := p.match_root(token):
                        new_matchers.extend(sentence_matcher.chain_matchers)
                        sentence_matcher_list.append(sentence_matcher)

            to_continue.extend(new_matchers)

            # Children = direct children + inherited_left_children
            children = list(token.children)

            if token.dep_ in ["conj", "appos"]:
                children_deps = frozenset(t.dep_ for t in children)

                for left_token in inherited_left_children:
                    assert left_token != token

                    if left_token.dep_ not in children_deps:
                        children.append(left_token)

            children.sort()

            for child in children:
                if child.dep_ in ["conj", "appos"]:
                    inherited_dep = inherited_dep or token.dep_
                    left_children = []
                    for t in itertools.chain(token.lefts, inherited_left_children):
                        if t.i < child.i and t.i < token.i:
                            left_children.append(t)
                else:
                    inherited_dep = None
                    left_children = []

                to_continue = dfs(child, to_continue, inherited_dep, left_children)

            to_return.extend(set(to_continue).difference(new_matchers))

            for matcher in to_return:
                matcher.rollback()

            if sentence_matcher_list:
                neg_flag = any(c.dep_ == "neg" for c in children)

                for sentence_matcher in sentence_matcher_list:
                    if sentence_matcher.fully_matched:
                        matched_results.append((sentence_matcher.pattern, neg_flag, sentence_matcher.get_result()))

            return to_return

        matched_results = []
        dfs(sent.root, [], None, [])
        return matched_results

    def annotate(self, document):
        def like_type(tok_list, target_type):
            """Check if the phrase started by the token is a data type or subsums a data type"""
            # Use BFS here to avoid a loop.
            bfs_queue = list(tok_list)
            seen = set(tok_list)
            i = 0

            while i < len(bfs_queue):
                tok = bfs_queue[i]
                i += 1

                if tok._.ent_type == target_type:
                    return True
                elif tok._.ent_type == "NN":
                    for _, linked_token, relationship in document.get_all_links(tok):
                        if relationship in ["SUBSUM", "COREF"] and linked_token not in seen:
                            bfs_queue.append(linked_token)
                            seen.add(linked_token)

            return False

        def link_pairs(ent_list, data_list, relationship):
            for entity in ent_list:
                for dtype in data_list:
                    self.logger.info("Edge %s: %r -> %r", relationship, entity, dtype)
                    document.link(entity, dtype, relationship)

        def collect_handler(neg_flag, e1, dt, e2):
            if ((e1 and like_type(e1, "ACTOR")) and
                (dt and like_type(dt, "DATA"))):

                if neg_flag:
                    link_pairs(e1, dt, "NOT_COLLECT")
                else:
                    link_pairs(e1, dt, "COLLECT")

        def share_handler(neg_flag, e1, dt, e2):
            if ((dt and like_type(dt, "DATA")) and
                (e2 and like_type(e2, "ACTOR"))):

                if neg_flag:
                    link_pairs(e2, dt, "NOT_COLLECT")
                else:
                    link_pairs(e2, dt, "COLLECT")

        def use_handler(neg_flag, e1, dt):
            if ((e1 and like_type(e1, "ACTOR")) and
                (dt and like_type(dt, "DATA"))):

                if neg_flag:
                    pass
                else:
                    link_pairs(e1, dt, "COLLECT")

        for doc in document.iter_docs():
            for sent in doc.sents:
                if sent[-1].norm_ == "?":
                    # Skip interrogative sentences
                    continue

                if len(results := self.match_sentence(sent)) == 0:
                    continue

                self.logger.info("Found collection statement: %r", sent.text)

                for pattern, neg_flag, (verb, action, args) in results:
                    self.logger.info("Rule %d: verb = %s, action = %s, args = %r",
                                     pattern.rule_id, verb.lemma_, action, args)

                    match action:
                        case "COLLECT":
                            collect_handler(neg_flag, *args)
                        case "SHARE":
                            share_handler(neg_flag, *args)
                        case "USE":
                            use_handler(neg_flag, *args)
