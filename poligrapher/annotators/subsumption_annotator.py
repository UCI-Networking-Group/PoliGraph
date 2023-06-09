from collections import deque

import networkx as nx
from spacy.matcher import DependencyMatcher

from .base import BaseAnnotator


def ent_type_is_compatible(ent1, ent2):
    t1 = ent1.root.ent_type_
    t2 = ent2.root.ent_type_

    if t1 in ("ACTOR", "DATA"):
        return t2 in (t1, "NN")
    else:
        return t2 in ("ACTOR", "DATA")


class SubsumptionAnnotator(BaseAnnotator):
    def __init__(self, nlp):
        super().__init__(nlp)
        self.matcher = DependencyMatcher(nlp.vocab)

        pos_is_noun = {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}

        # such as
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"LEMMA": "as", "DEP": "prep"},
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ";",
                "RIGHT_ID": "r00",
                "RIGHT_ATTRS": {"LEMMA": "such"},
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": "<",
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": pos_is_noun,
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {"DEP": "pobj", **pos_is_noun},
            }
        ]
        self.matcher.add("SUBSUM_SUCH_AS", [pattern])

        # including / like
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"DEP": "prep", "LEMMA": {"IN": ["include", "like"]}},
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": "<",
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": pos_is_noun,
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {"DEP": "pobj", **pos_is_noun},
            }
        ]
        self.matcher.add("SUBSUM_INCLUDING_LIKE", [pattern])

        # include (verb)
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": "include"},
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": {"DEP": "nsubj",
                                # e.g. Prevent false positives like "Our website include social media features..."
                                "ENT_TYPE": {"NOT_IN": ["ACTOR"]},
                                **pos_is_noun},
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {"DEP": "dobj", **pos_is_noun},
            }
        ]
        self.matcher.add("SUBSUM_INCLUDE", [pattern])

        # ... which includes (clause)
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"POS": "VERB", "DEP": "relcl", "LEMMA": "include"},
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": "<",
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": pos_is_noun,
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {"DEP": "dobj", **pos_is_noun},
            }
        ]
        self.matcher.add("SUBSUM_WHICH_INCLUDE", [pattern])

        # including but not limited to
        pattern = [
            {
                "RIGHT_ID": "anchor_including",
                "RIGHT_ATTRS": {"LEMMA": "include"},
            },
            {
                "LEFT_ID": "anchor_including",
                "REL_OP": ">",
                "RIGHT_ID": "anchor_limited",
                "RIGHT_ATTRS": {"LEMMA": "limit"},
            },
            {
                "LEFT_ID": "anchor_limited",
                "REL_OP": ">",
                "RIGHT_ID": "anchor_not",
                "RIGHT_ATTRS": {"DEP": "neg"},
            },
            {
                "LEFT_ID": "anchor_limited",
                "REL_OP": ">",
                "RIGHT_ID": "anchor_to",
                "RIGHT_ATTRS": {"LEMMA": "to"},
            },
            {
                "LEFT_ID": "anchor_including",
                "REL_OP": "<",
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": pos_is_noun,
            },
            {
                "LEFT_ID": "anchor_to",
                "REL_OP": ">",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {"DEP": "pobj", **pos_is_noun},
            }
        ]
        self.matcher.add("SUBSUM_INCLUDING_LIMITED_TO", [pattern])

        # includes but not limits to
        pattern = [
            {
                "RIGHT_ID": "anchor_include",
                "RIGHT_ATTRS": {"LEMMA": "include"},
            },
            {
                "LEFT_ID": "anchor_include",
                "REL_OP": ">",
                "RIGHT_ID": "anchor_limit",
                "RIGHT_ATTRS": {"DEP": "conj", "LEMMA": "limit"},
            },
            {
                "LEFT_ID": "anchor_limit",
                "REL_OP": ">",
                "RIGHT_ID": "anchor_not",
                "RIGHT_ATTRS": {"DEP": "neg"},
            },
            {
                "LEFT_ID": "anchor_limit",
                "REL_OP": ">",
                "RIGHT_ID": "anchor_to",
                "RIGHT_ATTRS": {"LEMMA": "to"},
            },
            {
                "LEFT_ID": "anchor_include",
                "REL_OP": ">",
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": {"DEP": "nsubj", **pos_is_noun},
            },
            {
                "LEFT_ID": "anchor_to",
                "REL_OP": ">",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {"DEP": "pobj", **pos_is_noun},
            }
        ]
        self.matcher.add("SUBSUM_INCLUDES_LIMITS_TO", [pattern])

        # for example
        pattern = [
            {
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": pos_is_noun,
            },
            {
                "LEFT_ID": "upper_token",
                "REL_OP": ">",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {"DEP": "appos", **pos_is_noun},
            },
            {
                "LEFT_ID": "lower_token",
                "REL_OP": ">",
                "RIGHT_ID": "prep_for",
                "RIGHT_ATTRS": {"LEMMA": "for"},
            },
            {
                "LEFT_ID": "prep_for",
                "REL_OP": ">",
                "RIGHT_ID": "example",
                "RIGHT_ATTRS": {"LEMMA": "example"},
            },
        ]
        self.matcher.add("SUBSUM_FOR_EXAMPLE", [pattern])

        # especially
        pattern = [
            {
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": pos_is_noun,
            },
            {
                "LEFT_ID": "upper_token",
                "REL_OP": ">",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {"DEP": "appos", **pos_is_noun},
            },
            {
                "LEFT_ID": "lower_token",
                "REL_OP": ">",
                "RIGHT_ID": "token_especially",
                "RIGHT_ATTRS": {"LEMMA": {"IN": ["especially", "particularly", "namely"]}, "POS": 'ADV'},
            },
        ]
        self.matcher.add("SUBSUM_ESPECIALLY", [pattern])

        # e.g. / i.e.
        pattern = [
            {
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": pos_is_noun,
            },
            {
                "LEFT_ID": "upper_token",
                "REL_OP": ">",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {"DEP": "appos", **pos_is_noun},
            },
            {
                "LEFT_ID": "lower_token",
                "REL_OP": ">",
                "RIGHT_ID": "advmod_eg",
                "RIGHT_ATTRS": {"LEMMA": {"IN": ["e.g.", "eg", "i.e.", "ie"]}},
            },
        ]
        self.matcher.add("SUBSUM_EG", [pattern])

        # such XXX as YYY
        pattern = [
            {
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": pos_is_noun,
            },
            {
                "LEFT_ID": "upper_token",
                "REL_OP": ">",
                "RIGHT_ID": "amod_such",
                "RIGHT_ATTRS": {"DEP": "amod", "LEMMA": "such"},
            },
            {
                "LEFT_ID": "upper_token",
                "REL_OP": ">",
                "RIGHT_ID": "prep_as",
                "RIGHT_ATTRS": {"DEP": "prep", "LEMMA": "as"},
            },
            {
                "LEFT_ID": "prep_as",
                "REL_OP": ">",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {"DEP": "pobj", **pos_is_noun},
            },
        ]
        self.matcher.add("SUBSUM_SUCH_N_AS", [pattern])

        # ..., ... (collectively "XXX")
        pattern = [
            {
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": {"DEP": "appos", **pos_is_noun},
            },
            {
                "LEFT_ID": "upper_token",
                "REL_OP": ">",
                "RIGHT_ID": "advmod_collectively",
                "RIGHT_ATTRS": {"DEP": "advmod", "LEMMA": {"IN": ["collectively", "hereinafter"]}}
            },
            {
                "LEFT_ID": "upper_token",
                "REL_OP": "<",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": pos_is_noun,
            },
        ]
        self.matcher.add("SUBSUM_COLLECTIVELY", [pattern])

        # some/all/any/types/variety/categories of information
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"ORTH": "of", "POS": "ADP"}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": "<",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["some", "all", "any", "type", "kind", "variety", "category", "example"]},
                    "POS": {"IN": ["NOUN", "PRON"]}
                }
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": pos_is_noun,
            }
        ]
        self.matcher.add("COREF_SOME_OF", [pattern])

    def annotate_subsum_patterns(self, document, doc):

        def has_negation(matched_tokens):
            for token_id in matched_tokens:
                for child in doc[token_id].children:
                    if child.dep_ == "neg" and child.i not in matched_tokens:
                        return True

            return False

        def search_child_ent(root_token):
            bfs_queue = deque()
            bfs_queue.append(root_token)
            visited_tokens = set()

            while len(bfs_queue):
                token = bfs_queue.popleft()

                if has_negation([token.i]):
                    continue

                if token.i not in visited_tokens:
                    if ent := token._.ent:
                        visited_tokens.update(t.i for t in ent)
                        yield ent

                for child in token.children:
                    if (child.dep_ in ("pobj", "dobj", "conj", "appos") or
                        child.dep_ == "prep" and child.lemma_ in ("about", "regard")):
                        bfs_queue.extend(token.children)

        matches = self.matcher(doc)

        for match_id, matched_tokens in matches:
            if has_negation(matched_tokens):
                # Skip negations
                continue

            rule_name = self.vocab.strings[match_id]
            _, (match_spec, ) = self.matcher.get(match_id)

            match_info = {s["RIGHT_ID"]: doc[t] for t, s in zip(matched_tokens, match_spec)}

            # Move lower token to the first conj
            lower_token = match_info["lower_token"]
            lower_token = min((lower_token, *lower_token.conjuncts))

            # Make sure the upper token is always the nearest conj to the lower token
            # Fix: "We and our partners like Google" -- sometimes "We" is linked to "Google"
            upper_token = match_info["upper_token"]
            upper_conjuncts = filter(lambda t: (t.i > lower_token.i) == (upper_token.i > lower_token.i),
                                     upper_token.conjuncts)
            upper_token = min((upper_token, *upper_conjuncts), key=lambda t: abs(t.i - lower_token.i))

            if upper_token.pos_ == "PRON":
                # Fix: "we/they, such as..." sounds most likely a false positive
                continue

            if upper_token.dep_ == "attr" and upper_token.head.lemma_ == "be":
                # e.g.: "Personal data" is information ..., including ...
                # Move upper_token to the word before "be"
                for child in upper_token.head.children:
                    if child.dep_ == "nsubj":
                        upper_token = child
                        break

            upper_ent = upper_token._.ent
            sentence = upper_token.sent

            if upper_ent is None:
                continue

            self.logger.info("Rule %s matches %r", rule_name, sentence.text)
            self.logger.info("Matched upper token: %r", upper_ent.text)

            for child_ent in search_child_ent(lower_token):
                if ent_type_is_compatible(upper_ent, child_ent):
                    document.link(upper_ent.root, child_ent.root, "SUBSUM")
                    self.logger.info("Edge SUBSUM: %r -> %r", upper_ent.text, child_ent.text)

    def annotate_first_party_appos(self, document, doc):
        """Find synonyms to "we" -- the first party"""

        for sent in doc.sents:
            appos_graph = nx.Graph()
            first_party_references = []

            for ent in sent.ents:
                ent_root = ent.root

                if ent_root.dep_ == "appos":
                    appos_graph.add_edge(ent_root.i, ent_root.head.i)

                if ent.lemma_.lower() == "we":
                    first_party_references.append(ent_root.i)
                    appos_graph.add_node(ent_root.i)

            if first_party_references:
                upper_ent = doc[first_party_references[0]]._.ent
                length_to_us = nx.multi_source_dijkstra_path_length(appos_graph, first_party_references)
            else:
                continue

            if max(length_to_us.values()) > 0 and upper_ent:
                self.logger.info("Possible 1st-party appos: %r", sent.text)
            else:
                continue

            for idx in length_to_us:
                lower_ent = doc[idx]._.ent

                if lower_ent and idx not in first_party_references and ent_type_is_compatible(upper_ent, lower_ent):
                    self.logger.info("Matched tokens: upper = %r, lower = %r", upper_ent.text, lower_ent.text)
                    document.link(upper_ent.root, lower_ent.root, "SUBSUM")

    def annotate(self, document):
        for doc in document.iter_docs():
            self.annotate_first_party_appos(document, doc)
            self.annotate_subsum_patterns(document, doc)
