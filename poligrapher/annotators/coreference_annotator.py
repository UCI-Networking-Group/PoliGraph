#!/usr/bin/env python3

from itertools import chain

from spacy.matcher import DependencyMatcher

from .base import BaseAnnotator


class CoreferenceAnnotator(BaseAnnotator):
    def __init__(self, nlp):
        super().__init__(nlp)

    def annotate_one_doc(self, document, doc):

        def infer_type(token):
            """Infer noun phrase type through SUBSUM/COREF edges"""
            # Use BFS here to avoid a loop.
            bfs_queue = [token]
            seen = {token}
            i = 0

            while i < len(bfs_queue):
                tok = bfs_queue[i]
                i += 1

                if tok.ent_type_ in ["DATA", "ACTOR"]:
                    return tok.ent_type_
                elif tok.ent_type_ == "NN":
                    for _, linked_token, relationship in document.get_all_links(tok):
                        if relationship in ["SUBSUM", "COREF"] and linked_token not in seen:
                            bfs_queue.append(linked_token)
                            seen.add(linked_token)

            return None

        def link_coref(coref, coref_main, reason):
            self.logger.info("Sentence 1: %r", coref_main.sent.text)
            self.logger.info("Sentence 2: %r", coref.sent.text)
            self.logger.info("Edge COREF (%s): %r -> %r", reason, coref.text, coref_main.text)
            document.link(coref.root, coref_main.root, "COREF")

        last_sentence_ents = []

        # Handle pronouns
        for sent in doc.sents:
            current_sentence_ents = []

            for noun_phrase in sent.ents:
                found = False
                startswith_det = noun_phrase[0].lemma_ in {"this", "that", "these", "those", "such"}

                if startswith_det and noun_phrase[0].head == noun_phrase[-1]:
                    # Resolve this/that/these/those xxx
                    for prev_noun_phrase in chain(current_sentence_ents, last_sentence_ents):
                        if prev_noun_phrase[-1].lemma_ == noun_phrase[-1].lemma_:
                            link_coref(noun_phrase, prev_noun_phrase, "SAME_ROOT")

                            found = True
                            break

                if not found and startswith_det and noun_phrase.root.lemma_ in {"data", "datum", "information"}:
                    # Resolve this information
                    for prev_noun_phrase in chain(current_sentence_ents, last_sentence_ents):
                        if prev_noun_phrase.root.ent_type_  == "DATA":
                            link_coref(noun_phrase, prev_noun_phrase, "THIS_INFO")

                            for conj in prev_noun_phrase.root.conjuncts:
                                if (conj_ent := conj._.ent) is not None:
                                    link_coref(noun_phrase, conj_ent, "THIS_INFO")

                            found = True
                            break

                if not found and noun_phrase.lemma_ in ["it", "this", "they", "these"]:
                    inferred_type = infer_type(noun_phrase.root)

                    if noun_phrase.lemma_ in ["it", "this"]:
                        target_tags = ["NN", "NNP"]
                    else:
                        target_tags = ["NNS", "NNPS"]

                        if inferred_type is None and noun_phrase.lemma_ == "they":
                            inferred_type = "ACTOR"

                    for prev_noun_phrase in chain(current_sentence_ents, last_sentence_ents):
                        if (prev_noun_phrase.root.ent_type_ == inferred_type and
                            prev_noun_phrase.root.tag_ in target_tags and
                            prev_noun_phrase.root.pos_ != "PRON"):
                            link_coref(noun_phrase, prev_noun_phrase, "PRON_SAME_TYPE")

                            found = True
                            break

                current_sentence_ents.append(noun_phrase)

            last_sentence_ents = current_sentence_ents

    def annotate(self, document):
        for doc in document.iter_docs():
            self.annotate_one_doc(document, doc)
