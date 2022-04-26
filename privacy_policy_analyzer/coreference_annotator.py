#!/usr/bin/env python3

from itertools import chain

from spacy.matcher import DependencyMatcher


class CoreferenceAnnotator:
    def __init__(self, nlp):
        self.matcher = DependencyMatcher(nlp.vocab)

        # some/all/any/types/variety/categories of information
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"ORTH": "of", "POS": "ADP"}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": "<",
                "RIGHT_ID": "coref_token",
                "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["some", "all", "any", "type", "variety", "category", "example"]},
                    "POS": {"IN": ["NOUN", "PRON"]}
                }
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "main_token",
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
            }
        ]
        self.matcher.add("COREF_SOME_OF", [pattern])

    def annotate(self, doc):
        last_sentence_ents = []

        # Resolve this/that/these/those xxx
        for sent in doc.sents:
            current_sentence_ents = []

            for ent in sent.ents:
                if ent[0].orth_ in {"this", "that", "these", "those"} and ent[0].head == ent[-1]:
                    for prev_ent in chain(reversed(current_sentence_ents), reversed(last_sentence_ents)):
                        if prev_ent[-1].lemma_ == ent[-1].lemma_:
                            print("=" * 40)
                            print(doc, end="\n\n")
                            print(ent, "|", prev_ent)

                            doc.user_data["document"].link(ent[0], prev_ent[0], "COREF")
                            break

                current_sentence_ents.append(ent)

            last_sentence_ents = current_sentence_ents

        for match_id, matched_tokens in self.matcher(doc):
            _, (match_spec, ) = self.matcher.get(match_id)
            match_info = {s["RIGHT_ID"]: doc[t] for t, s in zip(matched_tokens, match_spec)}

            coref_ent = match_info["coref_token"]._.ent
            main_ent = match_info["main_token"]._.ent

            if coref_ent is None or main_ent is None or coref_ent == main_ent:
                continue

            print("+" * 40)
            print(coref_ent[0].sent, end="\n\n")
            print(coref_ent, "|", main_ent)
            print("+" * 40)

            doc.user_data["document"].link(coref_ent[0], main_ent[0], "COREF")
