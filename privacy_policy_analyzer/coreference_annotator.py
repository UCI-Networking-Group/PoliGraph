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

    def annotate_one_doc(self, document, doc):
        last_sentence_ents = []

        # Handle pronouns
        for sent in doc.sents:
            current_sentence_ents = []

            for noun_phrase in sent.ents:
                referent = None

                if (noun_phrase[0].lemma_ in {"this", "that", "these", "those"}
                    and noun_phrase[0].head == noun_phrase[-1]):
                    # Resolve this/that/these/those xxx
                    for prev_noun_phrase in chain(reversed(current_sentence_ents), reversed(last_sentence_ents)):
                        if prev_noun_phrase[-1].lemma_ == noun_phrase[-1].lemma_:
                            referent = prev_noun_phrase
                            break

                if referent is None and noun_phrase.lemma_ == "they":
                    # Resolve "they" (referring to an entity)
                    for prev_noun_phrase in chain(reversed(current_sentence_ents), reversed(last_sentence_ents)):
                        if (prev_noun_phrase._.ent_type == "ACTOR" and prev_noun_phrase.root.tag_ in ["NNS", "NNPS"]):
                            referent = prev_noun_phrase
                            break


                if referent is not None:
                    print("=" * 40)
                    print(doc, end="\n\n")
                    print(noun_phrase, "|", referent)

                    document.link(noun_phrase.root, referent.root, "COREF")

                current_sentence_ents.append(noun_phrase)

            last_sentence_ents = current_sentence_ents

        # Handle special patterns
        for match_id, matched_tokens in self.matcher(doc):
            _, (match_spec, ) = self.matcher.get(match_id)
            match_info = {s["RIGHT_ID"]: doc[t] for t, s in zip(matched_tokens, match_spec)}

            coref_noun_phrase = match_info["coref_token"]._.ent
            main_noun_phrase = match_info["main_token"]._.ent

            if coref_noun_phrase is None or main_noun_phrase is None or coref_noun_phrase == main_noun_phrase:
                continue

            print("+" * 40)
            print(coref_noun_phrase.sent, end="\n\n")
            print(coref_noun_phrase, "|", main_noun_phrase)
            print("+" * 40)

            document.link(coref_noun_phrase.root, main_noun_phrase.root, "COREF")

    def annotate(self, document):
        for doc in document.iter_docs():
            self.annotate_one_doc(document, doc)
