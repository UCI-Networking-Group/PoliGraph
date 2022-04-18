#!/usr/bin/env python3

from itertools import chain

class CoreferenceAnnotator:
    def __init__(self, nlp):
        pass

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
