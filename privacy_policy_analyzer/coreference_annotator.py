#!/usr/bin/env python3

import math
import re


class CoreferenceAnnotator:
    COREF_REGEX = re.compile(r"^(?:this|that|these|those)\b(?:.*\b)?(?:datum|information)\b")

    def __init__(self, nlp):
        import neuralcoref
        self.coref = neuralcoref.NeuralCoref(nlp.vocab)

    def annotate(self, doc):
        doc = self.coref(doc)

        for this_mention, score_dict in doc._.coref_scores.items():
            if self.COREF_REGEX.match(this_mention.lemma_) and this_mention.root.pos_ in ["NOUN", "PROPN"]:
                best_mention, best_score = None, -math.inf

                for other_mention, score in score_dict.items():
                    if other_mention.start >= this_mention.start \
                        or self.COREF_REGEX.match(other_mention.lemma_) \
                        or other_mention.root.pos_ == "PRON":
                        continue

                    if score > 0.0 and score > best_score:
                        best_score = score
                        best_mention = other_mention

                if best_mention is not None:
                    print("#" * 40)
                    print(doc, end="\n\n")
                    print(this_mention, best_mention, sep=" | ")
                    print("Score =", best_score)

                    doc.user_data["document"].link(this_mention.root, best_mention.root, "COREF")
                    doc.user_data["document"].link(best_mention.root, this_mention.root, "COREF")
                    doc.user_data["document"].group(this_mention.root, best_mention.root)
