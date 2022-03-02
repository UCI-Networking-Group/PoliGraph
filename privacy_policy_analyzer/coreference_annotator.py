#!/usr/bin/env python3

COREF_BLACKLISTS = frozenset([
    "my", "mine", "our", "ours",
    "you", "your", "yours",
    "he", "him", "his",
    "she", "her", "hers",
    "its", "their", "theirs"
])
COREF_ALLOWED_PRONS = frozenset(["it", "them", "this", "these", "that", "those", "the"])


class CoreferenceAnnotator:
    def __init__(self, nlp):
        import neuralcoref
        self.coref = neuralcoref.NeuralCoref(nlp.vocab)

    def annotate(self, doc):
        doc = self.coref(doc)

        coref_flag = False
        filtered_clusters = []

        for cluster in doc._.coref_clusters:
            for mention in cluster.mentions:
                if mention[0].lower_ in COREF_BLACKLISTS:
                    continue
                elif mention[0].lower_ in COREF_ALLOWED_PRONS:
                    coref_flag = True

                filtered_clusters.append(mention)

            coref_flag &= len({s.lemma_ for s in filtered_clusters}) > 1

            if coref_flag:
                first_token = filtered_clusters[0].root

                for span in filtered_clusters[1:]:
                    doc.user_data["document"].link(first_token, span.root, "COREF")
                    doc.user_data["document"].link(span.root, first_token, "COREF")

                print("#" * 40)
                print(doc, end="\n\n")
                print([t.lower_ for t in filtered_clusters])
