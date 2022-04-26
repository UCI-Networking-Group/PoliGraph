from spacy.matcher import DependencyMatcher


class SubsumptionAnnotator:
    def __init__(self, nlp):
        self.matcher = DependencyMatcher(nlp.vocab)

        # such as
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"LEMMA": "as", "DEP": "prep"}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ";",
                "RIGHT_ID": "r00",
                "RIGHT_ATTRS": {"LEMMA": "such"}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": "<",
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
            }
        ]
        self.matcher.add("SUBSUM_SUCH_AS", [pattern])

        # including / like
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"DEP": "prep", "LEMMA": {"IN": ["include", "like"]}}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": "<",
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
            }
        ]
        self.matcher.add("SUBSUM_INCLUDING_LIKE", [pattern])

        # include (verb)
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"DEP": "prep", "LEMMA": "include"}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": "<",
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
            }
        ]
        self.matcher.add("SUBSUM_INCLUDE", [pattern])

        # including but not limited to
        pattern = [
            {
                "RIGHT_ID": "anchor_including",
                "RIGHT_ATTRS": {"LEMMA": "include"}
            },
            {
                "LEFT_ID": "anchor_including",
                "REL_OP": ">",
                "RIGHT_ID": "anchor_limited",
                "RIGHT_ATTRS": {"LEMMA": "limit"}
            },
            {
                "LEFT_ID": "anchor_limited",
                "REL_OP": ">",
                "RIGHT_ID": "anchor_to",
                "RIGHT_ATTRS": {"LEMMA": "to"}
            },
            {
                "LEFT_ID": "anchor_including",
                "REL_OP": "<",
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
            },
            {
                "LEFT_ID": "anchor_to",
                "REL_OP": ">",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
            }
        ]
        self.matcher.add("SUBSUM_INCLUDING_LIMITED_TO", [pattern])

    def annotate(self, doc):
        matches = self.matcher(doc)

        for match_id, matched_tokens in matches:
            _, (match_spec, ) = self.matcher.get(match_id)
            match_info = {s["RIGHT_ID"]: doc[t] for t, s in zip(matched_tokens, match_spec)}

            upper_ent = match_info["upper_token"]._.ent
            lower_ent = match_info["lower_token"]._.ent

            if upper_ent is None or lower_ent is None:
                continue

            lower_conjuncts = lower_ent._.conjunct_chunks

            print("+" * 40)
            print(upper_ent[0].sent, end="\n\n")
            print(upper_ent, "->", lower_conjuncts)
            print("+" * 40)

            for lower_ent in lower_conjuncts:
                doc.user_data["document"].link(upper_ent[0], lower_ent[0], "SUBSUM")
