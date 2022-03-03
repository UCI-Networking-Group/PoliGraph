from spacy.matcher import DependencyMatcher

 
def get_conjuncts(token):
    for child in token.rights:
        if child.dep_ in ["conj", "appos"] and child.pos_ in ["NOUN", "PROPN", "PRON"]:
            yield child


class SubsumptionAnnotator:
    def __init__(self, nlp):
        self.matcher = DependencyMatcher(nlp.vocab)

        # such as
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"ORTH": "as", "DEP": "prep"}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ";",
                "RIGHT_ID": "r00",
                "RIGHT_ATTRS": {"ORTH": "such"}
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
                "RIGHT_ATTRS": {"ORTH": {"IN": ["including", "like"]}}
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

        # including but not limited to
        pattern = [
            {
                "RIGHT_ID": "anchor_including",
                "RIGHT_ATTRS": {"ORTH": "including"}
            },
            {
                "LEFT_ID": "anchor_including",
                "REL_OP": ">",
                "RIGHT_ID": "anchor_limited",
                "RIGHT_ATTRS": {"ORTH": "limited"}
            },
            {
                "LEFT_ID": "anchor_limited",
                "REL_OP": ">",
                "RIGHT_ID": "anchor_to",
                "RIGHT_ATTRS": {"ORTH": "to"}
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

            upper_token = match_info["upper_token"]
            lower_token = match_info["lower_token"]

            if lower_token._.ent is None:
                continue

            print("+" * 40)
            print(upper_token.sent, end="\n\n")
            print(upper_token._.ent, "->", lower_token._.ent._.conjunct_chunks)
            print("+" * 40)

            try:
                doc.user_data["document"].link(upper_token, lower_token, "SUBSUM")
                doc.user_data["document"].group(upper_token, lower_token)
            except RuntimeError:
                pass
