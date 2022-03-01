from spacy.matcher import DependencyMatcher


def expand_token(token):
    doc = token.doc
    if token.ent_iob_ != 'O':
        span_start = span_end = token.i

        while doc[span_start].ent_iob_ != 'B':
            span_start -= 1

        while span_end < len(doc) and doc[span_end].ent_iob_ != 'O':
            span_end += 1

        return doc[span_start:span_end]
    else:
        subtoken_pos = {t.i for t in token.subtree}
        left_edge = token.i

        while left_edge - 1 in subtoken_pos:
            prev_token = doc[left_edge - 1]

            if prev_token.is_space or prev_token.pos_ == 'X' or prev_token.ent_iob_ != 'O':
                break

            left_edge -= 1

        return doc[left_edge:token.i + 1]

 
def get_conjuncts(token):
    for child in token.rights:
        if child.dep_ in ["conj", "appos"] and child.pos_ in ["NOUN", "PROPN", "PRON"]:
            yield child


class SubsumAnnotator:
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

        # includes / including / like
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"ORTH": {"IN": ["include", "including", "like"]}}
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

            lower_tokens = [match_info["lower_token"]]
            lower_tokens.extend(get_conjuncts(lower_tokens[0]))

            if any(t.ent_iob_ != 'O' for t in lower_tokens):
                upper_span = expand_token(match_info["upper_token"])

                for lower_token in lower_tokens:
                    lower_span = expand_token(lower_token)
                    doc.user_data["document"].link(upper_span.root, lower_span.root, "SUBSUM")
                    print(upper_span, lower_span, sep=" | ")
