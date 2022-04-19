from spacy.matcher import DependencyMatcher


class PurposeAnnotator:
    def __init__(self, nlp):
        self.matcher = DependencyMatcher(nlp.vocab)

        # to do sth.
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"POS": "VERB"}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "purpose_root",
                "RIGHT_ATTRS": {"POS": "VERB", "DEP": {"IN": ["xcomp", "advcl"]}}
            },
            {
                "LEFT_ID": "purpose_root",
                "REL_OP": ">",
                "RIGHT_ID": "r00",
                "RIGHT_ATTRS": {"LEMMA": "to", "DEP": "aux"}
            },
        ]
        self.matcher.add("PURPOSE_TO_DO", [pattern])

        # in order to do sth.
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"POS": "VERB"}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "purpose_root",
                "RIGHT_ATTRS": {"LEMMA": "in", "DEP": "prep"}
            },
            {
                "LEFT_ID": "purpose_root",
                "REL_OP": ">",
                "RIGHT_ID": "r00",
                "RIGHT_ATTRS": {"LEMMA": "order"}
            },
            {
                "LEFT_ID": "r00",
                "REL_OP": ">",
                "RIGHT_ID": "r01",
                "RIGHT_ATTRS": {"POS": "VERB", "DEP": "acl"}
            },
        ]
        self.matcher.add("PURPOSE_IN_ORDER_TO_DO", [pattern])

        # for xxx purpose
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"POS": "VERB"}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "purpose_root",
                "RIGHT_ATTRS": {"LEMMA": "for", "DEP": "prep"}
            },
            {
                "LEFT_ID": "purpose_root",
                "REL_OP": ">",
                "RIGHT_ID": "r00",
                "RIGHT_ATTRS": {"LEMMA": "purpose"}
            },
        ]
        self.matcher.add("PURPOSE_FOR_PURPOSE", [pattern])

    def annotate(self, doc):
        matches = self.matcher(doc)
        document = doc.user_data["document"]
        collected_dtypes = []

        for e in doc.ents:
            for _, relationship in document.get_links(e[0]):
                if relationship == "COLLECTED_BY":
                    collected_dtypes.append(e[0])

        if len(collected_dtypes) == 0:
            return

        for match_id, matched_tokens in matches:
            _, (match_spec, ) = self.matcher.get(match_id)
            match_info = {s["RIGHT_ID"]: doc[t] for t, s in zip(matched_tokens, match_spec)}

            purpose_root = match_info["purpose_root"]
            right_end = max(t.i for t in purpose_root.subtree) + 1
            purpose_part = doc[purpose_root.i:right_end]

            root_verb = match_info["anchor"]
            for token in root_verb.subtree:
                if token in collected_dtypes and token not in purpose_part:
                    break
            else:
                continue

            print("%" * 40)
            print(purpose_root.sent, end="\n\n")
            print(purpose_part)
            print("%" * 40)
