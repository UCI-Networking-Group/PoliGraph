from spacy.matcher import DependencyMatcher

from ..utils import TRIVIAL_WORDS
from .base import BaseAnnotator
from .collection_annotator import CollectionAnnotator


class PurposeValidator:
    ADDITIONAL_STOP_WORDS = frozenset({*TRIVIAL_WORDS, "purpose", "reason", "use"})

    def __init__(self, vocab):
        self.deny_matcher = DependencyMatcher(vocab)

        patterns = []
        # For ... period
        patterns.append([
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"LEMMA": "for"}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "r00",
                "RIGHT_ATTRS": {"LEMMA": {"IN": [
                    "day", "week", "month", "year", "period", "time", "instance", "duration"]
                }}
            },
        ])
        patterns.append([
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"LEMMA": {"IN": ["purpose", "reason", "use"]}}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "r00",
                "RIGHT_ATTRS": {"LEMMA": {"IN": [
                    "this", "that", "these", "those", "any", "such", "various", "certain",
                ]}}
            },
        ])
        patterns.append([
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"LEMMA": {"IN": ["purpose", "reason", "use"]}}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "r00",
                "RIGHT_ATTRS": {"DEP": "nummod"}
            },
        ])
        self.deny_matcher.add("DENY", patterns)

    def __call__(self, span):
        if self.deny_matcher(span):
            return False

        if span.root.lemma_ == "collect":
            return False

        for token in span:
            if (not token.is_stop
                and token.ent_type_ not in ('DATA', 'ACTOR')
                and token.lemma_ not in self.ADDITIONAL_STOP_WORDS):
                return True


class PurposeAnnotator(BaseAnnotator):
    def __init__(self, nlp):
        super().__init__(nlp)
        self.matcher = DependencyMatcher(nlp.vocab)
        self.validator = PurposeValidator(nlp.vocab)

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
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}
            },
        ]
        self.matcher.add("PURPOSE_FOR_PURPOSE", [pattern])

    def annotate_one_doc(self, document, doc):
        collected_dtypes = []

        for noun_phrase in doc.ents:
            for _, _, relationship in document.get_all_links(noun_phrase.root, "in"):
                if relationship in CollectionAnnotator.EDGE_TYPES:
                    collected_dtypes.append(noun_phrase.root)
                    break

        if len(collected_dtypes) == 0:
            return

        matches = self.matcher(doc)

        for match_id, matched_tokens in matches:
            rule_name = self.vocab.strings[match_id]
            _, (match_spec, ) = self.matcher.get(match_id)
            match_info = {s["RIGHT_ID"]: doc[t] for t, s in zip(matched_tokens, match_spec)}

            purpose_root = match_info["purpose_root"]
            sentence = purpose_root.sent

            if sentence[-1].lemma_ == ":":
                continue

            # right_end is the end of purpose_root's subtree
            right_end = max(t.i for t in purpose_root.subtree) + 1

            # left_end is a bit tricky. If the purpose phrase span two segments (listitem), then build_graph will have
            # difficulty finding the full doc. So we make sure the phrase never spans segments by moving the left_end
            # to the start of the last segment.
            # A better solution would require improving the link mechanism
            left_end = purpose_root.i
            last_segment_id = -1

            for i in range(purpose_root.i, right_end):
                src = doc[i]._.src

                if src and src[0] > last_segment_id:
                    last_segment_id = doc[i]._.src[0]
                    left_end = i

            purpose_part = doc[left_end:right_end]

            if not self.validator(purpose_part):
                continue

            self.logger.info("Rule %s matches %r", rule_name, sentence.text)
            self.logger.info("Purpose phrase: %r", purpose_part.text)

            # Select data types under the verb's subtree
            associate_dtypes = []
            for token in match_info["anchor"].subtree:
                if token in collected_dtypes and token not in purpose_part:
                    associate_dtypes.append(token)

            if len(associate_dtypes) > 0:
                self.logger.info("Linkable data types: %r", associate_dtypes)

                for dtype in associate_dtypes:
                    document.link(dtype, purpose_part.root, "PURPOSE")

    def annotate(self, document):
        for doc in document.iter_docs():
            self.annotate_one_doc(document, doc)
