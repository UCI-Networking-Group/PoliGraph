from spacy.matcher import DependencyMatcher

from .base import BaseAnnotator
from .collection_annotator import CollectionAnnotator


class SubjectAnnotator(BaseAnnotator):
    """The very naive (data) subject annotator
    
    It only recognizes children vs others (general)."""

    def __init__(self, nlp):
        super().__init__(nlp)

        self.matcher = DependencyMatcher(self.vocab)

        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"LEMMA": "from", "DEP": "prep"},
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "pobj_children",
                "RIGHT_ATTRS": {"LEMMA": {"REGEX": r"^(child|minor|kid)$"}, "DEP": "pobj"}
            },
        ]

        self.matcher.add("FROM_CHILDREN", [pattern])

        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"LEMMA": "from", "DEP": "prep"},
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">>",
                "RIGHT_ID": "prep_under",
                "RIGHT_ATTRS": {"LEMMA": {"REGEX": r"^(under|of|in)$"}, "DEP": "prep"}
            },
            {
                "LEFT_ID": "prep_under",
                "REL_OP": ">",
                "RIGHT_ID": "pobj_age",
                "RIGHT_ATTRS": {"LEMMA": {"REGEX": r"^(age|year|\d+)$"}, "DEP": "pobj"}
            },
        ]

        self.matcher.add("UNDER_AGE", [pattern])

    def annotate(self, document):
        def poss_is_children(root_token):
            for token in root_token.subtree:
                if token.dep_ == "poss" and token.lemma_ in ("child", "minor", "kid"):
                    return True

            return False

        visited_data_src = set()

        for _, src2, relationship in document.token_relationship.edges(keys=True):
            if relationship in CollectionAnnotator.EDGE_TYPES and src2 not in visited_data_src:
                visited_data_src.add(src2)

                data_token = document.get_token_with_src(src2)
                sentence = data_token.sent

                if len(self.matcher(sentence)) > 0 or poss_is_children(data_token):
                    self.logger.info("Set children as data subject: %r", sentence.text)
                    document.token_relationship.nodes[src2]['subject'] = 'children'
