from spacy.matcher import DependencyMatcher

from .base import BaseAnnotator
from .collection_annotator import CollectionAnnotator


class SubjectAnnotator(BaseAnnotator):
    """The very naive (data) subject annotator
    
    It only recognizes children vs others (general)."""

    def __init__(self, nlp):
        super().__init__(nlp)

        self.matcher = DependencyMatcher(self.vocab)

        # from/about/of children / minors / kids
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"LEMMA": {"IN": ["from", "about", "of"]}, "DEP": "prep"},
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "pobj_children",
                "RIGHT_ATTRS": {"LEMMA": {"REGEX": r"^(child|minor|kid)$"}, "DEP": "pobj"}
            },
        ]

        self.matcher.add("FROM_CHILDREN", [pattern])

        # NOUN under/between ... age/years
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}},
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">>",
                "RIGHT_ID": "prep_under",
                "RIGHT_ATTRS": {"LEMMA": {"REGEX": r"^(under|between)$"}, "DEP": "prep"}
            },
            {
                "LEFT_ID": "prep_under",
                "REL_OP": ">",
                "RIGHT_ID": "pobj_age",
                "RIGHT_ATTRS": {"LEMMA": {"REGEX": r"^(age|year|old|\d+)$"}, "DEP": {"IN": ["pobj", "amod"]}}
            },
        ]

        self.matcher.add("UNDER_AGE", [pattern])

        # children's information
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"ENT_TYPE": "DATA"},
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "poss",
                "RIGHT_ATTRS": {"LEMMA": {"REGEX": r"^(child|minor|kid)$"}, "DEP": "poss"}
            },
        ]

        self.matcher.add("CHILDREN_POSS", [pattern])

    def annotate(self, document):
        visited_data_src = set()

        for _, src2, relationship in document.token_relationship.edges(keys=True):
            if relationship in CollectionAnnotator.EDGE_TYPES and src2 not in visited_data_src:
                visited_data_src.add(src2)

                data_token = document.get_token_with_src(src2)
                sentence = data_token.sent

                if len(self.matcher(sentence)) > 0:
                    self.logger.info("Set children as data subject: %r", sentence.text)
                    document.token_relationship.nodes[src2]['subject'] = 'children'
