from ..document import PolicyDocument, SegmentType
from .base import BaseAnnotator


class ListAnnotator(BaseAnnotator):
    """List annotator."""

    def __init__(self, nlp):
        super().__init__(nlp)

    def try_annotate_segment(self, document, root_segment):
        child_listitems = []

        for c in root_segment.children:
            if c.segment_type == SegmentType.LISTITEM and len(c.children) == 1:
                child_listitems.append(c)

        if len(child_listitems) == 0:
            return

        context_doc = document.get_doc_with_context(root_segment)
        context_tokens = {t._.src for t in context_doc}
        link_to_apply = dict()
        child_tokens = []

        for token in context_doc:
            if (token.lemma_, token.dep_) in [("follow", "amod"), ("following", "amod"), ("below", "advmod")]:
                if ent := token.head._.ent:
                    link_to_apply[(ent.root, None)] = "SUBSUM"
                    self.logger.info("SUBSUM edges from: %r", ent.sent.text)
                    break

        for c in child_listitems:
            doc = document.get_doc_without_context(c.children[0])
            root_noun_phrase = next(doc.sents).root._.ent

            if root_noun_phrase is None:
                # TODO: copy purpose links
                continue

            linked_token = root_noun_phrase.root
            child_tokens.append(linked_token)

            # In link -- from context to list item
            for token1, _, relationship in document.get_all_links(linked_token, "in"):
                if token1._.src in context_tokens:
                    link_spec = (token1, None)
                    link_to_apply[link_spec] = relationship
                    self.logger.info("%s edges from: %r", relationship, token1.sent.text)

            # Out links -- from list item to context
            for _, token2, relationship in document.get_all_links(linked_token, "out"):
                if token2._.src in context_tokens:
                    link_spec = (None, token2)
                    link_to_apply[link_spec] = relationship
                    self.logger.info("%s edges to: %r", relationship, token2.sent.text)

        for t in child_tokens:
            for link_spec, relationship in link_to_apply.items():
                endpoints = (link_spec[0] or t, link_spec[1] or t)
                existing_relationship = document.get_link(*endpoints)

                self.logger.info("Edge %s: %r -> %r", relationship, endpoints[0]._.ent, endpoints[1]._.ent)

                if existing_relationship:
                    if existing_relationship != relationship:
                        self.logger.warning("Refuse to overwrite existing relationship %s", existing_relationship)
                else:
                    document.link(*endpoints, relationship)

    def annotate(self, document: PolicyDocument):
        for s in document.segments:
            self.try_annotate_segment(document, s)
