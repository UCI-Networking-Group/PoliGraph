from privacy_policy_analyzer.document import PolicyDocument, SegmentType


class ListAnnotator:
    def __init__(self, nlp):
        pass

    def try_annotate_segment(self, document, root_segment):
        if root_segment.segment_type == SegmentType.LISTITEM:
            return

        context_doc = document.get_doc_with_context(root_segment)
        token_map = {t._.src: t for t in context_doc}
        link_to_apply = dict()
        child_tokens = []

        for c in root_segment.children:
            if not(c.segment_type == SegmentType.LISTITEM and len(c.children) == 1):
                # Might work on other segments. But let's not be too progressive
                continue

            text_segment = c.children[0]

            doc = document.get_doc_without_context(text_segment)
            root_noun_phrase = next(doc.sents).root._.ent

            if root_noun_phrase is None:
                # TODO: copy purpose links
                continue

            linked_token = root_noun_phrase.root
            child_tokens.append(linked_token)

            # In link -- from context to list item
            for token1, _, relationship in document.get_all_links(linked_token, "in"):
                if token1._.src in token_map:
                    link_spec = (token1, None)
                    link_to_apply[link_spec] = relationship

            # Out links -- from list item to context
            for _, token2, relationship in document.get_all_links(linked_token, "out"):
                if token2._.src in token_map:
                    link_spec = (None, token2)
                    link_to_apply[link_spec] = relationship

        for t in child_tokens:
            for link_spec, relationship in link_to_apply.items():
                full_link_spec = (link_spec[0] or t, link_spec[1] or t)
                previous_relationship = document.get_link(*full_link_spec)

                if previous_relationship:
                    if previous_relationship != relationship:
                        print("WARNING: Existing relationship is different")
                else:
                    print("_" * 40)
                    print(full_link_spec[0].sent, end="\n\n")
                    print(full_link_spec[1].sent, end="\n\n")
                    print(relationship, full_link_spec[0]._.ent, full_link_spec[1]._.ent, sep=", ")
                    document.link(*full_link_spec, relationship)
                    print("_" * 40)

    def annotate(self, document: PolicyDocument):
        for s in document.segments:
            self.try_annotate_segment(document, s)
