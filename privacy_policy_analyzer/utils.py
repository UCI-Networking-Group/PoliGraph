try:
    from spacy.tokens import SpanGroup
except ImportError:
    pass


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
    for child in token.children:
        if child.dep_ in ["conj", "appos"]:
            yield child
            yield from get_conjuncts(child)


def token_to_ent(token):
    doc = token.doc
    if token.ent_iob_ not in "BI":
        return None

    left = token.i
    while doc[left].ent_iob_ != "B":
        left -= 1

    right = left + 1
    while right < len(doc) and doc[right].ent_iob_ == "I":
        right += 1

    return doc[left:right]


def token_to_ent_id(token):
    doc = token.doc
    noun_chunk_mapping = doc.user_data["noun_chunk"]

    span = token_to_ent(token)

    if span is None:
        return None
    else:
        return noun_chunk_mapping[span.start]


def chunk_to_conjuncts(chunk):
    conjuncts = set()

    def dfs(token):
        chunk = token_to_ent(token)
        if chunk is not None:
            conjuncts.add(chunk)

        for child in token.children:
            if child.dep_ in ["conj", "appos"]:
                dfs(child)

    doc = chunk.doc
    dfs(chunk.root)
    group = SpanGroup(doc, name="errors", spans=[s for s in conjuncts])

    return group
