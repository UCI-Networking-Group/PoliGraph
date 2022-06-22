from spacy.tokens import SpanGroup


def get_conjuncts(token):
    for child in token.children:
        if child.dep_ == "conj":
            yield child
            yield from get_conjuncts(child)
        elif child.dep_ == "appos":
            # appos often appears to be wrong in long doc
            # take it only when parent and child are in the same segment and parent.i < child.i
            src1 = token._.src or (-1, 0)
            src2 = child._.src or (-2, 0)

            if src1[0] == src2[0] and src1[1] < src2[1]:
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


def token_to_source(token):
    doc = token.doc
    source_mapping = doc.user_data["source"]
    return source_mapping[token.i]


def chunk_to_conjuncts(chunk):
    conjuncts = []

    def dfs(token):
        chunk = token_to_ent(token)
        if chunk is not None:
            conjuncts.append(chunk)

        for child in token.children:
            if child.dep_ in ["conj", "appos"]:
                dfs(child)

    doc = chunk.doc
    dfs(chunk.root)
    group = SpanGroup(doc, name="conjuncts", spans=conjuncts)

    return group


def __normalize_ent_label(label):
    if label == "DATA":
        return "DATA"
    elif label in {"ACTOR", "EVENT", "FAC", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"}:
        return "ACTOR"
    elif label == "NN":
        return "NN"
    else:
        return "OTHER"


def token_ent_type(token):
    return __normalize_ent_label(token.ent_type_)


def span_ent_type(span):
    return __normalize_ent_label(span.label_)
