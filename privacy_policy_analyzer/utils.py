from spacy.tokens import Doc, Token, Span, SpanGroup
from spacy.language import Language


def is_left_bracket(char):
    return char in ['(', '[', '{']


def is_right_bracket(char):
    return char in [')', ']', '}']


def get_matched_bracket(char):
    return {
        '(': ')', ')': '(',
        '[': ']', ']': '[',
        '{': '}', '}': '{',
    }[char]


@Language.component("get_noun_phrases")
def get_noun_phrases(doc: Doc):
    """Partitions noun phrases in the doc.

    Noun phrases are used to align boundaries of named entities. A noun phrase
    is conceptually the same as a noun chunk in spaCy. We handle some corner
    cases so that phrase boundaries better match our intuition.

    Some corner cases that are not handled by spaCy noun chunks:
    + Prep modifiers on right: type _of device_ / info _about your usage_
    + Unmatched puncts or brackets - will be dropped: _("_We
    + Nested phrases - traverse the dep tree to prioritize higher-level phrases

    For nested phrases, we exclude "complex" right trees (which typically have
    clauses or conjuncts) in a phrase to avoid cluttering many concepts in one:
        e.g. personal info _that includes X info, Y info..._
    So "personal info", "X info", "Y info" are identified as separated phrases.
    """

    # Dependencies that will be excluded in right subtrees
    complex_deps = frozenset([
        "relcl", "advcl", "acl", "acomp", "pcomp", "ccomp", "xcomp",  # clauses
        "appos", "conj", "cc",  # conjuncts
        "punct", "dep",  # NLP glitches
    ])

    def dfs(current_token: Token):
        if current_token.pos_ in ("NOUN", "PROPN", "PRON"):
            left_boundaries = [current_token.i]

            # Include all the continous left children
            for child in sorted(current_token.lefts, reverse=True):
                child_indices = sorted(t.i for t in child.subtree)

                if (left_boundaries[-1] == child_indices[-1] + 1
                    and child_indices[-1] - child_indices[0] + 1 == len(child_indices)):
                    left_boundaries.append(child_indices[0])
                else:
                    break

            # Include "simple" right tokens that do not contain "complex" dependencies
            right_boundaries = [current_token.i + 1]

            for child in sorted(current_token.rights):
                child_indices = sorted(t.i for t in child.subtree)

                if (all(t.dep_ not in complex_deps for t in child.subtree)  # doesn't contain clauses or conj
                    and right_boundaries[-1] == child_indices[0]  # continuous
                    and child_indices[-1] - child_indices[0] + 1 == len(child_indices)):  # subtree is continuous
                    right_boundaries.append(child_indices[-1] + 1)
                else:
                    break

            # Fix mismatched parenthesis or quote (assuming only one exists)
            mismatch_punct_idx = -1
            punct_queue = [(None, None)]

            for i in range(left_boundaries[-1], right_boundaries[-1]):
                orth = doc[i].orth_

                if is_left_bracket(orth):
                    punct_queue.append((orth, i))
                elif is_right_bracket(orth):
                    if punct_queue[-1][0] == get_matched_bracket(orth):
                        punct_queue.pop()
                    else:
                        mismatch_punct_idx = i
                        break
                elif orth in '"':
                    if punct_queue[-1][0] == orth:
                        punct_queue.pop()
                    else:
                        punct_queue.append((orth, i))
            else:
                if len(punct_queue) > 1:
                    mismatch_punct_idx = punct_queue.pop()[1]

            # Drop children containing mismached puncts
            if 0 <= mismatch_punct_idx < current_token.i:
                while left_boundaries[-1] <= mismatch_punct_idx:
                    left_boundaries.pop()
            elif current_token.i < mismatch_punct_idx:
                while right_boundaries[-1] > mismatch_punct_idx:
                    right_boundaries.pop()

            # Save the phrase
            left_idx, right_idx = left_boundaries[-1], right_boundaries[-1]
            yield doc[left_idx:right_idx]

            # Iterate through the rest of children to label phrases
            for child in current_token.children:
                if child.i < left_idx or child.i >= right_idx:
                    yield from dfs(child)
        else:
            for child in current_token.children:
                yield from dfs(child)

    noun_phrases = []

    for sent in doc.sents:
        noun_phrases.extend(dfs(sent.root))

    doc.spans["noun_phrases"] = noun_phrases
    assert not doc.spans["noun_phrases"].has_overlap

    return doc


@Language.component("align_named_entities")
def align_named_entities(doc: Doc):
    """Align NER results to noun phrase boundaries"""

    root_token_ne_labels = {e.root.i: e.label for e in doc.ents}
    aligned_ne = []

    for span in doc.spans["noun_phrases"]:
        if ne_label := root_token_ne_labels.get(span.root.i):
            aligned_ne.append(Span(doc, span.start, span.end, label=ne_label))

    doc.set_ents(aligned_ne, default="outside")

    return doc


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
    return __normalize_ent_label(span.root.ent_type_)
