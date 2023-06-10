import importlib.resources as pkg_resources

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

import poligrapher

TRIVIAL_WORDS = frozenset([
    "some", "all", "any", "variety", "category", "example", "more", "such",
    "another", "other", "following", "follow", "below", "additional", "certain",
    "similar", "limited", "various", "further", "enough", "e.g.", "i.e.", "etc",
    "which", "that", "collectively", "hereinafter", "detailed",
])


def is_left_bracket(char: str) -> bool:
    return char in ('(', '[', '{')


def is_right_bracket(char: str) -> bool:
    return char in (')', ']', '}')


def get_matched_bracket(char: str) -> str:
    return {
        '(': ')', ')': '(',
        '[': ']', ']': '[',
        '{': '}', '}': '{',
    }[char]


@Language.component(
    "align_noun_phrases",
    requires=["doc.ents", "token.ent_iob", "token.ent_type", "token.tag", "token.sent_start"],
)
def align_noun_phrases(doc: Doc) -> Doc:
    """Partitions noun phrases and aligns named entities

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

    # Interested root deps of noun phrases
    phrase_root_deps = frozenset([
        "nsubj", "nsubjpass", "attr", "dobj", "dative", "oprd", "pobj",
        "conj", "appos", "ROOT"
    ])

    # Excluded dependencies of left children
    left_forbidden_deps = frozenset([
        "dep",   # NLP glitches
        "meta",  # e.g. bullet point
    ])

    # Excluded dependencies of right children
    right_forbidden_deps = frozenset([
        "relcl", "advcl", "acomp", "pcomp", "ccomp", "xcomp",  # clauses except "acl"
        "appos", "conj", "cc", "preconj", # conjuncts
        "punct", "dep",                   # NLP glitches
        "meta",                           # e.g. bullet point
    ])

    def like_phrase_root(token):
        return token.pos_ in ("NOUN", "PROPN", "PRON") and token.dep_ in phrase_root_deps

    # Map root tokens of named entities to labels
    ent_root_mapping = {e.root.i: e.label_ for e in doc.ents if like_phrase_root(e.root)}

    def dfs(current_token: Token):
        if like_phrase_root(current_token):
            # Include left children
            left_boundaries = [current_token.i]

            for child in sorted(current_token.lefts, reverse=True):
                child_indices = sorted(t.i for t in child.subtree)

                if (
                    not any(t.dep_ in left_forbidden_deps for t in child.subtree)      # Forbidden deps
                    and left_boundaries[-1] == child_indices[-1] + 1                   # Continuous
                    and child_indices[-1] - child_indices[0] + 1 == len(child_indices) # Continuous subtree
                ):
                    left_boundaries.append(child_indices[0])
                else:
                    break

            # Include right children
            right_boundaries = [current_token.i + 1]

            for child in sorted(current_token.rights):
                child_indices = sorted(t.i for t in child.subtree)

                if (
                    not any(t.dep_ in right_forbidden_deps for t in child.subtree)     # Forbidden deps
                    and ent_root_mapping.keys().isdisjoint(child_indices)              # No overlap w/ other named ents
                    and right_boundaries[-1] == child_indices[0]                       # Continuous
                    and child_indices[-1] - child_indices[0] + 1 == len(child_indices) # Continuous subtree
                ):
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
            yield Span(doc, left_idx, right_idx, ent_root_mapping.get(current_token.i, 0))

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

    doc.set_ents(noun_phrases)

    return doc


@Language.component(
    "label_all_phrases",
    requires=["doc.ents", "token.ent_iob", "token.ent_type", "token.tag", "token.sent_start"],
)
def label_all_phrases(doc: Doc) -> Doc:
    align_noun_phrases(doc)

    doc.set_ents([], default="outside")

    relabelled_ents = []

    for span in doc.spans["noun_phrases"]:
        root_token = span.root

        if root_token.pos_ == "PRON" and root_token.lemma_ in ("I", "we"):
            # Pronoun I / we = First party
            label = "ACTOR"
        else:
            # Use NER label, or "NN" indicating unknown NER type
            label = span.label_ or 'NN'

        relabelled_ents.append(Span(doc, span.start, span.end, label))

    doc.set_ents(relabelled_ents, default="outside")
    del doc.spans["noun_phrases"]

    return doc


def setup_nlp_pipeline(ner_path: str):
    if not ner_path:
        with pkg_resources.path(poligrapher, "extra-data") as extra_data:
            our_ner = spacy.load(extra_data / "named_entity_recognition")
    else:
        our_ner = spacy.load(ner_path)

    nlp = spacy.load("en_core_web_trf", disable=["ner"])

    # Disable spaCy's NER and use our NER
    our_ner.replace_listeners("transformer", "ner", ["model.tok2vec"])
    nlp.add_pipe(
        "ner",
        name="privacy_policy_ner",
        source=our_ner,
    )
    nlp.add_pipe(
        "label_all_phrases",
        name="label_all_phrases",
        after="privacy_policy_ner",
    )

    return nlp


def token_to_ent(token: Token):
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


def token_to_source(token: Token) -> tuple:
    doc = token.doc
    source_mapping = doc.user_data["source"]
    return source_mapping[token.i]
