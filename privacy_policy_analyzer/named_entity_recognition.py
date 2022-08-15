import re

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Span

DATATYPE_KEYWORDS = frozenset([
    "information", "data", "datum", "address", "number", "location",
    "identifier", "ID", "preference", "setting"
])
ACTOR_KEYWORDS = frozenset([
    'advertiser', 'affiliate', 'analytic', 'analytics', 'app', 'application',
    'broker', 'business', 'carrier', 'company', 'corporation', 'distributor',
    'network', 'operator', 'organization', 'partner', 'party', 'platform',
    'processor', 'product', 'provider', 'publisher', 'service', 'site',
    'software', 'subsidiary', 'vendor', 'website'])
TRIVIAL_WORDS = frozenset([
    "some", "all", "any", "variety", "category", "example", "more",
    "such", "other", "following", "below", "additional", "certain", "similar",
    "limited", "various", "further", "enough", "e.g.", "i.e.", "etc",
    "which", "that", "collectively", "hereinafter", "detailed",
])


@Language.component(
    "label_all_phrases",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
)
def label_all_phrases(doc):
    def has_alpha(s):
        return re.search(r"[a-zA-Z]", s, flags=re.ASCII) is not None

    iob_map = np.zeros(len(doc) + 1, dtype=np.uint8)
    ent_label_map = dict()

    def dfs(root_token, inside_noun_modifier=False):
        inside_ent = root_token.ent_type_ != ""

        match root_token.dep_:
            case "dep" | "meta":
                # NLP glitches
                i_am_simple = False
            case "relcl" | "advcl" | "acl" | "prep" | "appos" | "conj" | "cc":
                # Not simple. Exception: inside a noun modifier (e.g. shipping and payment information)
                i_am_simple = inside_ent or inside_noun_modifier
            case "punct":
                # Keep punct only it's inside a word (e.g. health-related info) or it's a quotation mark
                if (root_token.i > 0 and root_token.i + 1 < len(doc)
                    and doc[root_token.i - 1].whitespace_ == "" and has_alpha(doc[root_token.i - 1].text)
                    and root_token.whitespace_ == "" and has_alpha(doc[root_token.i + 1].text)):
                    i_am_simple = True
                elif root_token.lemma_ in "'\"":
                    i_am_simple = True
                else:
                    i_am_simple = inside_ent or inside_noun_modifier
            case _:
                i_am_simple = True

        if root_token.dep_ in ["amod", "nmod", "nummod", "compound"] and root_token.i < root_token.head.i:
            inside_noun_modifier = True

        # Not root of a noun phrase. Don't set ent
        if root_token.pos_ not in ["NOUN", "PROPN", "PRON"] or not has_alpha(root_token.text):
            for c in root_token.children:
                if not dfs(c, inside_noun_modifier):
                    i_am_simple = False

            return i_am_simple

        # Noun phrase, try to include simple children
        left_idx = root_token.i
        for child in sorted(root_token.lefts, reverse=True):
            if not dfs(child, inside_noun_modifier):
                i_am_simple = False
            else:
                child_indices = sorted(t.i for t in child.subtree)
                if (child_indices[-1] - child_indices[0] == len(child_indices) - 1
                    and left_idx - 1 == child_indices[-1]):
                    left_idx = child_indices[0]

        right_idx = root_token.i + 1
        for child in sorted(root_token.rights):
            if not dfs(child, inside_noun_modifier):
                i_am_simple = False
            else:
                child_indices = sorted(t.i for t in child.subtree)
                if (child_indices[-1] - child_indices[0] == len(child_indices) - 1
                    and right_idx == child_indices[0]):
                    right_idx = child_indices[-1] + 1

        iob_map[left_idx] = 1
        iob_map[left_idx + 1:right_idx] = 2

        # Rule-based NER completion
        if root_token.lemma_.lower() in DATATYPE_KEYWORDS:
            label = "DATA"
        elif root_token.lemma_.lower() in ACTOR_KEYWORDS:
            label = "ACTOR"
        elif root_token.pos_ == "PRON" and root_token.lemma_ in {"I", "we", "you", "he", "she"}:
            label = "ACTOR"
        else:
            label = root_token.ent_type_ or 'NN'

        ent_label_map[left_idx] = label

        return i_am_simple

    for sent in doc.sents:
        dfs(sent.root)

    all_ents = []

    for left in np.where(iob_map == 1)[0]:
        ent_label = ent_label_map[left]

        # Skip useless ent types. DATE/LOC/LAW are kept in case we need in the future.
        if ent_label in ["ORDINAL", "CARDINAL", "PERCENT"]:
            continue

        right = left + 1

        while iob_map[right] == 2:
            right += 1

        ent = Span(doc, left, right, label=ent_label_map[left])
        all_ents.append(ent)

    doc.set_ents(all_ents, default="outside")
    return doc


def setup_models(ner_path):
    nlp = spacy.load("en_core_web_trf")
    our_ner = spacy.load(ner_path)

    # Chain NERs: https://github.com/explosion/projects/tree/v3/tutorials/ner_double
    our_ner.replace_listeners("transformer", "ner", ["model.tok2vec"])
    nlp.add_pipe(
        "ner",
        name="ner2",
        source=our_ner,
        after="ner",
    )

    nlp.add_pipe(
        "label_all_phrases",
        name="label_all_phrases",
        after="ner2",
    )

    return nlp
