import re

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Span

DATATYPE_KEYWORDS = frozenset({"information", "data", "datum", "address", "number", "identifier",
                               "preference", "setting"})
ACTOR_KEYWORDS = frozenset(['advertiser', 'affiliate', 'analytic', 'analytics', 'app', 'application',
                            'broker', 'business', 'carrier', 'company', 'corporation',
                            'distributor', 'network', 'operator', 'organization',
                            'partner', 'party', 'platform', 'processor', 'product', 'provider', 'publisher',
                            'service', 'site', 'software', 'subsidiary', 'vendor', 'website'])

@Language.component(
    "remove_unused_entities",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
)
def remove_unused_entities(doc):
    ents = []
    for e in doc.ents:
        # Drop ORDINAL/CARDINAL because otherwise "third" in "third party" would become an NE
        if e.label_ not in ["ORDINAL", "CARDINAL"]:
            ents.append(e)

    doc.set_ents(ents, default="outside")
    return doc

def label_simple_noun_phrases(doc):
    def has_alpha(s):
        return re.search(r"[a-zA-Z]", s) is not None

    iob_map = np.zeros(len(doc) + 1, dtype=np.uint8)
    ent_label_map = dict()

    def dfs(root_token, inside_noun_modifier=False):
        inside_ent = root_token.ent_type_ != ""

        if root_token.dep_ in ["relcl", "advcl"]:
            # Exclude because not simple
            i_am_simple = inside_ent
        elif root_token.dep_ == "prep" and root_token.lemma_ in ["from", "to", "with"]:
            # Exclude some preps because they confuse collection annotators (share with/collect from) 
            i_am_simple = inside_noun_modifier or inside_ent
        elif root_token.dep_ in ["appos", "conj", "cc"]:
            # Allow conj only if inside a noun modifier (e.g. shipping and payment information)
            i_am_simple = inside_noun_modifier or inside_ent
        elif root_token.pos_ in ["X", "PUNCT"]:
            # Keep punct only it's inside a word (e.g. health-related info) or it's a quotation mark
            if (root_token.i > 0 and root_token.i + 1 < len(doc)
                and doc[root_token.i - 1].whitespace_ == "" and has_alpha(doc[root_token.i - 1].text)
                and root_token.whitespace_ == "" and has_alpha(doc[root_token.i + 1].text)):
                i_am_simple = True
            elif root_token.lemma_ in "'\"":
                i_am_simple = True
            else:
                i_am_simple = False
        else:
            i_am_simple = True

        if root_token.dep_ in ["amod", "nmod", "compound"]:
            inside_noun_modifier = True

        # Not root of a noun phrase. Don't set ent
        if root_token.pos_ not in ["NOUN", "PROPN", "PRON"] or not has_alpha(root_token.text):
            for c in root_token.children:
                if not dfs(c, inside_noun_modifier):
                    i_am_simple = False

            return i_am_simple

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
        ent_label_map[left_idx] = root_token.ent_type_ or 'NN'

        return i_am_simple

    for sent in doc.sents:
        dfs(sent.root)

    all_ents = []

    for left in np.where(iob_map == 1)[0]:
        right = left + 1

        while iob_map[right] == 2:
            right += 1

        ent = Span(doc, left, right, label=ent_label_map[left])
        all_ents.append(ent)

    doc.set_ents(all_ents, default="outside")

def setup_models(ner_path):
    nlp = spacy.load("en_core_web_trf")
    our_ner = spacy.load(ner_path)

    # Chain NERs: https://github.com/explosion/projects/tree/v3/tutorials/ner_double
    our_ner.replace_listeners("transformer", "ner", ["model.tok2vec"])
    nlp.add_pipe(
        "remove_unused_entities",
        name="remove_unused_entities",
        after="ner",
    )
    nlp.add_pipe(
        "ner",
        name="ner_datatype",
        source=our_ner,
        after="ner",
    )

    return nlp
