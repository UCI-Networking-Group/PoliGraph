import re

import spacy
from spacy.language import Language
from spacy.tokens import Span

DATATYPE_KEYWORDS = frozenset({"information", "data", "datum", "address", "number", "identifier",
                               "preference", "setting"})
ACTOR_KEYWORDS = frozenset(['advertiser', 'affiliate', 'analytics', 'app', 'application',
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


@Language.component(
    "adjust_entities",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
)
def adjust_entities(doc):
    """Drop invalid named entities and align them to noun chunks"""

    # REF:
    ## https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md
    ## https://www.mathcs.emory.edu/~choi/doc/cu-2012-choi.pdf
    ## nlp.pipe_labels["parser"]
    allowed_deps = {
        "nsubj", "nsubjpass",  # subjects
        "pobj", "dobj", "dative", "oprd", "attr",  # objects
        "nmod", "poss", "appos",  # nominals
        "conj", "ROOT",
        "dep", "meta"  # unclassified
    }

    ents = []
    for e in doc.ents:
        ent_root = e.root
        if ent_root.pos_ in ["NOUN", "PROPN"] and ent_root.dep_ in allowed_deps:
            subtoken_pos = {t.i for t in ent_root.subtree}
            left_edge = ent_root.i

            # keep left tokens as long as they are in the subtree
            while (left_edge - 1) >= e.start and (left_edge - 1) in subtoken_pos:
                left_edge -= 1

            # take in more left tokens if they are in the subtree
            while left_edge - 1 in subtoken_pos:
                prev_token = doc[left_edge - 1]

                # is_space: drop prefixing spaces; pos_ = X: remove prefixing "e.g."
                if prev_token.is_space or prev_token.pos_ == 'X':
                    break

                left_edge -= 1

            # drop prefixing puncts
            while left_edge < e.start and doc[left_edge].norm_ in ".,!?;:)]}>":
                left_edge += 1

            # keep right tokens as long as they are in the subtree
            right_edge = ent_root.i + 1
            while right_edge < e.end and right_edge in subtoken_pos:
                right_edge += 1

            ent_span = Span(doc, left_edge, right_edge, e.label_)
            if re.search('[a-zA-Z0-9]+', ent_span.text):
                while len(ents) > 0 and ents[-1].end > ent_span.start:
                    ents.pop()

                ents.append(ent_span)

    doc.set_ents(ents, default="outside")
    return doc


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
        after="remove_unused_entities",
    )
    nlp.add_pipe(
        "adjust_entities",
        name="adjust_entities",
        after="ner_datatype",
    )

    return nlp
