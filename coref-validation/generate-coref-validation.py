#!/usr/bin/env python3

import argparse
import json
import logging
import os

from spacy.matcher import DependencyMatcher

from poligrapher.document import PolicyDocument
from poligrapher.utils import setup_nlp_pipeline


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] <%(name)s> %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("--nlp", default="", help="NLP model directory")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    nlp = setup_nlp_pipeline(args.nlp)

    matcher = DependencyMatcher(nlp.vocab)
    pattern = [
        {
            "RIGHT_ID": "anchor",
            "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["DATA", "ACTOR"]}},
        },
        {
            "LEFT_ID": "anchor",
            "REL_OP": ">",
            "RIGHT_ID": "demo_pronoun",
            "RIGHT_ATTRS": {"LEMMA": {"IN": ["this", "that", "these", "those", "such"]}},
        }
    ]
    matcher.add("COREF", [pattern])

    with open(args.output, "w", encoding="utf-8") as fout:
        for d in args.workdirs:
            logging.info("Processing %s ...", d)

            app_id = os.path.basename(os.path.realpath(d))

            document = PolicyDocument.load(d, nlp)
            coref_dict = {}

            for u, v, rel in document.token_relationship.edges(keys=True):
                if rel == "COREF":
                    coref_dict[u] = v

            for seg in document.segments:
                try:
                    doc = document.get_doc_with_context(seg)
                except KeyError:
                    continue

                src_to_token = {t._.src: t for t in doc}

                entities = []
                relations = []

                for _, matched_tokens in matcher(doc):
                    anchor_token = doc[matched_tokens[0]]
                    ent = anchor_token._.ent

                    if anchor_token == ent.root:
                        entities.append((ent.start_char, ent.end_char, anchor_token.ent_type_))

                for token in doc:
                    token_src = token._.src
                    coref_main_src = coref_dict.get(token_src, (-1. -1))
                    coref_main = src_to_token.get(coref_main_src)

                    if coref_main is None or token.ent_type_ == coref_main.ent_type_ == "NN":
                        continue

                    coref_ent = token._.ent
                    coref_main = coref_main._.ent

                    if coref_ent is None or coref_main is None:
                        continue

                    from_to_ids = []

                    for ent in (coref_ent, coref_main):
                        ent_tuple = (ent.start_char, ent.end_char, ent.root.ent_type_)

                        try:
                            ent_idx = entities.index(ent_tuple)
                        except ValueError:
                            ent_idx = len(entities)
                            entities.append(ent_tuple)

                        from_to_ids.append(ent_idx)

                    relations.append(from_to_ids)

                if entities:
                    sample = {
                        "text": doc.text,
                        "entities": [{
                            "id": i,
                            "start_offset": s_off,
                            "end_offset": e_off,
                            "label": label,
                        } for i, (s_off, e_off, label) in enumerate(entities)],
                        "relations": [{
                            "from_id": from_id,
                            "to_id": to_id,
                            "type": "coref"
                        } for from_id, to_id in relations],
                        "app_id": app_id,
                    }

                    print(json.dumps(sample), file=fout)


if __name__ == "__main__":
    main()
