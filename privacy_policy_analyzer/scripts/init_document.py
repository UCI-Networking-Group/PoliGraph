#!/usr/bin/env python3

import argparse

import spacy
from privacy_policy_analyzer.document import PolicyDocument
from privacy_policy_analyzer.named_entity_recognition import setup_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("--ner", required=True, help="NER model directory")
    args = parser.parse_args()

    spacy.prefer_gpu()
    nlp = setup_models(args.ner)

    for d in args.workdirs:
        print(f"Processing {d} ...")
        document = PolicyDocument(d, nlp=nlp, use_cache=False)
        document.save()
        # document.render_ner()


if __name__ == "__main__":
    main()
