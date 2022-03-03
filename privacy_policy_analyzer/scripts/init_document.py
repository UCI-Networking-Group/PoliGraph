#!/usr/bin/env python3

import argparse

import spacy
from privacy_policy_analyzer.data_type_recognition import setup_models
from privacy_policy_analyzer.document import PolicyDocument


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", help="working directory")
    parser.add_argument("ner", help="NER model directory")
    args = parser.parse_args()

    spacy.prefer_gpu()
    nlp = setup_models(args.ner)

    document = PolicyDocument(args.workdir, nlp=nlp, use_cache=False)
    document.save()
    document.render_ner()


if __name__ == "__main__":
    main()
