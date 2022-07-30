#!/usr/bin/env python3

import argparse
import logging

import spacy
from privacy_policy_analyzer.document import PolicyDocument


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("--nlp", required=True, help="NLP model directory")
    parser.add_argument("--display", action="store_true", help="Show NER results")
    args = parser.parse_args()

    spacy.prefer_gpu()
    nlp = spacy.load(args.nlp)

    for d in args.workdirs:
        logging.info(f"Processing {d} ...")

        document = PolicyDocument.initialize(d, nlp=nlp)
        document.save()

        if args.display:
            document.render_ner()


if __name__ == "__main__":
    main()
