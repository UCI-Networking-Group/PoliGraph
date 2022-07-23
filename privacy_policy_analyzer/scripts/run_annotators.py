#!/usr/bin/env python3

import argparse

import spacy
from privacy_policy_analyzer.annotators import (
    CollectionAnnotator,
    CoreferenceAnnotator,
    ListAnnotator,
    PurposeAnnotator,
    SubsumptionAnnotator
)
from privacy_policy_analyzer.document import PolicyDocument


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp", required=True, help="NLP model directory")
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    args = parser.parse_args()

    nlp = spacy.load(args.nlp)

    annotators = [
        SubsumptionAnnotator(nlp),
        CoreferenceAnnotator(nlp),
        CollectionAnnotator(nlp),
        PurposeAnnotator(nlp),
        ListAnnotator(nlp),
    ]

    for d in args.workdirs:
        print(f"Processing {d} ...")

        document = PolicyDocument.load(d, nlp)
        document.token_relationship.clear()

        for annotator in annotators:
            annotator.annotate(document)

        document.save()


if __name__ == "__main__":
    main()
