#!/usr/bin/env python3

import argparse

import spacy
from privacy_policy_analyzer.collection_annotator import CollectionAnnotator
from privacy_policy_analyzer.coreference_annotator import CoreferenceAnnotator
from privacy_policy_analyzer.document import PolicyDocument
from privacy_policy_analyzer.purpose_annotator import PurposeAnnotator
from privacy_policy_analyzer.subsumption_annotator import SubsumptionAnnotator
from privacy_policy_analyzer.list_annotator import ListAnnotator


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
