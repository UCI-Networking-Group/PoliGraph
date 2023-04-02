#!/usr/bin/env python3

import argparse
import logging

from privacy_policy_analyzer.annotators import (
    CollectionAnnotator,
    CoreferenceAnnotator,
    ListAnnotator,
    PurposeAnnotator,
    SubsumptionAnnotator,
    SubjectAnnotator,
)
from privacy_policy_analyzer.document import PolicyDocument
from privacy_policy_analyzer.utils import setup_nlp_pipeline


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] <%(name)s> %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp", required=True, help="NLP model directory")
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    args = parser.parse_args()

    nlp = setup_nlp_pipeline(args.nlp)

    annotators = [
        SubsumptionAnnotator(nlp),
        CoreferenceAnnotator(nlp),
        CollectionAnnotator(nlp),
        PurposeAnnotator(nlp),
        ListAnnotator(nlp),
        SubjectAnnotator(nlp),
    ]

    for d in args.workdirs:
        logging.info("Processing %s ...", d)

        document = PolicyDocument.load(d, nlp)
        document.token_relationship.clear()

        for annotator in annotators:
            annotator.annotate(document)

        document.save()


if __name__ == "__main__":
    main()
