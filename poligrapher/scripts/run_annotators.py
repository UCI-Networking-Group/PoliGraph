#!/usr/bin/env python3

import argparse
import logging

from poligrapher.annotators import (
    CollectionAnnotator,
    CoreferenceAnnotator,
    ListAnnotator,
    PurposeAnnotator,
    SubsumptionAnnotator,
    SubjectAnnotator,
)
from poligrapher.document import PolicyDocument
from poligrapher.utils import setup_nlp_pipeline


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] <%(name)s> %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp", default="", help="NLP model directory")
    parser.add_argument("--disable", default="", help="Disable annotators for ablation study")
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    args = parser.parse_args()

    nlp = setup_nlp_pipeline(args.nlp)

    disabled_annotators = frozenset(args.disable.split(","))
    annotators = []

    for annotator_class in (SubsumptionAnnotator,
                            CoreferenceAnnotator,
                            CollectionAnnotator,
                            PurposeAnnotator,
                            ListAnnotator,
                            SubjectAnnotator):
        if annotator_class.__name__ not in disabled_annotators:
            annotators.append(annotator_class(nlp))

    for d in args.workdirs:
        logging.info("Processing %s ...", d)

        document = PolicyDocument.load(d, nlp)
        document.token_relationship.clear()

        for annotator in annotators:
            annotator.annotate(document)

        document.save()


if __name__ == "__main__":
    main()
