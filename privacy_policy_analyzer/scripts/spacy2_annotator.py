#!/usr/bin/env python3

import argparse
from packaging.version import parse as parse_ver
import spacy
from spacy.tokens import Span
from privacy_policy_analyzer.coreference_annotator import CoreferenceAnnotator
from privacy_policy_analyzer.document import PolicyDocument


def main():
    if parse_ver(spacy.__version__) >= parse_ver("3.0.0"):
        raise ImportError("spaCy 2.x is required")

    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", help="working directory")
    args = parser.parse_args()

    #spacy.prefer_gpu()

    document = PolicyDocument(args.workdir)
    nlp = spacy.load("en_core_web_lg")
    annotator = CoreferenceAnnotator(nlp)

    for seg in document.segments:
        doc = document.build_doc(seg, nlp, apply_pipe=True, load_ner=True)
        annotator.annotate(doc)

    document.save()


if __name__ == "__main__":
    main()
