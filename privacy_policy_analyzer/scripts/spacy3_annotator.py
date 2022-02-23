#!/usr/bin/env python3

import argparse

import spacy
from privacy_policy_analyzer.document import PolicyDocument
from privacy_policy_analyzer.subsumption_annotator import SubsumAnnotator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", help="working directory")
    args = parser.parse_args()

    spacy.prefer_gpu()

    document = PolicyDocument(args.workdir)
    nlp = spacy.load("en_core_web_trf")
    # setup_models(args.ner)

    annotator = SubsumAnnotator(nlp)

    for seg in document.segments:
        doc = document.build_doc(seg, nlp, apply_pipe=True, load_ner=True)
        annotator.annotate(doc)


if __name__ == "__main__":
    main()
