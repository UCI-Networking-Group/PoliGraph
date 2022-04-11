#!/usr/bin/env python3

import argparse

import spacy
from privacy_policy_analyzer.document import PolicyDocument
from privacy_policy_analyzer.subsumption_annotator import SubsumptionAnnotator
from privacy_policy_analyzer.collection_annotator import CollectionAnnotator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", help="working directory")
    args = parser.parse_args()

    spacy.prefer_gpu()

    document = PolicyDocument(args.workdir)
    nlp = spacy.load("en_core_web_trf")
    # setup_models(args.ner)

    subsumption_annotator = SubsumptionAnnotator(nlp)
    collection_annotator = CollectionAnnotator(nlp)

    for seg in document.segments:
        doc = document.build_doc(seg, nlp, apply_pipe=True, load_ner=True)
        subsumption_annotator.annotate(doc)
        collection_annotator.annotate(doc)

    document.save()


if __name__ == "__main__":
    main()