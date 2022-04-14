#!/usr/bin/env python3

import argparse

import spacy
from privacy_policy_analyzer.collection_annotator import CollectionAnnotator
from privacy_policy_analyzer.coreference_annotator import CoreferenceAnnotator
from privacy_policy_analyzer.document import PolicyDocument
from privacy_policy_analyzer.subsumption_annotator import SubsumptionAnnotator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    args = parser.parse_args()

    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")

    subsumption_annotator = SubsumptionAnnotator(nlp)
    collection_annotator = CollectionAnnotator(nlp)
    coreference_annotator = CoreferenceAnnotator(nlp)

    for d in args.workdirs:
        print(f"Processing {d} ...")

        document = PolicyDocument(d)

        for seg in document.segments:
            doc = document.build_doc(seg, nlp, apply_pipe=True, load_ner=True)
            subsumption_annotator.annotate(doc)
            coreference_annotator.annotate(doc)
            collection_annotator.annotate(doc)

        document.save()


if __name__ == "__main__":
    main()
