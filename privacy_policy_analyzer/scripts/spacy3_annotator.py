#!/usr/bin/env python3

import argparse

import spacy
from privacy_policy_analyzer.collection_annotator import CollectionAnnotator
from privacy_policy_analyzer.coreference_annotator import CoreferenceAnnotator
from privacy_policy_analyzer.document import PolicyDocument, SegmentType
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
            if seg.segment_type == SegmentType.LISTITEM:
                continue

            for annotator in subsumption_annotator, coreference_annotator, collection_annotator:
                for i in range(len(seg.context)):
                    doc = document.build_doc(seg.context[:i+1], nlp, apply_pipe=True, load_ner=True)
                    annotator.annotate(doc)

        document.save()


if __name__ == "__main__":
    main()
