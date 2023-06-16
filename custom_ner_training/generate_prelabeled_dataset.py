#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random

import spacy

from poligrapher.document import PolicyDocument, SegmentType


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("-b", "--base", default="", help="Existing verified data")
    parser.add_argument("-o", "--out", required=True, help="Output jsonl file path")
    parser.add_argument("--nlp", required=True, help="NLP model directory")
    args = parser.parse_args()

    spacy.prefer_gpu()
    nlp = spacy.load(args.nlp)

    verified_data_map = {}

    if args.base:
        with open(args.base, "r", encoding="utf-8") as fin:
            for line in fin:
                data = json.loads(line)
                verified_data_map[data["text"]] = data

    verified_samples = []
    todo_samples = []
    seen_text = set()

    for d in args.workdirs:
        logging.info("Processing %s ...", d)

        document = PolicyDocument.load(d, nlp=nlp)

        for seg in document.segments:
            if seg.segment_type == SegmentType.LISTITEM:
                continue

            doc = document.get_doc_with_context(seg)

            # Skip incomplete sentences
            if (len(doc) == 0 or
                doc[-1].dep_ != "punct" or
                doc[-1].lemma_ == ":" or
                all(sent.root.pos_ != "VERB" for sent in doc.sents)):
                continue

            text = doc.text

            if text not in seen_text:
                seen_text.add(text)
            else:
                continue

            doc.set_ents([e for e in doc.ents if e.root.pos_ != "PRON"])

            sample = {
                "text": text,
                "policy_id": os.path.basename(d),
                "segment_id": seg.segment_id,
                "cats": [],
            }

            if info := verified_data_map.get(doc.text):
                sample["entities"] = info["entities"]
                sample["verified"] = True
                sample["cats"] = info["cats"]

                verified_samples.append(sample)
            else:
                ents = []

                for e in doc.ents:
                    if e.label_ in {"EVENT", "FAC", "ORG", "PERSON", "PRODUCT"}:
                        ner_label = "ACTOR"
                    elif e.label_ in {"ACTOR", "DATA"}:
                        ner_label = e.label_
                    else:
                        continue

                    ents.append([e.start_char, e.end_char, ner_label])

                sample["entities"] = ents
                sample["verified"] = False
                todo_samples.append(sample)

    with open(args.out, "w", encoding="utf-8") as fout:
        verified_samples.sort(key=lambda i: i["cats"])

        for sample in verified_samples:
            print(json.dumps(sample), file=fout)

        random.shuffle(todo_samples)

        for sample in todo_samples:
            print(json.dumps(sample), file=fout)

        print(f"First {len(verified_samples)} are already verified")


if __name__ == "__main__":
    main()
