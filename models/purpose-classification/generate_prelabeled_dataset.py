#!/usr/bin/env python3

import argparse
import json
import logging
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from poligrapher.document import PolicyDocument
from poligrapher.utils import setup_nlp_pipeline

# This is where the candidate labels are defined.
PURPOSE_LABEL_MAPPING = {
    "service": "services",
    "functionality": "services",
    "transaction": "services",
    "maintenance": "services",
    "operation": "services",
    "security": "security",
    "authorization": "security",
    "authentication": "security",
    "legal": "legal",
    "liability": "legal",
    "acquisition": "legal",
    "analytics": "analytics",
    "research": "analytics",
    "advertising": "advertising",
    "marketing": "advertising",
}
CLASSIFICATION_LABELS = list(PURPOSE_LABEL_MAPPING.keys())
THRESHOLD_1 = 0.95
THRESHOLD_2 = 0.60


class PurposeClassifier:
    def __init__(self, is_multi_label=True):
        # Initialize the NLP model and classifier.
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
        model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
        self.classifier = pipeline(task='zero-shot-classification', tokenizer=tokenizer, model=model, device=0)

        self.is_multi_label = is_multi_label

    def __call__(self, text):
        def get_main_label(result_dict):
            labels = set()

            for label, score in zip(result_dict["labels"], result_dict["scores"]):
                relabel = PURPOSE_LABEL_MAPPING[label]

                if score > THRESHOLD_1:
                    labels.add(relabel)
                elif len(labels) == 0 and score > THRESHOLD_2:
                    labels.add(relabel)
                    break
                else:
                    break

            return labels

        results = self.classifier(sequences=text, candidate_labels=CLASSIFICATION_LABELS,
                                  multi_label=self.is_multi_label,
                                  num_workers=0)  # to prevent HuggingFace from spawning a lot of processes

        if isinstance(results, dict):
            return get_main_label(results)
        else:
            return [get_main_label(r) for r in results]


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp", default="", help="NLP model directory")
    parser.add_argument("-o", "--output", required=True, help="CSV file of prelabelled data")
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    args = parser.parse_args()

    nlp = setup_nlp_pipeline(args.nlp)
    purpose_classifier = PurposeClassifier()
    seen_phrases = set()

    with open(args.output, "w", newline="", encoding="utf-8") as fout:
        for d in args.workdirs:
            logging.info("Processing %s ...", d)

            document = PolicyDocument.load(d, nlp)
            phrases = []

            for _, purpose_src, edge_data in document.token_relationship.edges(data=True):
                if edge_data["relationship"] == "PURPOSE":
                    purpose_root = document.get_token_with_src(purpose_src)

                    left = purpose_root.left_edge.i
                    right = purpose_root.right_edge.i + 1
                    purpose_span = purpose_root.doc[left:right]

                    text = purpose_span.text

                    if text not in seen_phrases:
                        seen_phrases.add(text)
                        phrases.append(text)

            for text, purposes in zip(phrases, purpose_classifier(phrases)):
                print(json.dumps({
                    "text": text,
                    "label": sorted(purposes),
                    "policy_id": os.path.basename(os.path.realpath(d)),
                }), file=fout)

            fout.flush()


if __name__ == "__main__":
    main()
