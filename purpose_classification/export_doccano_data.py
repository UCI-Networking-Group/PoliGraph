#!/usr/bin/env python3

import json
import tempfile
import argparse
from doccano_client import DoccanoClient
from zipfile import ZipFile

PURPOSE_LABELS = [
    "advertising",
    "analytics",
    "legal",
    "security",
    "services",
]

NONE_LABELS = [
    "other",
    "invalid"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Doccano URL")
    parser.add_argument("project_id", help="Project ID")
    parser.add_argument("output", help="Output dataset path")
    parser.add_argument("-U", "--username", required=True, help="Username")
    parser.add_argument("-P", "--password", required=True, help="Password")
    args = parser.parse_args()

    client = DoccanoClient(args.url)
    client.login(username=args.username, password=args.password)

    with tempfile.TemporaryDirectory() as tempdir:
        zip_path = client.download(args.project_id, "JSONL", True, tempdir)

        with ZipFile(zip_path) as zipf:
            with (zipf.open(f"{args.username}.jsonl") as fin,
                  open(args.output, "w", encoding="utf-8") as fout):

                for line in fin:
                    sample = json.loads(line)

                    print(f"Processing example #{sample['id']}")

                    text = sample["text"]
                    labels = sample["label"]
                    label_array = [0] * len(PURPOSE_LABELS)

                    if labels[0] not in NONE_LABELS:
                        for l in labels:
                            label_array[PURPOSE_LABELS.index(l)] = 1

                    print(json.dumps({
                        "text": text,
                        "label": label_array,
                    }), file=fout)


if __name__ == "__main__":
    main()
