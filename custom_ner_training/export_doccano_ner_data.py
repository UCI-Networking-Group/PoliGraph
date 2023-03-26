#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import tempfile
from zipfile import ZipFile

import spacy
from doccano_client import DoccanoClient
from spacy.tokens import DocBin
from spacy.training import Example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Doccano URL")
    parser.add_argument("project_id", help="Project ID")
    parser.add_argument("-j", "--jsonl-output", default=None, help="Save JSONL to this path")
    parser.add_argument("-s", "--spacy-output", default=None, help="Generate spaCy NER dataset to this path")
    parser.add_argument("-U", "--username", required=True, help="Username")
    parser.add_argument("-P", "--password", required=True, help="Password")
    args = parser.parse_args()

    client = DoccanoClient(args.url)
    client.login(username=args.username, password=args.password)

    with tempfile.TemporaryDirectory() as tempdir:
        zip_path = client.download(args.project_id, "JSONL", True, tempdir)
        jsonl_output = args.jsonl_output or os.path.join(tempdir, "dataset.jsonl")

        with ZipFile(zip_path) as zipf:
            with (zipf.open(f"{args.username}.jsonl") as fin,
                  open(jsonl_output, "wb") as fout):
                shutil.copyfileobj(fin, fout)

        if args.spacy_output:
            nlp = spacy.blank("en")
            db = DocBin()

            with open(jsonl_output, "r", encoding="utf-8") as fin:
                for line in fin:
                    jsonl_row = json.loads(line)

                    if "good" not in jsonl_row["cats"]:
                        continue

                    text = jsonl_row["text"]
                    annotations = {"entities": jsonl_row["entities"]}

                    example = Example.from_dict(nlp.make_doc(text), annotations)

                    doc = example.reference
                    doc.set_ents(doc.ents, default="missing")

                    db.add(doc)

            print(f"{len(db)} spaCy examples imported")
            db.to_disk(args.spacy_output)


if __name__ == "__main__":
    main()
