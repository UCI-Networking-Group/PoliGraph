#!/usr/bin/env python3

import argparse
import json
import tempfile
from zipfile import ZipFile

from doccano_client import DoccanoClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Doccano URL")
    parser.add_argument("project_id", help="Project ID")
    parser.add_argument("-o", "--jsonl-output", default=None, help="Save JSONL to this path")
    parser.add_argument("-U", "--username", required=True, help="Username")
    parser.add_argument("-P", "--password", required=True, help="Password")
    args = parser.parse_args()

    client = DoccanoClient(args.url)
    client.login(username=args.username, password=args.password)

    with tempfile.TemporaryDirectory() as tempdir:
        zip_path = client.download(args.project_id, "JSONL", True, tempdir)
        all_samples = []

        with ZipFile(zip_path) as zipf:
            with zipf.open(f"{args.username}.jsonl") as fin:
                for line in fin:
                    sample = json.loads(line)

                    if 'x' in sample['Comments'] or not sample['relations']:
                        continue

                    del sample['Comments']
                    all_samples.append(sample)

    all_samples.sort(key=lambda item: item['app_id'])

    with open(args.jsonl_output, "w", encoding="utf-8") as fout:
        for sample in all_samples:
            print(json.dumps(sample), file=fout)
                    

if __name__ == "__main__":
    main()
