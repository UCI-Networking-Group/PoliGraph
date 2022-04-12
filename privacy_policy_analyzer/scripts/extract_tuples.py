#!/usr/bin/env python3

import argparse

import spacy
from privacy_policy_analyzer.document import PolicyDocument
from privacy_policy_analyzer.utils import get_conjuncts



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", help="working directory")
    args = parser.parse_args()

    spacy.prefer_gpu()

    document = PolicyDocument(args.workdir)
    full_doc = document.get_full_doc()

    print("========== SUBSUMPTION ==========")
    for noun_chunk in full_doc.ents:
        sub_data_types = []

        for dest_token, relationship in document.get_links(noun_chunk.root):
            if relationship != "SUBSUM":
                continue

            sub_data_types.append(dest_token._.ent)

        if len(sub_data_types) > 0:
            print(noun_chunk, sub_data_types)

    print("========== COLLECTION ==========")
    for noun_chunk in full_doc.ents:
        entities = []

        for dest_token, relationship in document.get_links(noun_chunk.root):
            if relationship != "COLLECTED_BY":
                continue

            entities.append(dest_token)

        if len(entities) > 0:
            print(entities, noun_chunk)


if __name__ == "__main__":
    main()
