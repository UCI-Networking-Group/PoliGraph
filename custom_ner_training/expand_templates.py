#!/usr/bin/env python3

import yaml
import exrex
import argparse
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("template_file", help="YAML template file")
    parser.add_argument("output_file", help="Output file")
    args = parser.parse_args()

    with open(args.template_file, encoding="utf-8") as fin:
        yml_data = yaml.safe_load(fin)

    alias_list = {}

    for key, expand_list in yml_data["alias"].items():
        alias_list[key] = "(" + "|".join(expand_list) + ")"

    all_phrases = set()

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for template in yml_data["template"]:
            for phrase in exrex.generate(template.format(**alias_list)):
                phrase = re.sub(r"\s+", " ", phrase.strip())

                if phrase not in all_phrases:
                    print(phrase, file=fout)
                    all_phrases.add(phrase)


if __name__ == "__main__":
    main()
