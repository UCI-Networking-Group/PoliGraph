#!/usr/bin/env python3
import json
import random
import re
import sys

import faker
import inflect
from pkg_resources import DEVELOP_DIST
import spacy
import tqdm
import yaml
from requests_cache import CachedSession


def load_list(fname):
    with open(fname) as fin:
        for line in fin:
            line = line.strip()
            if line and not line.startswith('#'):
                yield line


class NERDataGenerator:
    def __init__(self, template_file, data_type_file):
        self.inflect_engine = inflect.engine()
        self.faker = faker.Faker()
        
        with open(data_type_file) as fin:
            yml_data = yaml.safe_load(fin)

        self.term_aliases = yml_data.pop("alias")

        self.data_types = []
        for v in yml_data.values():
            self.data_types.extend(v)

        self.templates = list(load_list(template_file))

    def __iter__(self):
        return self

    def __next__(self):

        def expand_data_type(term):
            parts = []

            for token in term.split():
                if token[0] == '(' and token[-1] == ')':
                    if random.randint(0, 1) == 0:
                        continue
                    else:
                        token = token[1:-1]

                if token in self.term_aliases:
                    token = random.choice(self.term_aliases[token])

                parts.append(token)

            return " ".join(parts)

        labels = []
        sentence = random.choice(self.templates)

        while True:
            match = re.search('{(?:DATA|ORG)}', sentence)

            if not match:
                break

            if match[0] == '{DATA}':
                label = "DATA"
                replaced_term = random.choice(self.data_types)
                replaced_term = expand_data_type(replaced_term)

                if random.randint(0, 1) == 0:
                    replaced_term = self.inflect_engine.plural(replaced_term)
            elif match[0] == '{ORG}':
                label = "ORG"
                replaced_term = " and "
                while " and " in replaced_term:
                    replaced_term = self.faker.company()

            ent_start = match.span()[0]
            labels.append((ent_start, ent_start + len(replaced_term), label))
            sentence = sentence.replace(match[0], replaced_term, 1)

        return sentence, labels


def main():
    TRAIN_SIZE = 20000
    DEV_SIZE = 2000
    NOISE_RATIO = 0.1

    spacy.prefer_gpu()

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    session = CachedSession("py_request_cache", backend="filesystem", use_temp=True)
    res = session.get("https://www.gutenberg.org/files/84/84-0.txt")

    noise_data = []
    for paragraph in re.split(r'(?:\r\n){2,}', res.text):
        paragraph = " ".join(paragraph.split('\r\n'))
        noise_data.extend(s.text for s in nlp(paragraph).sents)

    random.shuffle(noise_data)

    generator = NERDataGenerator("template.txt", "data_types.yml")
    nlp = spacy.blank("en")
    doc_bin = spacy.tokens.DocBin(attrs=["ENT_IOB", "ENT_TYPE"])

    for dataset, size in [("train", TRAIN_SIZE), ("dev", DEV_SIZE)]:
        n_normal_samples = size - int(size * NOISE_RATIO)

        for (text, annotations), i in zip(generator, tqdm.tqdm(range(size))):
            if i <= n_normal_samples:
                doc = nlp(text)
                ents = []
                for start, end, label in annotations:
                    if label == "DATA":
                        span = doc.char_span(start, end, label=label)
                        ents.append(span)

                doc.ents = ents
            else:
                doc = nlp(noise_data.pop())
                doc.ents = []

            doc_bin.add(doc)

        doc_bin.to_disk(f"{dataset}_dataset.spacy")


if __name__ == "__main__":
    main()
