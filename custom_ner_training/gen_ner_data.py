#!/usr/bin/env python3
import argparse
import itertools
import random
import re
from pathlib import Path

import spacy
from requests_cache import CachedSession
from spacy.tokens import Doc, Span
from unidecode import unidecode

from poligrapher.document import PolicyDocument
from poligrapher.utils import align_noun_phrases

DATATYPE_KEYWORDS = frozenset([
    "information", "data", "datum", "address", "number", "location",
    "geolocation", "identifier", "id", "preference", "setting", "cookie",
])
ACTOR_KEYWORDS = frozenset([
    'agency', 'advertiser', 'affiliate', 'analytic', 'analytics', 'app',
    'application', 'broker', 'business', 'carrier', 'company', 'corporation',
    'distributor', 'group', 'institution', 'network', 'operator',
    'organization', 'partner', 'party', 'platform', 'processor', 'product',
    'provider', 'publisher', 'service', 'site', 'software', 'subsidiary',
    'vendor', 'website',
])
IGNORE_KEYWORDS = frozenset([
    'act', 'regulartion', 'law', 'policy', 'jurisdiction', 'purpose', 'notice',
    'security', 'safety', 'question', 'protection', 'behalf', 'consent', 'term',
    'disclosure', 'obligation', 'agreement', 'mean', 'stage', 'right', 'fraud',
    'practice', 'basis', 'device', 'friend', 'consumer', 'child', 'kid', 'age',
    'user', 'parent', 'guardian', 'minor', 'customer', 'visitor', 'guest',
])


def load_list(fname):
    with open(fname, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line and not line.startswith('#'):
                yield line


def rule_based_ner(doc: Doc):
    entities = []
    unknown_spans = []
    blocked_spans = []

    for span in doc.spans["noun_phrases"]:
        root_token = span.root
        ent_label = span.label_

        # Rule-based NER completion
        if ent_label in {"DATA", "ACTOR"}:
            new_ent_label = ent_label
        elif ent_label in {"EVENT", "FAC", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"}:
            new_ent_label = "ACTOR"
        elif ent_label in {"LAW", "LANGUAGE", "GPE", "DATE", "TIME", "PERCENT", "MONEY",
                           "QUANTITY", "ORDINAL", "CARDINAL"}:
            new_ent_label = None
        elif root_token.lemma_.lower() in DATATYPE_KEYWORDS:
            new_ent_label = "DATA"
        elif root_token.lemma_.lower() in ACTOR_KEYWORDS:
            new_ent_label = "ACTOR"
        elif root_token.lemma_.lower() in IGNORE_KEYWORDS and root_token.pos_ == "NOUN":
            new_ent_label = None
        elif root_token.pos_ == "PRON":
            new_ent_label = None
        else:
            new_ent_label = "UNKNOWN"

        match new_ent_label:
            case None:
                blocked_spans.append(Span(doc, span.start, span.end))
            case "UNKNOWN":
                unknown_spans.append(Span(doc, span.start, span.end))
            case _:
                entities.append(Span(doc, span.start, span.end, new_ent_label))

    doc.set_ents(entities, missing=unknown_spans, outside=blocked_spans, default="outside")
    return doc


class NERDataGenerator:
    P_CAPITALIZE_SENT_BEGINNING = 0.80
    P_TITLECASE = 0.10

    def __init__(self, sentence_templates, data_types, actor_entities):
        self.templates = list(sentence_templates)
        self.candidate_terms = {
            "DATA": list(data_types),
            "ACTOR": list(actor_entities),
        }

    def __iter__(self):
        return self

    def __next__(self):
        labels = []
        sentence = random.choice(self.templates)

        while True:
            match = re.search(r'%([^%]+)%', sentence)

            if not match:
                break

            label, *candidates = [s.strip() for s in match[1].split(":")]

            if len(candidates) == 0:
                candidates = self.candidate_terms[label]

            replacing_term = random.choice(candidates)
            ent_start = match.span()[0]
            labels.append((ent_start, ent_start + len(replacing_term), label))

            if random.random() < NERDataGenerator.P_TITLECASE:
                replacing_term = replacing_term.title()
            elif ent_start == 0 and random.random() < NERDataGenerator.P_CAPITALIZE_SENT_BEGINNING:
                replacing_term = replacing_term[0].upper() + replacing_term[1:]

            sentence = sentence.replace(match[0], replacing_term, 1)

        sentence = sentence + random.choice(['.', ';', ''])

        return sentence, labels


def get_synthetic_data(nlp, size, sentence_template_file, data_type_list_file, actor_entity_list_file):
    generator = NERDataGenerator(
        load_list(sentence_template_file),
        load_list(data_type_list_file),
        load_list(actor_entity_list_file)
    )

    synthetic_data_tuples = itertools.islice(generator, size)

    for doc, annotations in nlp.pipe(synthetic_data_tuples, as_tuples=True):
        ents = [doc.char_span(sp, ep, label=label) for sp, ep, label in annotations]
        doc.set_ents(filter(lambda e: e is not None, ents))
        doc.user_data["source"] = "synthetic"
        yield doc


def get_rule_rehearsal_data(workdirs, nlp):
    for d in workdirs:
        document = PolicyDocument.initialize(d, nlp=nlp)

        for doc in document.iter_docs():
            doc = rule_based_ner(doc)
            doc.user_data["source"] = "rule-rehearsal"
            yield doc


def get_noise_rehearsal_data(nlp):
    rehearsal_data_sources = [
        # Corpus of rehearsal data. Hopefully there is no DATA types at all.
        "https://www.gutenberg.org/files/84/84-0.txt",
        "https://www.gutenberg.org/cache/epub/67503/pg67503.txt",
        "https://www.gutenberg.org/cache/epub/64317/pg64317.txt",
        "https://www.gutenberg.org/files/12/12-0.txt",
    ]

    # Generate "noise" samples from irrelevant text
    session = CachedSession("py_request_cache", backend="filesystem", use_temp=True)
    rehearsal_sentences = []

    sentencizer = spacy.blank("en")
    sentencizer.add_pipe("sentencizer")

    for url in rehearsal_data_sources:
        res = session.get(url)

        for paragraph in re.split(r'(?:\r\n){2,}', res.text):
            paragraph.replace("_", "")
            paragraph = unidecode(paragraph)
            paragraph = " ".join(paragraph.split('\r\n'))

            rehearsal_sentences.extend(s.text for s in sentencizer(paragraph).sents)

    random.shuffle(rehearsal_sentences)

    for doc in nlp.pipe(rehearsal_sentences):
        doc = rule_based_ner(doc)
        doc.user_data["source"] = "noise-rehearsal"
        yield doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", default=50000, type=int,
                        help="Number of sythetic samples")
    parser.add_argument("root_dir",
                        help="Path to templates of synthetic data as well as output dataset")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)

    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("align_noun_phrases", name="align_noun_phrases", after="ner")

    doc_bin = spacy.tokens.DocBin(attrs=["ENT_IOB", "ENT_TYPE"])

    print("Generating synthetic data...")
    count = 0

    for doc in get_synthetic_data(nlp, args.synthetic,
                                  root_dir / "templates.list",
                                  root_dir / "data_types.list",
                                  root_dir / "actor_entities.list"):
        doc_bin.add(doc)
        count += 1

    print(f"# synthetic data = {count}")

    print("Generating rule rehearsal data...")
    count = 0

    for doc in get_rule_rehearsal_data((root_dir / "rule_rehearsal").iterdir(), nlp):
        doc_bin.add(doc)
        count += 1

    print(f"# rule rehearsal data = {count}")   # expected: 36480

    print("Generating noise rehearsal data...")
    count = 0

    for doc in get_noise_rehearsal_data(nlp):
        doc_bin.add(doc)
        count += 1

    print(f"# noise rehearsal data = {count}")  # expected: 11632

    doc_bin.to_disk(root_dir / "dataset.spacy")


if __name__ == "__main__":
    main()
