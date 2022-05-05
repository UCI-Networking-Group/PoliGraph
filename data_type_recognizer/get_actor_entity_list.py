#!/usr/bin/env python3

import csv
import inflect
import spacy
import re

class_members = dict()
nlp = spacy.load("en_core_web_md")

with open("query.csv", "r", encoding="utf-8") as fin:
    for row in csv.DictReader(fin):
        name = row["label"]
        class_name = row["className"]

        try:
            class_members[class_name].append(name)
        except KeyError:
            class_members[class_name] = [name]

actor_keywords = frozenset(['advertiser', 'affiliate', 'app', 'application', 'broker', 'business',
                            'carrier', 'company', 'corporation', 'distributor', 'network', 'operator',
                            'organization', 'partner', 'party', 'platform', 'processor', 'product',
                            'provider', 'publisher', 'service', 'site', 'software', 'subsidiary',
                            'vendor', 'website'])
inflect_engine = inflect.engine()
selected_class_names = []
actor_names = set()

for class_name, members in class_members.items():
    doc = nlp(class_name)
    if next(doc.sents).root.lemma_ in actor_keywords:
        selected_class_names.append(class_name)

        plural = inflect_engine.plural_noun(class_name)
        actor_names.add("the " + class_name)
        actor_names.add(plural)
        actor_names.add("third-party " + plural)
        actor_names.add("affiliated " + plural)

        for s in members:
            if (not re.match(r'^(?:\w+\.)?\w+\.\w+$', s, re.ASCII) and
                re.match(r"""^[\w\s,.'"&+-]+$""", s, re.ASCII)):
                actor_names.add(s)

with open("actor_entities.list", "w", encoding="utf-8") as fout:
    for name in sorted(actor_names):
        print(name, file=fout)
