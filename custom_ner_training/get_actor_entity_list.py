#!/usr/bin/env python3

import csv
import io
import re

import inflect
import requests
import spacy

WIKIDATA_QUERY = """
SELECT DISTINCT ?item ?label ?className
WHERE
{
  {
    ?item (wdt:P31|wdt:P452)/wdt:P279* wd:Q1640628.    # network service
  } UNION {
    ?item (wdt:P31|wdt:P452)/wdt:P279* wd:Q11371.      # Internet service provider
  } UNION {
    ?item (wdt:P31|wdt:P452)/wdt:P279* wd:Q1641122.    # online service provider
  } UNION {
    ?item (wdt:P31|wdt:P452)/wdt:P279* wd:Q5154231.    # communications service provider
  } UNION {
    ?item (wdt:P31|wdt:P452)/wdt:P279* wd:Q7696092.    # telecommunications service provider
  } UNION {
    ?item (wdt:P31|wdt:P452)/wdt:P279* wd:Q105538253.  # online shop
  } UNION {
    ?item (wdt:P31|wdt:P452)/wdt:P279* wd:Q193424.     # web service
  } UNION {
    ?item (wdt:P31|wdt:P452)/wdt:P279* wd:Q18388277.   # technology company
  }

  ?item wdt:P856 ?website.  # has a website

  ?item rdfs:label ?label.
  FILTER(LANG(?label) = "en").

  ?item wdt:P31 ?class.
  ?class rdfs:label ?className.
  FILTER(LANG(?className) = "en").
}
"""

class_members = {}
nlp = spacy.load("en_core_web_sm")

req = requests.get("https://query.wikidata.org/sparql",
    params={"query": WIKIDATA_QUERY},
    headers={"Accept": "text/csv"},
)

with io.StringIO(req.text, newline='') as fin:
    for row in csv.DictReader(fin):
        name = row["label"]
        class_name = row["className"]

        try:
            class_members[class_name].append(name)
        except KeyError:
            class_members[class_name] = [name]

ACTOR_KEYWORDS = frozenset(['advertiser', 'affiliate', 'analytic', 'analytics', 'app', 'application',
                            'broker', 'business', 'carrier', 'company', 'corporation',
                            'distributor', 'network', 'operator', 'organization',
                            'partner', 'party', 'platform', 'processor', 'product', 'provider', 'publisher',
                            'service', 'site', 'software', 'subsidiary', 'vendor', 'website'])
inflect_engine = inflect.engine()
selected_class_names = []
actor_names = set()

for class_name, members in class_members.items():
    doc = nlp(class_name)
    if next(doc.sents).root.lemma_ in ACTOR_KEYWORDS:
        selected_class_names.append(class_name)

        plural = inflect_engine.plural_noun(class_name)
        actor_names.add("the {class_name}")
        actor_names.add(plural)
        actor_names.add("third-party {plural}")
        actor_names.add("affiliated {plural}")

        for s in members:
            if (not re.match(r'^(?:\w+\.)?\w+\.\w+$', s, re.ASCII) and
                re.match(r"""^[\w\s,.'"&+-]+$""", s, re.ASCII)):
                actor_names.add(s)

for class_name in ACTOR_KEYWORDS:
    actor_names.add(f'the {class_name}')
    actor_names.add(f'our {class_name}')
    plural = inflect_engine.plural_noun(class_name)
    actor_names.add(f'third-party {plural}')

with open("actor_entities.list", "w", encoding="utf-8") as fout:
    for name in sorted(actor_names):
        print(name, file=fout)
