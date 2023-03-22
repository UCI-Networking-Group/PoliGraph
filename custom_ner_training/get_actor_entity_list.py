#!/usr/bin/env python3
"""Generates list of ACTOR names for NER training"""

import argparse
import csv
import io
import re

import inflect
import requests
import spacy

ACTOR_KEYWORDS = frozenset([
    'agency', 'advertiser', 'affiliate', 'analytic', 'analytics', 'app', 'application',
    'broker', 'business', 'carrier', 'company', 'corporation', 'distributor',
    'group', 'institution', 'network', 'operator', 'organization',
    'partner', 'party', 'platform', 'processor', 'product', 'provider', 'publisher',
    'service', 'site', 'software', 'subsidiary', 'vendor', 'website',
])

SINGULAR_PREFIXES = ["the", "our", "this", "another"]

PLURAL_PREFIXES = [
    "third-party", "third party", "3rd-party", "affiliated", "external",
    "our", "such", "these", "following", "other", "certain",
]

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

DOMAIN_RE = re.compile(r'^(?:[\w-]+\.)*[\w-]+\.[\w-]+$', re.ASCII)
REGULAR_CHARS_RE = re.compile(r"""^[\w\s,.'"&+-]+$""", re.ASCII)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", help="Output file")
    args = parser.parse_args()

    class_members = {}
    nlp = spacy.load("en_core_web_sm")

    # Crawl names from Wikidata
    req = requests.get("https://query.wikidata.org/sparql",
        params={"query": WIKIDATA_QUERY},
        headers={"Accept": "text/csv"},
        timeout=30,
    )

    # Using className to categorize Wikidata names
    with io.StringIO(req.text, newline='') as fin:
        for row in csv.DictReader(fin):
            name = row["label"]
            class_name = row["className"]

            try:
                class_members[class_name].append(name)
            except KeyError:
                class_members[class_name] = [name]

    # Populate valid actor_names
    inflect_engine = inflect.engine()
    selected_class_names = []
    actor_names = set()

    for class_name, members in class_members.items():
        doc = nlp(class_name)

        if next(doc.sents).root.lemma_ in ACTOR_KEYWORDS:
            selected_class_names.append(class_name)

            plural = inflect_engine.plural_noun(class_name)
            actor_names.add(plural)

            for prefix in SINGULAR_PREFIXES:
                actor_names.add(f"{prefix} {class_name}")

            for prefix in PLURAL_PREFIXES:
                actor_names.add(f"{prefix} {plural}")

            for s in members:
                if (DOMAIN_RE.match(s) is None                 # Not like a domain
                    and REGULAR_CHARS_RE.match(s) is not None  # Contains only "good" characters
                    and any(c.isupper() for c in s)):          # Has at least one uppercase character
                    actor_names.add(s)

    for class_name in ACTOR_KEYWORDS:
        plural = inflect_engine.plural_noun(class_name)
        actor_names.add(plural)

        for prefix in SINGULAR_PREFIXES:
            actor_names.add(f"{prefix} {class_name}")

        for prefix in PLURAL_PREFIXES:
            actor_names.add(f"{prefix} {plural}")

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for name in sorted(actor_names):
            print(name, file=fout)


if __name__ == "__main__":
    main()
