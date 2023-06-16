#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import networkx as nx
import spacy
import tldextract

CRUNCHBASE_CATEGORY_MAPPING = {
    "Advertising": ["advertiser"],
    "Mobile Advertising": ["advertiser"],
    "Advertising Platforms": ["advertiser"],
    "Advertising Networks": ["advertiser"],
    "Advertising Exchanges": ["advertiser"],

    "Analytics": ["analytic provider"],
    "Big Data Analytics": ["analytic provider"],
    "Predictive Analytics": ["analytic provider"],

    "SMS": ["social media"],

    "Email Marketing": ["email service provider"],
    "Email": ["email service provider"],
    "Email Newsletters": ["email service provider"],
}

TRACKER_RADAR_CATEGORY_MAPPING = {
    "SSO": ["auth provider"],
    "Federated Login": ["auth provider"],

    "Social Network": ["social media"],
    "Social - Comment": ["social media"],
    "Social - Share": ["social media"],

    "Embedded Content": ["content provider"],

    "Third-Party Analytics Marketing": ["analytic provider"],
    "Audience Measurement": ["analytic provider"],
    # These categories are analytics but it often include first-party domains
    # "Analytics": ["analytic provider"],
    # "Session Replay": ["analytic provider"],

    "Advertising": ["advertiser"],
    "Ad Fraud": ["advertiser"],
    # This category is a bit broader
    # "Ad Motivated Tracking": ["advertiser"],
}


def cache_ngrams(entity_info):
    # en_core_web_md supports is_oov
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "lemmatizer"])

    # Gather all names for NLP processing
    entity_names = {e: i["aliases"] for e, i in entity_info.items()}
    name_to_doc = dict()
    all_names = set.union(*[s for s in entity_names.values()])

    # Domain to entity mapping. Put domain names in all_names as well.
    domain_mapping = dict()

    for entity, info in entity_info.items():
        for full_domain in info["domains"]:
            domain_mapping[full_domain] = entity

            domain = tldextract.extract(full_domain).domain
            all_names.add(domain)

    for doc in nlp.pipe(all_names):
        name_to_doc[doc.text] = doc

    # Find all ngrams for fuzzy matching
    all_ngrams = dict()

    for entity, info in entity_info.items():
        ngrams = dict()

        for name in entity_names[entity]:
            doc = name_to_doc[name]

            # Find all ngrams that do not start/end with a punct/space
            #   oov_flag: determine whether to do case-sensitive match
            #   like_name_flag: if false, will be dropped later
            for i in range(len(doc)):
                like_name_flag = False
                oov_flag = False

                if doc[i].is_space or doc[i].is_punct:
                    continue

                for j in range(i, len(doc)):
                    if doc[j].is_space or doc[j].is_punct:
                        continue

                    like_name_flag |= doc[j].pos_ == "PROPN"
                    oov_flag |= doc[j].is_oov
                    ngrams[doc[i:j+1].text.lower()] = (oov_flag, like_name_flag)

        # Also put domain names to ngram list
        for full_domain in info["domains"]:
            domain = tldextract.extract(full_domain).domain

            if domain not in ngrams:
                doc = name_to_doc[domain]
                oov_flag = any(t.is_oov for t in doc)
                ngrams[domain] = (oov_flag, oov_flag)

        # If an ngram can be found in more than one entities, remove it
        for s, (oov_flag, like_name_flag) in ngrams.items():
            if s not in all_ngrams:
                all_ngrams[s] = (entity, oov_flag, like_name_flag)
            elif all_ngrams[s] is not None:
                all_ngrams[s] = None

    for v in entity_info.values():
        v["ngrams"] = dict()

    # Filter out ngrams that has like_name_flag=True and uniquely identifies entities
    for ngram, ngram_info in all_ngrams.items():
        if ngram_info and ngram_info[-1]:
            entity = ngram_info[0]
            oov_flag = ngram_info[1]

            entity_info[entity]["ngrams"][ngram] = oov_flag

    # For well-known online trackers, main entity name should always be in the ngram_mapping
    for entity, info in entity_info.items():
        if info["prevalence"] > 2e-5:
            entity_info[entity]["ngrams"][entity.lower()] = False


def load_crunchbase_data(crunchbase_dir, tracker_radar_domain_mapping):
    # CrunchBase data
    crunchbase_records = dict()

    # companies.csv: main company data
    with open(crunchbase_dir / "companies.csv", "r", newline="", encoding="utf-8") as fin:
        for row in csv.DictReader(fin):
            permalink = row.pop("permalink")
            home_url = urlparse(row["homepage_url"])

            if row["status"] == "closed":
                continue

            if home_url.hostname is None or home_url.path.strip("/") != "":
                # Ignore because the company likely hosts the homepage on a third-party domain
                row["domains"] = set()
                row["tracker_radar_entity"] = None
            else:
                home_domain = tldextract.extract(home_url.hostname).registered_domain
                row["domains"] = {home_domain}
                row["tracker_radar_entity"] = tracker_radar_domain_mapping.get(home_domain)

            row["categories"] = set(row["category_list"].split("|"))
            row["aliases"] = {row["name"]}
            crunchbase_records[permalink] = row

    # acquisitions.csv: merge companies according to acquisition records
    with open(crunchbase_dir / "acquisitions.csv", "r", newline="", encoding="utf-8") as fin:
        acquisition_graph = nx.DiGraph()

        for row in csv.DictReader(fin):
            company_permalink = row.pop("company_permalink")
            acquirer_permalink = row.pop("acquirer_permalink")

            for permalink, role in ((company_permalink, "company"), (acquirer_permalink, "acquirer")):
                if permalink not in crunchbase_records:
                    crunchbase_records[permalink] = {
                        "name": row[f"{role}_name"],
                        "aliases": {row[f"{role}_name"]},
                        "categories": set(row[f"{role}_category_list"].split("|")),
                        "domains": set(),
                        "tracker_radar_entity": None
                    }

            if acquirer_permalink == company_permalink:
                continue
            elif (company_permalink in acquisition_graph.nodes and
                  acquirer_permalink in nx.ancestors(acquisition_graph, company_permalink)):
                print(f"Acquisition of {company_permalink} by {acquirer_permalink} causes a cycle. Skipped.")
            else:
                acquisition_graph.add_edge(company_permalink, acquirer_permalink)

    for company_permalink, acquirer_permalink in nx.topological_sort(nx.line_graph(acquisition_graph)):
        if company_permalink not in crunchbase_records:
            # A company can be acquired for multiple times. Only take the first one.
            continue

        company_info = crunchbase_records[company_permalink]
        acquirer_info = crunchbase_records[acquirer_permalink]

        if (company_info["tracker_radar_entity"] and
            company_info["tracker_radar_entity"] != acquirer_info["tracker_radar_entity"]):
            print(f"{company_permalink} and {acquirer_permalink} are different entities in Tracker Radar.")
            continue

        acquirer_info["aliases"].update(company_info["aliases"])
        acquirer_info["domains"].update(company_info["domains"])
        # This often cause acquirer to have too many categories
        # acquirer_info["categories"].update(company_info["categories"])

        crunchbase_records.pop(company_permalink)

    returned_items = []
    for permalink, info in crunchbase_records.items():
        if not info["name"].isprintable():
            continue

        returned_info = dict()
        returned_items.append(returned_info)

        for key in "name", "aliases", "domains", "tracker_radar_entity":
            returned_info[key] = info[key]

        returned_info["categories"] = categories = set()

        for cat in info["categories"]:
            if cat in CRUNCHBASE_CATEGORY_MAPPING:
                categories.update(CRUNCHBASE_CATEGORY_MAPPING[cat])

    return returned_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tracker_radar_data", help="Path to tracker-radar repo")
    parser.add_argument("crunchbase_data", help="Path to crunchbase-data repo")
    parser.add_argument("output", help="Path to output JSON file")
    args = parser.parse_args()

    tracker_radar_dir = Path(args.tracker_radar_data)
    crunchbase_dir = Path(args.crunchbase_data)

    entity_info = dict()
    domain_mapping = dict()

    # DuckDuckGo Tracker Radar data
    with open(tracker_radar_dir / "build-data" / "generated" / "entity_map.json", "r", encoding="utf-8") as fin:
        for canonical_name, info in json.load(fin).items():
            # Use the displayName as the identifier for the company
            entity_id = info.get("displayName") or info.get("displayname")

            with open(tracker_radar_dir / "entities" / re.sub(r'["?/!:*|\\]', "", canonical_name + ".json"),
                      "r", encoding="utf-8") as fin:
                extra_info = json.load(fin)

            # Possible variants of company names
            aliases = set(info["aliases"])
            aliases.add(canonical_name)
            aliases.add(entity_id)

            entity_info[entity_id] = {
                "aliases": aliases,
                "domains": set(info["properties"]),
                "prevalence": extra_info.get("prevalence", {}).get("total", 0.0),
                "categories": set(),
            }

            for k in info["properties"]:
                domain_mapping[k] = entity_id

    for domain_json in (tracker_radar_dir / "domains").rglob("*.json"):
        with domain_json.open(encoding="utf-8") as fin:
            domain_info = json.load(fin)

        domain = domain_info["domain"]

        if domain not in domain_mapping:
            # print(domain)
            continue

        entity_id = domain_mapping[domain]

        domain_categories = set()
        for cat in domain_info["categories"]:
            if cat in TRACKER_RADAR_CATEGORY_MAPPING:
                domain_categories.update(TRACKER_RADAR_CATEGORY_MAPPING[cat])

        entity_info[entity_id]["categories"].update(domain_categories)

    # CrunchBase data
    cb_data = load_crunchbase_data(crunchbase_dir, domain_mapping)

    # Merge cb_data to entity_info
    for cb_company_info in cb_data:
        entity_id = cb_company_info["tracker_radar_entity"] or cb_company_info["name"]

        if entity_id not in entity_info:
            entity_info[entity_id] = dict(aliases=set(), domains=set(), categories=set(), prevalence=0.0)

        info = entity_info[entity_id]
        info["aliases"].update(cb_company_info["aliases"])
        info["domains"].update(cb_company_info["domains"])
        info["categories"].update(cb_company_info["categories"])

    cache_ngrams(entity_info)

    # Remove entities that causes a lot false detections
    entity_to_delete = ["Online", "Platform", "Answers", "Rokt"]

    # Remove entities with prevalence=0 and we don't know the category (to avoid false detections)
    for entity, info in entity_info.items():
        if info["prevalence"] == 0 and len(info["categories"]) == 0:
            entity_to_delete.append(entity)

    for entity in entity_to_delete:
        del entity_info[entity]

    # Custom entries for MobiPurpose / PoliCheck traffic
    # Third parties
    entity_info["ironSource"]["domains"].add("ironbeast.io")
    entity_info["InMobi"]["domains"].add("aerserv.com")
    entity_info["Start.io"] = {
        "aliases": [],
        "domains": ["startappexchange.com", "startappservice.com"],
        "prevalence": 0.0,
        "categories": ["advertiser", "analytic provider"],
        "ngrams": {"start.io": True, "startapp": True}
    }
    entity_info["SessionM"] = {
        "aliases": [],
        "domains": ["sessionm.com"],
        "prevalence": 0.0,
        "categories": ["analytic provider"],
        "ngrams": {"sessionm": True}
    }
    entity_info["Kidoz"] = {
        "aliases": [],
        "domains": ["kidoz.net"],
        "prevalence": 0.0,
        "categories": ["advertiser"],
        "ngrams": {"kidoz": True}
    }
    entity_info["GreedyGame"] = {
        "aliases": [],
        "domains": ["greedygame.com"],
        "prevalence": 0.0,
        "categories": ["advertiser"],
        "ngrams": {"greedygame": True}
    }
    entity_info["SponsorPay"] = {
        "aliases": [],
        "domains": ["sponsorpay.com"],
        "prevalence": 0.0,
        "categories": ["advertiser"],
        "ngrams": {"sponsorpay": True}
    }
    entity_info["Cloudmobi"] = {
        "aliases": [],
        "domains": ["cloudmobi.net"],
        "prevalence": 0.0,
        "categories": ["advertiser"],
        "ngrams": {"Cloudmobi": True}
    }
    entity_info["AppTornado"] = {
        "aliases": ["AppBrain"],
        "domains": ["appbrain.com", "apptornado.com"],
        "prevalence": 0.0,
        "categories": ["advertiser"],
        "ngrams": {"appbrain": True, "apptornado": True}
    }
    # Developers
    entity_info["Kaufcom"] = {
        "aliases": [],
        "domains": ["maxpedia.com", "kauf.com"],
        "prevalence": 0.0,
        "categories": [],
        "ngrams": {"kaufcom": True}
    }
    entity_info["GOMO"] = {
        "aliases": [],
        "domains": ["jiubang.com", "goforandroid.com", "gomo.com"],
        "prevalence": 0.0,
        "categories": [],
        "ngrams": {"gomo": True, "jiubang": True}
    }

    with open(args.output, "w", encoding="utf-8") as fout:
        json.dump(entity_info, fout, default=lambda o: sorted(o) if isinstance(o, set) else o)


if __name__ == "__main__":
    main()
