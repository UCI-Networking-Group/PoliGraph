#!/usr/bin/env python3

import argparse
from collections import Counter
import csv
import re
import json
from pathlib import Path
from urllib.parse import urlparse

import networkx as nx
import tldextract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tracker_radar_data", help="Path to tracker-radar repo")
    parser.add_argument("crunch_base_data", help="Path to crunchbase-data repo")
    parser.add_argument("output", help="Path to output JSON file")
    args = parser.parse_args()

    tracker_radar_dir = Path(args.tracker_radar_data)
    crunch_base_dir = Path(args.crunch_base_data)

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
            }

            for k in info["properties"]:
                domain_mapping[k] = entity_id

    # CrunchBase data
    cb_data = dict()

    with open(crunch_base_dir / "companies.csv", "r", newline="", encoding="utf-8") as fin:
        for row in csv.DictReader(fin):
            permalink = row.pop("permalink")
            home_url = urlparse(row["homepage_url"])

            if row["status"] == "closed":
                continue

            if home_url.hostname is None or home_url.path.strip("/") != "":
                # Probably the company is hosting the homepage on domain not their own.
                row["domains"] = set()
                row["tracker_radar_entity"] = None
            else:
                home_domain = tldextract.extract(home_url.hostname).registered_domain
                row["domains"] = {home_domain}
                row["tracker_radar_entity"] = domain_mapping.get(home_domain)

            row["aliases"] = {row["name"]}
            cb_data[permalink] = row

    with open(crunch_base_dir / "acquisitions.csv", "r", newline="", encoding="utf-8") as fin:
        acquisition_graph = nx.DiGraph()

        for row in csv.DictReader(fin):
            company_permalink = row.pop("company_permalink")
            acquirer_permalink = row.pop("acquirer_permalink")

            if company_permalink not in cb_data:
                cb_data[company_permalink] = {
                    "name": row["company_name"],
                    "aliases": {row["company_name"]},
                    "domains": set(),
                    "tracker_radar_entity": None
                }

            if acquirer_permalink not in cb_data:
                cb_data[acquirer_permalink] = {
                    "name": row["acquirer_name"],
                    "aliases": {row["acquirer_name"]},
                    "domains": set(),
                    "tracker_radar_entity": None,
                }

            if acquirer_permalink == company_permalink:
                continue
            elif (company_permalink in acquisition_graph.nodes and
                  acquirer_permalink in nx.ancestors(acquisition_graph, company_permalink)):
                print(f"Acquisition of {company_permalink} by {acquirer_permalink} causes a cycle. Skipped.")
            else:
                acquisition_graph.add_edge(company_permalink, acquirer_permalink)

    for company_permalink, acquirer_permalink in nx.topological_sort(nx.line_graph(acquisition_graph)):
        if company_permalink not in cb_data:
            # A company can be acquired for multiple times. Only take the first one.
            continue

        company_info = cb_data[company_permalink]
        acquirer_info = cb_data[acquirer_permalink]

        if (company_info["tracker_radar_entity"] and
            company_info["tracker_radar_entity"] != acquirer_info["tracker_radar_entity"]):
            print(f"{company_permalink} and {acquirer_permalink} are different entities in Tracker Radar.")
            continue

        acquirer_info["aliases"].update(company_info["aliases"])
        acquirer_info["domains"].update(company_info["domains"])

        cb_data.pop(company_permalink)

    for cb_company_info in cb_data.values():
        entity_id = cb_company_info["tracker_radar_entity"] or cb_company_info["name"]

        # Fix buggy data in the CrunchBase data
        if not entity_id.isprintable():
            continue

        cb_company_info["aliases"] = [s for s in cb_company_info["aliases"] if s.isprintable()]

        if entity_id not in entity_info:
            entity_info[entity_id] = dict(aliases=set(), domains=set(), prevalence=0.0)

        info = entity_info[entity_id]
        info["aliases"].update(cb_company_info["aliases"])
        info["domains"].update(cb_company_info["domains"])

    with open(args.output, "w", encoding="utf-8") as fout:
        json.dump(entity_info, fout, default=lambda o: sorted(o) if isinstance(o, set) else o)


if __name__ == "__main__":
    main()
