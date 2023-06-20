import argparse
import csv
import importlib.resources as pkg_resources
import json
import os
from functools import lru_cache

import tldextract
import poligrapher
from poligrapher.graph_utils import ExtKGraph, KGraph, load_ontologies


class DomainMapper:
    def __init__(self, entity_info_json):
        self.domain_map = {}

        with open(entity_info_json, encoding="utf-8") as fin:
            for entity, entity_info in json.load(fin).items():
                self.domain_map.update({dom: entity for dom in entity_info["domains"]})

    @lru_cache
    def map_domain(self, domain):
        registered_domain = tldextract.extract(domain).registered_domain
        return self.domain_map.get(registered_domain)

    @lru_cache
    def map_package_name(self, package_name):
        reverse_package_name = ".".join(reversed(package_name.split(".")))
        return self.map_domain(reverse_package_name) or self.map_domain(package_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("flow_json", help="Flow JSON file")
    parser.add_argument("out_csv", help="Output CSV file")
    parser.add_argument("-p", "--privacy-policy-root", required=True, help="Input privacy policy directories")
    args = parser.parse_args()

    with pkg_resources.path(poligrapher, "extra-data") as extra_data:
        entity_info = extra_data / "entity_info.json"

    with open(args.flow_json, encoding="utf-8") as fin:
        input_data = json.load(fin)

    domain_mapper = DomainMapper(entity_info)
    data_ontology, entity_ontology = load_ontologies()

    all_data_types = set()

    with open(args.out_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["app_id", "data_type", "dest_entity",
                                                  "collection_consistency", "policheck_consistency", "purposes"])
        writer.writeheader()

        for app_id, app_info in input_data.items():
            kgraph_path = os.path.join(args.privacy_policy_root, app_info["privacy_policy_id"], "graph-original.yml")

            if not os.path.exists(kgraph_path):
                continue

            kgraph = KGraph(kgraph_path, merge_geolocation=True)
            ext_kgraph = ExtKGraph(kgraph_path, data_ontology, entity_ontology, merge_geolocation=True)
            seen_tuples = set()

            row = dict(app_id=app_id)

            for flow in app_info["flows"]:
                data_type = flow["data_type"]
                all_data_types.add(data_type)

                if flow["party"] == "first party":
                    entity = "we"
                else:
                    entity = domain_mapper.map_domain(flow["dest_domain"])

                if entity is None:
                    continue

                if (data_type, entity) in seen_tuples:
                    continue
                else:
                    seen_tuples.add((data_type, entity))

                row["data_type"] = data_type
                row["dest_entity"] = entity

                # "Entity-insensitive" consistency
                if kgraph.validate_collection(data_type):
                    row["collection_consistency"] = "clear"
                elif ext_kgraph.validate_collection(data_type):
                    row["collection_consistency"] = "vague"
                else:
                    row["collection_consistency"] = "omitted"

                # "Entity-sensitive" consistency
                # To align with PoliCheck: exclude unspecific_data but include unspecific_actor
                if kgraph.validate_sharing(entity, data_type):
                    row["policheck_consistency"] = "clear"
                elif ext_kgraph.validate_sharing(entity, data_type, accept_unspecific_data=False) \
                    or ext_kgraph.validate_sharing("UNSPECIFIED_ACTOR", data_type, accept_unspecific_data=False):
                    row["policheck_consistency"] = "vague"
                else:
                    row["policheck_consistency"] = "omitted"

                # Infer purposes
                possible_purposes = set()

                if row["policheck_consistency"] == "clear":
                    for p in kgraph.purposes(entity, data_type):
                        possible_purposes.add(p)

                if len(possible_purposes) == 0:
                    for p in ext_kgraph.purposes(entity, data_type):
                        possible_purposes.add(p)

                    for p in ext_kgraph.purposes("UNSPECIFIED_ACTOR", data_type):
                        possible_purposes.add(p)

                row["purposes"] = "|".join(sorted(possible_purposes))

                writer.writerow(row)

if __name__ == "__main__":
    main()
