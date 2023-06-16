#!/usr/bin/env python3

import argparse
import csv
import re
import json
import logging
import os
from collections import defaultdict

from poligrapher.graph_utils import KGraph

# PoliCheck data types to PoliGraph data types
DATATYPE_MAPPING = {
    "mac address": ["mac address"],
    "router ssid": ["router ssid"],
    "android id": ["android id"],
    "sim serial number": ["sim serial number"],
    "imei": ["imei"],
    "advertising identifier": ["advertising id"],
    "gsfid": ["gsf id"],
    "serial number": ["serial number"],
    "email address": ["email address"],
    "phone number": ["phone number"],
    "person name": ["person name"],
    "geographical location": ["geolocation", "coarse geolocation", "precise geolocation"],
}
# PoliGraph data types to PoliCheck data types
DATATYPE_MAPPING_REVERSE = {pg_dt: pc_dt for pc_dt, li in DATATYPE_MAPPING.items() for pg_dt in li}


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("-v", "--variant", default="original", help="Variant of the graph")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    term_regex = re.compile(r'^(?P<term>.+?)(?: @(?P<subject>\S+))?(?: (?P<src>\(\d+, \d+\)))?$')

    with open(args.output, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["app_id", "entity", "datatype", "text"])
        writer.writeheader()

        for d in args.workdirs:
            logging.info("Processing %s ...", d)

            app_id = os.path.basename(os.path.realpath(d))

            kgraph_path = os.path.join(d, f'graph-{args.variant}.yml')
            kgraph = KGraph(kgraph_path)

            local_dtype_map = {node: term_regex.match(node)[1] for node in kgraph.datatypes}

            # First party entity names
            first_party_entity_nodes = {"we"}

            for entity_node in kgraph.entities:
                entity = term_regex.match(entity_node)[1]

                if entity == "we":
                    first_party_entity_nodes.add(entity_node)
                    first_party_entity_nodes.update(kgraph.descendants(entity_node))

            app_tuples = defaultdict(set)

            for dtype_node, dtype in local_dtype_map.items():
                if policheck_dtype := DATATYPE_MAPPING_REVERSE.get(dtype):
                    for entity_node in kgraph.who_collect(dtype_node):
                        all_text = kgraph.get_text(entity_node, dtype_node)

                        if entity_node in first_party_entity_nodes:
                            app_tuples[("we", policheck_dtype)].update(all_text)
                        else:
                            app_tuples[("3rd-party", policheck_dtype)].update(all_text)

            for entity, datatype in sorted(app_tuples):
                all_text = sorted(app_tuples[entity, datatype])

                writer.writerow({
                    "app_id": app_id,
                    "entity": entity,
                    "datatype": datatype,
                    "text": "\n".join(json.dumps(s) for s in all_text),
                })


if __name__ == "__main__":
    main()
