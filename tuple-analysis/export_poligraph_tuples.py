#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import os
from collections import defaultdict

from privacy_policy_analyzer.graph_utils import KGraph

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
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    with open(args.output, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["app_id", "entity", "datatype", "text"])
        writer.writeheader()

        for d in args.workdirs:
            logging.info("Processing %s ...", d)

            app_id = os.path.basename(os.path.realpath(d))

            kgraph_path = os.path.join(d, 'graph_trimmed.yml')
            kgraph = KGraph(kgraph_path)

            first_party_entities = set(kgraph.descendants("we"))
            first_party_entities.add("we")
            app_tuples = defaultdict(set)

            for dtype in DATATYPE_MAPPING_REVERSE.keys():
                policheck_dtype = DATATYPE_MAPPING_REVERSE[dtype]

                for entity in kgraph.who_collect(dtype):
                    all_text = kgraph.get_text(entity, dtype)

                    if entity in first_party_entities:
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
