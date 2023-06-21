#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import os
import sqlite3
from collections import defaultdict

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


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("-e", "--policheck_root", help="PoliCheck root dir")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    con = sqlite3.connect(os.path.join(args.policheck_root, "output", "db", "consistency_results_1.db"))

    with open(args.output, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["app_id", "entity", "datatype", "text"])
        writer.writeheader()

        for d in args.workdirs:
            logging.info("Processing %s ...", d)

            app_id = os.path.basename(os.path.realpath(d))
            app_tuples = defaultdict(list)

            for sentence, entity, datatype in con.execute("""
                SELECT sentenceId, entity, data FROM AppPolicySentences S, Policy P
                WHERE S.policyId == P.policyId AND P.collect == "collect" AND S.appId == ?
            """, (app_id,)):
                if datatype in DATATYPE_MAPPING:
                    if entity == "we":
                        app_tuples[("we", datatype)].append(sentence)
                    else:
                        app_tuples[("3rd-party", datatype)].append(sentence)

            for (entity, datatype), all_text in app_tuples.items():
                writer.writerow({
                    "app_id": app_id,
                    "entity": entity,
                    "datatype": datatype,
                    "text": "\n".join(json.dumps(s) for s in dict.fromkeys(all_text)),
                })


if __name__ == "__main__":
    main()
