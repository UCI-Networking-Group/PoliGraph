#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import os
import re
from collections import defaultdict

from poligrapher.document import PolicyDocument
from poligrapher.utils import setup_nlp_pipeline

# PoliCheck data types to PoliGraph data types
DATATYPE_REGEX = {
    "mac address": r"\b(mac address|mac|media access control|network card address)",
    "router ssid": r"\b(b?ssids?|id(entifier?)?\b.*\brouter|wi[ -]?fi ((end|access)[-\s]?point|name))",
    "android id": r"\b(android[_\s]?(ud)?id|android device id|SSAID)",
    "sim serial number": r"\b(imsi|international mobile subscriber identification number|sim code|iccid)",
    "imei": r"\b(imei|international mobile (equipment |device )id)",
    "advertising identifier": r"\b(ad\w* id\w*)",
    "gsfid": r"\bgsf id",
    "serial number": r"\b((device|android)('s)? serial (number|no)|serial number)",
    "email address": r"\b(e[\s-]?mail|electronic mail)\b.*\b(address|account|address information)",
    "phone number": r"\b((tele)?phone (mobile )?number|(mobile|fax|contact) number)",
    "person name": r"\bname",
    "geographical location": r"\b(A?GPS|(geo-?(graphic)?\s*)?location|coordinate|latitude|longitude)",
}


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("--nlp", required=True, help="NLP model directory")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    nlp = setup_nlp_pipeline(args.nlp)

    with open(args.output, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["app_id", "entity", "datatype", "text"])
        writer.writeheader()

        for d in args.workdirs:
            logging.info("Processing %s ...", d)

            document = PolicyDocument.load(d, nlp)

            app_id = os.path.basename(os.path.realpath(d))
            app_tuples = defaultdict(set)

            for doc in document.iter_docs():
                for sentence in doc.sents:
                    for datatype, regex in DATATYPE_REGEX.items():
                        if re.search(regex, sentence.text, re.I):
                            app_tuples[("we", datatype)].add(sentence.text)

            for (entity, datatype), all_text in app_tuples.items():
                writer.writerow({
                    "app_id": app_id,
                    "entity": entity,
                    "datatype": datatype,
                    "text": "\n".join(json.dumps(s) for s in all_text),
                })


if __name__ == "__main__":
    main()
