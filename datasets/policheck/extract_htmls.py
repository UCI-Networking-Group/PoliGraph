#!/usr/bin/env python3

import pickle
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import bs4
import langdetect
import tldextract

input_dir, output_dir = sys.argv[1:]
input_dir = Path(input_dir)
output_dir = Path(output_dir)

output_dir.mkdir(exist_ok=True)

for f in input_dir.glob("*.pkl"):
    bname = f.name.split(".", 1)[0]

    with f.open("rb") as fin:
        res = pickle.load(fin)

    if res is None:
        continue

    original_link = None

    for link_header in res.headers["Link"].split(", "):
        if m := re.match(r'^<(\S+)>; rel="original"$', link_header.strip()):
            original_link = m[1]

    parsed_url = urlparse(original_link)
    parsed_domain = tldextract.extract(parsed_url.hostname)
    if parsed_domain.subdomain in ["www", ""] and parsed_url.path == "/" and parsed_url.query == "":
        print(f"Skip {original_link} (like homepage)")
        continue

    soup = bs4.BeautifulSoup(res.content, 'lxml')
    soup_text = soup.body.text if soup.body else ""
    lang = (soup.html or {}).get("lang")

    if lang is None:
        try:
            lang = langdetect.detect(soup_text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = "UNKNOWN"

    if not lang.lower().startswith("en"):
        print(f"Skip {original_link} (not English)")
        continue

    if re.search(r"(data|privacy)\s*(?:policy|notice)", soup_text, re.I) is None:
        print(f"Skip {original_link} (not a privacy policy)")
        continue

    with open(output_dir / f"{bname}.html", "wb") as fout:
        print(f"Accept {original_link}")
        fout.write(res.content)
