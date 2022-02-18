#!/usr/bin/env python3
"""Download a web page and export the accessibility tree for parsing."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright
from requests_cache import CachedSession

READABILITY_JS_URL = "https://raw.githubusercontent.com/mozilla/readability/5ea9c2f/Readability.js"


def get_readability_js():
    session = CachedSession("py_request_cache", backend="filesystem", use_temp=True)
    res = session.get(READABILITY_JS_URL)
    res.raise_for_status()
    return res.text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="input URL")
    parser.add_argument("output", help="output dir")
    args = parser.parse_args()

    if args.url.split("://")[0] not in ["http", "https", "file"]:
        args.url = Path(args.url).absolute().as_uri()

    with sync_playwright() as p:
        # Firefox generates simpler accessibility tree than chromium
        # Tested on Debian's firefox-esr 91.5.0esr-1~deb11u1
        browser = p.firefox.launch()

        page = browser.new_page()
        page.set_viewport_size({"width": 1080, "height": 1920})
        page.goto(args.url)

        # apply readability.js
        page.add_script_tag(content=get_readability_js())
        cleaned_html = page.evaluate("""() => {
            const documentClone = document.cloneNode(true); 
            const article = new Readability(document).parse();
            return article.content;
        }""", [])
        page.set_content(cleaned_html)

        # obtain the accessibility tree
        snapshot = page.accessibility.snapshot(interesting_only=False)

        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / "cleaned.html", "w", encoding="utf-8") as fout:
            fout.write(cleaned_html)

        with open(output_dir / "accessibility_tree.json", "w", encoding="utf-8") as fout:
            json.dump(snapshot, fout)

        browser.close()


if __name__ == "__main__":
    main()
