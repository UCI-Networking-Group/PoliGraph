#!/usr/bin/env python3
"""Download a web page and export the accessibility tree for parsing."""

import argparse
import json
import logging
from pathlib import Path
import urllib.parse as urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright
import requests
from requests_cache import CachedSession

READABILITY_JS_URL = "https://raw.githubusercontent.com/mozilla/readability/5ea9c2f/Readability.js"
REQUESTS_TIMEOUT = 10


def get_readability_js():
    session = CachedSession("py_request_cache", backend="filesystem", use_temp=True)
    res = session.get(READABILITY_JS_URL, timeout=REQUESTS_TIMEOUT)
    res.raise_for_status()
    return res.text


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="input URL or path")
    parser.add_argument("output", help="output dir")
    args = parser.parse_args()

    parsed_url = urlparse.urlparse(args.url)

    if parsed_url.scheme in ["http", "https"]:
        logging.info("Testing URL %r with HEAD request", args.url)

        try:
            requests.head(args.url, timeout=REQUESTS_TIMEOUT)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # Don't call the browser if we already know the URL doesn't work
            logging.error("Failed to connect to %r", args.url)
            logging.error("Error message: %s", e)
            exit(1)
    else:
        # Interpret args.url as a file path
        parsed_path = Path(args.url).absolute()

        if parsed_path.is_file():
            args.url = parsed_path.as_uri()
        else:
            logging.error("File %r not found", args.url)
            exit(1)

    firefox_configs = {
        # Bypass CSP so we can always inject scripts
        "security.csp.enable": False,
        # Allow insecure TLS versions
        "security.tls.version.min": 1,
        "security.tls.version.enable-deprecated": True,
        # Prevent some background traffic
        "dom.serviceWorkers.enabled": False,
        "network.websocket.max-connections": 0,
        "media.autoplay.default": 5,
        "media.peerconnection.enabled": False,
        "privacy.trackingprotection.enabled": True,
    }

    with sync_playwright() as p:
        # Firefox generates simpler accessibility tree than chromium
        # Tested on Debian's firefox-esr 91.5.0esr-1~deb11u1
        browser = p.firefox.launch(firefox_user_prefs=firefox_configs)

        page = browser.new_page()
        page.set_viewport_size({"width": 1080, "height": 1920})
        logging.info("Navigating to %r", args.url)

        # Record HTTP status and navigated URLs so we can check errors later
        url_status = dict()
        navigated_urls = []
        page.on("response", lambda r: url_status.update({r.url: r.status}))
        page.on("framenavigated", lambda f: f.parent_frame is None and navigated_urls.append(f.url))

        page.goto(args.url)

        try:
            page.wait_for_load_state("networkidle")
        except PlaywrightTimeoutError:
            logging.warning("Cannot reach networkidle but will continue")
            pass

        # Check HTTP errors
        for url in navigated_urls:
            if (status_code := url_status.get(url, 0)) >= 400:
                logging.error("Got HTTP status %d", status_code)
                browser.close()
                exit(-1)

        # Apply readability.js
        page.add_script_tag(content=get_readability_js())
        readability_info = page.evaluate("""() => {
            const documentClone = document.cloneNode(true);
            const article = new Readability(documentClone).parse();
            document.body.innerHTML = article.content;
            return article;
        }""", [])
        cleaned_html = page.content()

        # obtain the accessibility tree
        snapshot = page.accessibility.snapshot(interesting_only=False)

        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / "accessibility_tree.json", "w", encoding="utf-8") as fout:
            json.dump(snapshot, fout)

        with open(output_dir / "cleaned.html", "w", encoding="utf-8") as fout:
            fout.write(cleaned_html)

        with open(output_dir / "readability.json", "w", encoding="utf-8") as fout:
            json.dump(readability_info, fout)

        logging.info("Saved to %s", output_dir)
        browser.close()


if __name__ == "__main__":
    main()
