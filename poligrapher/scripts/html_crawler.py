#!/usr/bin/env python3
"""Download a web page and export the accessibility tree for parsing."""

import argparse
import base64
import json
import logging
from pathlib import Path
import re
import sys
import urllib.parse as urlparse

import bs4
import langdetect
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright
import requests
from requests_cache import CachedSession

READABILITY_JS_COMMIT = "8e8ec27cd2013940bc6f3cc609de10e35a1d9d86"
READABILITY_JS_URL = f"https://raw.githubusercontent.com/mozilla/readability/{READABILITY_JS_COMMIT}"
REQUESTS_TIMEOUT = 10


def get_readability_js():
    session = CachedSession("py_request_cache", backend="filesystem", use_temp=True)
    js_code = []

    res = session.get(f"{READABILITY_JS_URL}/Readability.js", timeout=REQUESTS_TIMEOUT)
    res.raise_for_status()
    js_code.append(res.text)
    js_code.append(res.text)

    res = session.get(f"{READABILITY_JS_URL}/Readability-readerable.js", timeout=REQUESTS_TIMEOUT)
    res.raise_for_status()
    js_code.append(res.text)

    return "\n".join(js_code)


def url_arg_handler(url):
    parsed_url = urlparse.urlparse(url)

    # Not HTTP(s): interpret as a file path
    if parsed_url.scheme not in ["http", "https"]:
        parsed_path = Path(url).absolute()

        if not parsed_path.is_file():
            logging.error("File %r not found", url)
            return None

        return parsed_path.as_uri()

    # Handle Google Docs URLs
    if (parsed_url.hostname == "docs.google.com"
            and not parsed_url.path.endswith("/pub")
            and (m := re.match(r"/document/d/(1[a-zA-Z0-9_-]{42}[AEIMQUYcgkosw048])", parsed_url.path))):
        logging.info("Exporting HTML from Google Docs URL...")

        export_url = f"https://docs.google.com/feeds/download/documents/export/Export?id={m[1]}&exportFormat=html"

        req = requests.get(export_url)
        req.raise_for_status()

        base64_url = "data:text/html;base64," + base64.b64encode(req.content).decode()

        return base64_url

    # Handle other URLs: test with a HEAD request before starting browser
    logging.info("Testing URL %r with HEAD request", url)

    try:
        requests.head(url, timeout=REQUESTS_TIMEOUT)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        logging.error("Failed to connect to %r", url)
        logging.error("Error message: %s", e)
        return None
    else:
        return url


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Input URL or path")
    parser.add_argument("output", help="Output dir")
    parser.add_argument("--no-readability-js", action="store_true", help="Disable readability.js")
    args = parser.parse_args()

    access_url = url_arg_handler(args.url)

    if access_url is None:
        logging.error("URL failed pre-tests. Exiting...")
        sys.exit(-1)

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
        "privacy.trackingprotection.lower_network_priority": True,
        "privacy.trackingprotection.socialtracking.enabled": True,
    }

    with sync_playwright() as p:
        # Firefox generates simpler accessibility tree than chromium
        # Tested on Debian's firefox-esr 91.5.0esr-1~deb11u1
        browser = p.firefox.launch(firefox_user_prefs=firefox_configs)
        context = browser.new_context(bypass_csp=True)

        def error_cleanup(msg):
            logging.error(msg)
            context.close()
            browser.close()
            sys.exit(-1)

        page = context.new_page()
        page.set_viewport_size({"width": 1080, "height": 1920})
        logging.info("Navigating to %r", access_url)

        # Record HTTP status and navigated URLs so we can check errors later
        url_status = dict()
        navigated_urls = []
        page.on("response", lambda r: url_status.update({r.url: r.status}))
        page.on("framenavigated", lambda f: f.parent_frame is None and navigated_urls.append(f.url))

        page.goto(access_url)

        try:
            page.wait_for_load_state("networkidle")
        except PlaywrightTimeoutError:
            logging.warning("Cannot reach networkidle but will continue")

        # Check HTTP errors
        for url in navigated_urls:
            if (status_code := url_status.get(url, 0)) >= 400:
                error_cleanup(f"Got HTTP error {status_code}")

        # Apply readability.js
        page.evaluate("window.stop()")
        page.add_script_tag(content=get_readability_js())
        readability_info = page.evaluate(r"""(no_readability_js) => {
            window.stop();

            const documentClone = document.cloneNode(true);
            const article = new Readability(documentClone).parse();
            article.applied = false;

            document.querySelectorAll('[aria-hidden=true]').forEach((x) => x.setAttribute("aria-hidden", false));

            if (isProbablyReaderable(document) && !no_readability_js) {
                documentClone.body.innerHTML = article.content;

                if (documentClone.body.innerText.search(/(data|privacy|cookie)\s*(policy|notice)/) >= 0) {
                    document.body.innerHTML = article.content;
                    article.applied = true;
                }
            }

            for (const elem of document.querySelectorAll('script, link, style, header, footer, nav'))
                elem.remove();

            return article;
        }""", [args.no_readability_js])
        cleaned_html = page.content()

        # Check language
        soup = bs4.BeautifulSoup(cleaned_html, 'lxml')
        soup_text = soup.body.text if soup.body else ""

        try:
            lang = langdetect.detect(soup_text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = "UNKNOWN"

        if not lang.lower().startswith("en"):
            error_cleanup(f"Content language {lang} isn't English")

        if re.search(r"(data|privacy)\s*(?:policy|notice)", soup_text, re.I) is None:
            error_cleanup("Not like a privacy policy")

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
        context.close()
        browser.close()


if __name__ == "__main__":
    main()
