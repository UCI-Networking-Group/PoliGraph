import argparse
import csv
import datetime as dt
import hashlib
import logging
import math
import os
import pickle
import random
import time

from bs4 import BeautifulSoup
import requests
from requests.exceptions import HTTPError
from werkzeug.urls import url_fix, url_join

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

DATE_RANGE = ("20180101", "20191231")
DATE_REF = "20190101"


def check_meta_redirect(content, base_url):
    soup = BeautifulSoup(content, "lxml")

    for elem in soup.find_all("meta"):
        if elem.get("http-equiv", "").lower() == "refresh":
            try:
                _, url = elem.get("content", "").lower().split(";", 1)
            except ValueError:
                continue

            url = url.strip()

            if url.startswith("url="):
                url = url[4:]

            return url_join(base_url, url)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("flow_csv", help="PoliCheck flow.csv path")
    parser.add_argument("out_dir", help="Output directory")
    args = parser.parse_args()

    all_urls = set()

    # read PoliCheck's flow.csv
    with open(args.flow_csv, newline="") as fin:
        for row in csv.DictReader(fin, fieldnames=list("01234567") + ["policy_url"]):
            all_urls.add(url_fix(row["policy_url"]))

    # shuffle URLs so multiple servers crawl in different orders
    all_urls = list(all_urls)
    random.shuffle(all_urls)

    # Wayback CDX API params
    cdx_params = {
        "url": "",
        "fl": "timestamp",
        "from": DATE_RANGE[0],
        "to": DATE_RANGE[1],
        "collapse": "timestamp:8",  # 1 result per day
        "filter": ["statuscode:200|301|302|-", "mimetype:text/html|warc/revisit"],
    }

    reference_date = dt.datetime.strptime(DATE_REF, "%Y%m%d")
    os.makedirs(args.out_dir, exist_ok=True)
    wait_seconds = 1

    for url in all_urls:
        # use hashed URL as filename
        url_hash = hashlib.blake2s(url.encode()).hexdigest()
        out_path = os.path.join(args.out_dir, f"{url_hash}.pkl")

        if os.path.lexists(out_path):
            continue

        cdx_params["url"] = url
        logging.info("Current URL: %r", url)
        time.sleep(wait_seconds)

        # Wayback CDX API
        try:
            req = requests.get("https://web.archive.org/cdx/search/cdx", params=cdx_params)
            req.raise_for_status()
        except HTTPError:
            runtime_error = req.headers.get("x-archive-wayback-runtime-error", "")
            if not runtime_error.endswith("Blocked Site Error"):
                wait_seconds *= 2
            continue

        # choose the snapshot nearest to the reference_date
        best_offset = math.inf
        best_timestamp = None

        for timestamp_str in req.text.strip().split():
            timestamp = dt.datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            offset = abs((timestamp - reference_date).total_seconds())

            if offset < best_offset:
                best_offset = offset
                best_timestamp = timestamp_str

        redirected_url = url
        redirect_limit = 5

        if best_timestamp is None:
            req = None
        else:
            while redirected_url is not None and redirect_limit > 0:
                # crawl the webpage
                wm_url = f'https://web.archive.org/web/{best_timestamp}id_/{redirected_url}'

                try:
                    req = requests.get(wm_url)

                    if req.status_code == 404:
                        req = None
                        break

                    req.raise_for_status()
                except (HTTPError, ConnectionError):
                    wait_seconds *= 2
                    continue

                if redirected_url := check_meta_redirect(req.content, redirected_url):
                    logging.info("Redirect to: %s", redirected_url)
                    redirect_limit -= 1

        logging.info("Saving to %s ...", out_path)

        # save the whole response
        # because HTTP headers contains more info about the snapshot
        try:
            with open(out_path, "xb") as fout:
                pickle.dump(req, fout)
        except FileExistsError:
            pass

        wait_seconds = 1


if __name__ == "__main__":
    main()

