#!/usr/bin/env python3

from urllib.parse import urlparse
import csv
from base64 import urlsafe_b64encode
import os
import shlex
import sys


policheck_csv_columns = ['package_name', 'app_name', 'version_name', 'version_code', 'data_type',
                         'dest_domain', 'dest_ip', 'arb_number', 'privacy_policy']

input_csv, target_dir = sys.argv[1:]
seen_id = set()

with open(input_csv, newline="") as fin:
    for row in csv.DictReader(fin, fieldnames=policheck_csv_columns):
        privacy_policy_url = row["privacy_policy"]

        parsed_url = urlparse(row["privacy_policy"])

        if parsed_url.scheme not in ["http", "https"]:
            continue

        privacy_policy_id = urlsafe_b64encode(privacy_policy_url.encode()).decode()

        if privacy_policy_id in seen_id:
            continue
        else:
            seen_id.add(privacy_policy_id)

        output_dir = os.path.join(target_dir, privacy_policy_id)

        args1 = ['test', '-e', output_dir]
        args2 = [
            'python', '-u',
            '-m', 'poligrapher.scripts.html_crawler',
            privacy_policy_url,
            os.path.join(target_dir, privacy_policy_id)
        ]

        print(shlex.join(args1), "||", shlex.join(args2))
