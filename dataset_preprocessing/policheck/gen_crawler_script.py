#!/usr/bin/env python3

import csv
import hashlib
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
        privacy_policy_id = hashlib.blake2s(privacy_policy_url.encode()).hexdigest()

        if privacy_policy_id in seen_id:
            continue
        else:
            seen_id.add(privacy_policy_id)

        output_dir = os.path.join(target_dir, privacy_policy_id)

        args1 = ['test', '-e', output_dir]
        args2 = [
            'python', '-u',
            '-m', 'privacy_policy_analyzer.scripts.html_crawler',
            privacy_policy_url,
            os.path.join(target_dir, privacy_policy_id)
        ]

        print(shlex.join(args1), "||", shlex.join(args2))
