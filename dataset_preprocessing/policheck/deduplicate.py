#!/usr/bin/env python3
"""
Deduplicate privacy policies
"""

import argparse
from pathlib import Path
import hashlib
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("in_dir", help="Input dir")
parser.add_argument("out_dir", help="Output dir")
args = parser.parse_args()

in_dir = Path(args.in_dir)
out_dir = Path(args.out_dir)

out_dir.mkdir(exist_ok=True)

for html_path in in_dir.glob("*/cleaned.html"):
    with html_path.open("rb") as fin:
        checksum = hashlib.blake2s(fin.read()).hexdigest()

        if (out_dir / checksum).is_dir():
            continue

        shutil.copytree(html_path.parent, out_dir / checksum)
