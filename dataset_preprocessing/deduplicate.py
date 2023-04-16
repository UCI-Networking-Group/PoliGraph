#!/usr/bin/env python3
"""
Deduplicate privacy policies
"""

import argparse
import hashlib
import os
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("input_root", help="Input root dir")
parser.add_argument("dedup_root", help="Dedup root dir containing deduplicated workdirs")
parser.add_argument("symlink_root", help="Copy of input root containing symlinks to dedup root")
args = parser.parse_args()

input_root = Path(args.input_root)
dedup_root = Path(args.dedup_root)
symlink_root = Path(args.symlink_root)

dedup_root.mkdir(exist_ok=True)
symlink_root.mkdir(exist_ok=True)

for html_path in input_root.glob("*/cleaned.html"):
    workdir = html_path.parent

    with html_path.open("rb") as fin:
        checksum = hashlib.blake2s(fin.read()).hexdigest()
        dedup_workdir = dedup_root / checksum
        symlink_workdir = symlink_root / workdir.name

        if not dedup_workdir.is_dir():
            shutil.copytree(workdir, dedup_workdir)

        symlink_workdir.symlink_to(os.path.relpath(dedup_workdir, symlink_root))
