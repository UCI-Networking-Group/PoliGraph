import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil

from werkzeug.urls import url_fix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("flow_csv", help="PoliCheck flow.csv path")
    parser.add_argument("in_dir", help="Input directory")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("--symlink", action="store_true", help="Use symlink")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(exist_ok=True)

    with open(args.flow_csv, newline="") as fin:
        seen_app = set()

        for row in csv.DictReader(fin, fieldnames=["app_id"] + list("1234567") + ["policy_url"]):
            url = url_fix(row["policy_url"])
            url_hash = hashlib.blake2s(url.encode()).hexdigest()
            app_id = row["app_id"]

            if app_id in seen_app:
                continue

            seen_app.add(app_id)
            for src_path in in_dir.glob(url_hash + "*"):
                out_name = src_path.name.replace(url_hash, app_id)
                dst_path = out_dir / out_name

                if not dst_path.exists():
                    if args.symlink:
                        os.symlink(src_path.absolute(), dst_path, target_is_directory=True)
                    else:
                        shutil.copy(src_path, dst_path)


if __name__ == "__main__":
    main()
