"""
One-time script to backfill raw_html/ from existing laws.jsonl without re-scraping.

Usage:
    uv run python extract_html_from_jsonl.py [--jsonl laws.jsonl] [--out-dir raw_html]
"""
import argparse
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def extract(jsonl_path: str, out_dir: str) -> None:
    written = skipped = errors = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {lineno}: JSON error — {e}")
                errors += 1
                continue

            url = item.get("url", "")
            html = item.get("full_text", "")

            if not url or not html:
                errors += 1
                continue

            match = re.search(r"/eli/(\d+)/act/(\d+)/", url)
            if not match:
                errors += 1
                continue

            year, number = match.group(1), match.group(2)
            dest_dir = os.path.join(out_dir, year)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, f"act_{number}.html")

            if os.path.exists(dest_path):
                skipped += 1
                continue

            with open(dest_path, "w", encoding="utf-8") as fout:
                fout.write(html)
            written += 1

            if written % 50 == 0:
                print(f"  Written {written} files...")

    print(f"Done: {written} written, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=os.path.join(BASE_DIR, "laws.jsonl"))
    parser.add_argument("--out-dir", default=os.path.join(BASE_DIR, "raw_html"))
    args = parser.parse_args()

    print(f"Reading {args.jsonl} → {args.out_dir}")
    extract(args.jsonl, args.out_dir)
