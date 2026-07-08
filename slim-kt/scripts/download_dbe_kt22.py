"""Download the DBE-KT22 knowledge-tracing dataset (text-rich, ANU database course).

Source: HuggingFace dataset ``Unggi/dbe-kt22_raw_data`` (public mirror of the ADA
Dataverse release). Files: Questions.csv (question text), Question_Choices.csv,
KCs.csv, Question_KC_Relationships.csv, Transaction.csv (interactions), plus KC/
student metadata. Downloads via hf-mirror.

  HF_ENDPOINT=https://hf-mirror.com python scripts/download_dbe_kt22.py \
    --out /root/autodl-tmp/raw/dbe_kt22
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/autodl-tmp/raw/dbe_kt22")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = snapshot_download(
        repo_id="Unggi/dbe-kt22_raw_data",
        repo_type="dataset",
        local_dir=args.out,
    )
    print(f"[dbe_kt22] downloaded to {path}")
    for f in sorted(os.listdir(args.out)):
        fp = os.path.join(args.out, f)
        if os.path.isfile(fp):
            print(f"  {f:<32} {os.path.getsize(fp):>12,} bytes")


if __name__ == "__main__":
    main()
