"""Download XES3G5M from HuggingFace mirrors (works on AutoDL; Drive is blocked).

The authors host XES3G5M on Google Drive, which times out from AutoDL / mainland
China. The community re-hosts the same data on HuggingFace as two datasets that
ARE reachable through the hf-mirror.com endpoint:

  * Atomi/XES3G5M_interaction_sequences   -> full pyKT sequences (train/test, fold, timestamps)
  * Atomi/XES3G5M_content_metadata        -> precomputed RoBERTa embeddings
                                             (question split: 7652 x 768, concept split: 1175)

NOTE: the HF metadata repo ships ONLY the precomputed embeddings, NOT the raw
Chinese question text (content/options/analysis). That raw text lives only in the
Drive package. So this downloader gets you everything needed for the semantic /
cold-start experiments; to additionally run the LLM attribute/option teacher you
must supply metadata/questions.json separately (see --how).

Usage (AutoDL):
  export HF_ENDPOINT=https://hf-mirror.com          # in-China mirror
  pip install -U huggingface_hub
  python scripts/download_xes3g5m.py                # -> /root/autodl-tmp/raw/xes3g5m
  python scripts/download_xes3g5m.py /some/dir
  python scripts/download_xes3g5m.py --how          # how to add raw question text
"""
from __future__ import annotations

import argparse
import os
import sys

INTERACTIONS_REPO = "Atomi/XES3G5M_interaction_sequences"
METADATA_REPO = "Atomi/XES3G5M_content_metadata"

MANUAL = """
=== Adding raw question TEXT (optional; needed only for LLM attribute/option distillation) ===
The HuggingFace mirror has embeddings but NOT the raw Chinese question text.
To unlock the LLM teacher on XES3G5M, obtain metadata/questions.json:

1) On a machine that can reach Google Drive (VPN), open and download:
     https://drive.google.com/file/d/1eFiIYyh5O2V90RA0brammGH6EpHvPDQe/view
2) Extract it and locate:  XES3G5M/metadata/questions.json
3) Upload just that file to AutoDL, e.g.:
     /root/autodl-tmp/raw/xes3g5m/metadata/questions.json
4) Re-run the preprocessor with --questions to merge the text into items.csv:
     python -m slimkt.data.preprocess.xes3g5m --raw /root/autodl-tmp/raw/xes3g5m \\
        --questions /root/autodl-tmp/raw/xes3g5m/metadata/questions.json --out $DATA_ROOT/xes3g5m

Without questions.json the pipeline still runs: it uses the authors' precomputed
question embeddings as the semantic teacher signal (Pillar 1 / cold-start).
"""


def _ensure_mirror() -> None:
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"[xes3g5m] HF_ENDPOINT not set; defaulting to {os.environ['HF_ENDPOINT']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dest", nargs="?", default="/root/autodl-tmp/raw/xes3g5m")
    ap.add_argument("--how", action="store_true", help="print how to add raw question text")
    ap.add_argument("--no-mirror", action="store_true", help="do not force hf-mirror endpoint")
    args = ap.parse_args()

    if args.how:
        print(MANUAL)
        return

    if not args.no_mirror:
        _ensure_mirror()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[xes3g5m] huggingface_hub not installed. Run:  pip install -U huggingface_hub")
        sys.exit(1)

    inter_dir = os.path.join(args.dest, "interaction_sequences")
    meta_dir = os.path.join(args.dest, "content_metadata")
    os.makedirs(inter_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    print(f"[xes3g5m] downloading {INTERACTIONS_REPO} (interactions) ...")
    snapshot_download(repo_id=INTERACTIONS_REPO, repo_type="dataset",
                      local_dir=inter_dir, allow_patterns=["*.parquet", "*.md"])

    print(f"[xes3g5m] downloading {METADATA_REPO} (precomputed embeddings) ...")
    snapshot_download(repo_id=METADATA_REPO, repo_type="dataset",
                      local_dir=meta_dir, allow_patterns=["*.parquet", "*.md"])

    print(f"[xes3g5m] done -> {args.dest}")
    for root, _, files in os.walk(args.dest):
        for f in files:
            if f.endswith(".parquet"):
                print("   ", os.path.join(root, f))
    print("\n[xes3g5m] NOTE: raw question TEXT is NOT included. Run --how to add it.")
    print("[xes3g5m] Next: python -m slimkt.data.preprocess.xes3g5m --raw", args.dest, "--out $DATA_ROOT/xes3g5m")


if __name__ == "__main__":
    main()
