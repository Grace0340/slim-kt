"""Download EdNet-KT1 + question contents from the official Riiid links.

Official (github.com/riiid/ednet):
  KT1      : http://bit.ly/ednet_kt1     (~1.2 GB zip, 784k per-user u*.csv)
  Contents : http://bit.ly/ednet-content (questions.csv etc.)

To avoid extracting 784k tiny files, we extract only the first --max-users
`u*.csv` from the KT1 zip (enough for the MVP; raise it later).

⚠️ Network note: the bit.ly links redirect to a cloud host (Dropbox/Drive) that
may be slow or blocked from some AutoDL regions. If this fails, use the reliable
HuggingFace mirror route printed at the end (`--how` for instructions), which
needs a different loader.

Usage:
  python scripts/download_ednet.py                      # -> /root/autodl-tmp/raw/ednet, 20000 users
  python scripts/download_ednet.py /dir --max-users 50000
  python scripts/download_ednet.py --how                # print HF-mirror alternative and exit
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
import zipfile

KT1_URL = "http://bit.ly/ednet_kt1"
CONTENT_URL = "http://bit.ly/ednet-content"

HF_HELP = """
=== EdNet via HuggingFace mirror (reliable on AutoDL) ===
The dataset `mgor/EDNet` hosts kt1 + questions as parquet. On AutoDL:

  export HF_ENDPOINT=https://hf-mirror.com
  pip install -U datasets huggingface_hub
  python - <<'PY'
  from huggingface_hub import snapshot_download
  p = snapshot_download(repo_id="mgor/EDNet", repo_type="dataset")
  print("downloaded to", p)
  PY

NOTE: the HF version is a FLATTENED table (not per-user u*.csv), so it needs a
different loader than slimkt.data.preprocess.ednet. Tell me if you go this route
and I will add an `ednet_hf.py` converter.
"""


def _progress(count, block_size, total_size):
    done = count * block_size
    if total_size > 0:
        pct = min(100.0, done * 100.0 / total_size)
        sys.stdout.write(f"\r  downloading {done/1e6:7.1f}/{total_size/1e6:7.1f} MB ({pct:5.1f}%)")
    else:
        sys.stdout.write(f"\r  downloading {done/1e6:7.1f} MB")
    sys.stdout.flush()


def _download(url: str, path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        print(f"  reuse existing {path}")
        return
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (slimkt-downloader)")]
    urllib.request.install_opener(opener)
    t0 = time.time()
    urllib.request.urlretrieve(url, path, _progress)
    print(f"\n  done in {time.time()-t0:.0f}s -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dest", nargs="?", default="/root/autodl-tmp/raw/ednet")
    ap.add_argument("--max-users", type=int, default=20000,
                    help="extract only the first N u*.csv from the KT1 zip")
    ap.add_argument("--how", action="store_true", help="print the HF-mirror alternative and exit")
    args = ap.parse_args()

    if args.how:
        print(HF_HELP)
        return

    dest = args.dest
    kt1_dir = os.path.join(dest, "KT1")
    os.makedirs(kt1_dir, exist_ok=True)

    try:
        print(f"[ednet] 1/3 contents -> {CONTENT_URL}")
        content_zip = os.path.join(dest, "contents.zip")
        _download(CONTENT_URL, content_zip)
        with zipfile.ZipFile(content_zip) as z:
            z.extractall(dest)

        print(f"[ednet] 2/3 KT1 -> {KT1_URL}")
        kt1_zip = os.path.join(dest, "KT1.zip")
        _download(KT1_URL, kt1_zip)

        print(f"[ednet] 3/3 extracting first {args.max_users} user files ...")
        with zipfile.ZipFile(kt1_zip) as z:
            users = [n for n in z.namelist()
                     if os.path.basename(n).startswith("u") and n.endswith(".csv")]
            users.sort()
            for n in users[: args.max_users]:
                z.extract(n, dest)
        print(f"[ednet] done. KT1 users under {dest} ; find questions.csv under {dest}")
    except Exception as e:  # noqa: BLE001
        print(f"\n[ednet] FAILED via bit.ly ({e}).")
        print(HF_HELP)
        sys.exit(1)


if __name__ == "__main__":
    main()
