"""Download the Eedi NeurIPS 2020 Education Challenge dataset (public, no login).

Source: the official public Azure blob used by the challenge starter kit.
  https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip
The zip contains data/train_data/train_task_1_2.csv and data/metadata/{question,
answer,student,subject}_metadata*.csv — everything slimkt.data.preprocess.eedi needs.

Pure stdlib (urllib + zipfile), so no wget/unzip required. Azure is reachable
from AutoDL. Resumable-download note in comments if the connection drops.

Usage:
  python scripts/download_eedi.py                       # -> /root/autodl-tmp/raw/eedi
  python scripts/download_eedi.py /some/other/dir
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
import zipfile

URL = "https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip"


def _progress(count, block_size, total_size):
    done = count * block_size
    if total_size > 0:
        pct = min(100.0, done * 100.0 / total_size)
        sys.stdout.write(f"\r[eedi] downloading {done/1e6:7.1f} MB / {total_size/1e6:7.1f} MB ({pct:5.1f}%)")
    else:
        sys.stdout.write(f"\r[eedi] downloading {done/1e6:7.1f} MB")
    sys.stdout.flush()


def main() -> None:
    dest = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/raw/eedi"
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "data.zip")

    if not (os.path.exists(zip_path) and os.path.getsize(zip_path) > 10_000_000):
        print(f"[eedi] source: {URL}")
        # set a UA; some CDNs reject the default python UA
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0 (slimkt-downloader)")]
        urllib.request.install_opener(opener)
        t0 = time.time()
        urllib.request.urlretrieve(URL, zip_path, _progress)
        print(f"\n[eedi] downloaded in {time.time()-t0:.0f}s -> {zip_path}")
    else:
        print(f"[eedi] reuse existing {zip_path}")

    print("[eedi] extracting ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    print(f"[eedi] done. data under: {os.path.join(dest, 'data')}")
    # show what the preprocessor will find
    for root, _, files in os.walk(os.path.join(dest, "data")):
        for f in files:
            if f.endswith(".csv"):
                print("   ", os.path.join(root, f))


if __name__ == "__main__":
    main()

# If the download drops midway, resume with wget instead:
#   wget -c -O /root/autodl-tmp/raw/eedi/data.zip \
#     https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip
#   python -c "import zipfile; zipfile.ZipFile('/root/autodl-tmp/raw/eedi/data.zip').extractall('/root/autodl-tmp/raw/eedi')"
