"""Inspect XES3G5M concept metadata + a sample question's KC routes (structure only)."""
import glob
import json
import os

import pandas as pd

raw = os.environ.get("RAW", "/root/autodl-tmp/raw/xes3g5m")

cpaths = glob.glob(os.path.join(raw, "content_metadata", "**", "concept*.parquet"), recursive=True)
print("=== concept parquet:", cpaths)
if cpaths:
    cdf = pd.read_parquet(cpaths[0])
    print("columns:", list(cdf.columns))
    print("shape:", cdf.shape)
    for c in cdf.columns:
        v = cdf[c].iloc[0]
        s = str(v)
        print(f"  col={c!r} type={type(v).__name__} sample={s[:80]}")
    print("--- first 5 rows of non-embedding cols ---")
    show = [c for c in cdf.columns if "embed" not in c.lower()]
    print(cdf[show].head(5).to_string())

qpath = os.path.join(raw, "metadata", "questions.json")
if os.path.exists(qpath):
    q = json.load(open(qpath, encoding="utf-8"))
    print("\n=== questions.json: n=", len(q))
    k0 = list(q.keys())[0]
    item = q[k0]
    print("sample qid:", k0, "keys:", list(item.keys()))
    for key in ("kc_routes", "type", "options"):
        if key in item:
            print(f"  {key} = {str(item[key])[:200]}")
