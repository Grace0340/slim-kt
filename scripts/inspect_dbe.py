"""Inspect DBE-KT22 raw tables (columns + samples) to plan preprocessing."""
import os

import pandas as pd

raw = os.environ.get("RAW", "/root/autodl-tmp/raw/dbe_kt22")
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)

for fn in ("Transaction.csv", "Questions.csv", "Question_Choices.csv",
           "KCs.csv", "Question_KC_Relationships.csv"):
    p = os.path.join(raw, fn)
    if not os.path.exists(p):
        print(fn, "MISSING\n")
        continue
    df = pd.read_csv(p)
    print(f"===== {fn}  shape={df.shape} =====")
    print("columns:", list(df.columns))
    print(df.head(3).to_string())
    print()

# quick keys/dtypes on Transaction for interaction fields
t = pd.read_csv(os.path.join(raw, "Transaction.csv"))
print("=== Transaction dtypes ===")
print(t.dtypes)
print("n_students:", t.iloc[:, 0].nunique() if t.shape[1] else "?")
