import os
import numpy as np
import pandas as pd

cache = "/root/autodl-tmp/slim-kt/data/teacher_cache/dbe_kt22"
raw = "/root/autodl-tmp/raw/dbe_kt22"

opt = np.load(os.path.join(cache, "teacher_options.npy"))
print("teacher_options shape:", opt.shape, "dtype:", opt.dtype)
valid = opt[opt >= 0]
print("valid cells:", valid.size, "label dist:", np.bincount(valid.astype(int), minlength=4).tolist())
print("rows with any label:", int((opt >= 0).any(1).sum()), "/", opt.shape[0])
print("rows fully missing:", int((opt < 0).all(1).sum()))

# per-row: how many have a unique argmax vs ties
rows_valid = np.where((opt >= 0).any(1))[0]
ties = 0
for i in rows_valid:
    row = opt[i]
    mx = row.max()
    if (row == mx).sum() > 1:
        ties += 1
print("rows with tied max:", ties, "/", len(rows_valid))

# check raw answer key alignment: does teacher_raw.jsonl exist?
print("\n--- files in cache ---")
for f in sorted(os.listdir(cache)):
    print(" ", f)

# Inspect raw choices ordering
ch = pd.read_csv(os.path.join(raw, "Question_Choices.csv"))
print("\nQuestion_Choices columns:", ch.columns.tolist())
print("sample:\n", ch.head(8).to_string())
# distribution of correct-option position within sorted-by-id groups
pos = []
for qid, g in ch.sort_values("id").groupby("question_id"):
    for j, (_, row) in enumerate(g.iterrows()):
        if bool(row.get("is_correct", False)):
            pos.append(j); break
print("\ncorrect-option position dist (sorted by id):", np.bincount(pos).tolist(), "n=", len(pos))
