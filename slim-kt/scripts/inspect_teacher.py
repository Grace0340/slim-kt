"""Quick sanity check on an LLM teacher cache: difficulty + option coverage."""
import os
import sys

import numpy as np

cache = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/slim-kt/data/teacher_cache/dbe_kt22"
diff = np.load(os.path.join(cache, "teacher_difficulty.npy"))
kc = np.load(os.path.join(cache, "teacher_kc.npy"))
opt_p = os.path.join(cache, "teacher_options.npy")
sem = np.load(os.path.join(cache, "teacher_sem.npy"))

nz = np.isfinite(diff) & (diff != 0)
print(f"[teacher] {cache}")
print(f"  sem        : shape={sem.shape} nonzero_rows={int((np.linalg.norm(sem,1)>0).sum())}")
print(f"  difficulty : n={diff.size} finite={int(np.isfinite(diff).sum())} "
      f"mean={np.nanmean(diff):.3f} min={np.nanmin(diff):.3f} max={np.nanmax(diff):.3f}")
print(f"  KC multihot: rows_with_kc={int((kc.sum(1)>0).sum())}/{kc.shape[0]} num_kc={kc.shape[1]}")
if os.path.exists(opt_p):
    opt = np.load(opt_p)
    filled = (opt >= 0)
    print(f"  options    : shape={opt.shape} filled_cells={int(filled.sum())} "
          f"rows_with_any={int(filled.any(1).sum())} label_dist={np.bincount(opt[filled].astype(int), minlength=4).tolist()}")
else:
    print("  options    : MISSING")
