"""Quick integrity/quality summary of the XES3G5M teacher cache (stats only)."""
import os
import numpy as np

d = os.environ.get("CACHE_DIR", "/root/autodl-tmp/slim-kt/data/teacher_cache/xes3g5m")
sem = np.load(os.path.join(d, "teacher_sem.npy"))
dif = np.load(os.path.join(d, "teacher_difficulty.npy"))
kc = np.load(os.path.join(d, "teacher_kc.npy"))
opt_path = os.path.join(d, "teacher_options.npy")
opt = np.load(opt_path) if os.path.exists(opt_path) else None

print("sem      :", sem.shape, "dtype", sem.dtype,
      "nonzero_rows", int((np.linalg.norm(sem, axis=1) > 0).sum()))
nan = np.isnan(dif)
valid = dif[~nan]
print("difficulty:", dif.shape, "valid", int((~nan).sum()), "nan", int(nan.sum()),
      "min %.3f mean %.3f max %.3f" % (valid.min(), valid.mean(), valid.max()) if valid.size else "no valid")
print("kc_multi :", kc.shape, "sum", float(kc.sum()), "(0 => KC names not mapped to idx, expected)")
if opt is not None:
    labeled_rows = int((opt >= 0).any(axis=1).sum())
    print("options  :", opt.shape, "rows_with_any_label", labeled_rows,
          "label_values", sorted(set(int(x) for x in np.unique(opt))))
else:
    print("options  : none")
