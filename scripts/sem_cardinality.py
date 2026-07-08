"""Count distinct semantic vectors per dataset (diagnose item-level resolution)."""
import os
import numpy as np

for name in ("eedi", "xes3g5m", "dbe_kt22"):
    p = f"/root/autodl-tmp/slim-kt/data/teacher_cache/{name}/teacher_sem.npy"
    if not os.path.exists(p):
        print(name, "missing")
        continue
    s = np.load(p)
    u = np.unique(np.round(s, 4), axis=0)
    print(f"{name:<9} rows={s.shape[0]:<7} dim={s.shape[1]:<4} distinct={u.shape[0]}")
