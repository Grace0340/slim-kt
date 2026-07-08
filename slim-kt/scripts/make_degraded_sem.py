"""Build resolution-degraded semantic tables for the XES3G5M mechanism study.

For each K, K-means-quantize the item embeddings into K prototypes and replace
every question's vector with its cluster centroid (renormalized to unit norm).
The result has exactly <=K distinct embeddings, so K directly controls the
item-level *semantic resolution* while the dataset, students and splits are held
fixed. Feeding these into sem_cs isolates resolution as the sole causal factor
behind the cold-start gain.

  python scripts/make_degraded_sem.py --cache /root/autodl-tmp/slim-kt/data/teacher_cache/xes3g5m \
    --ks 50 200 800 3200
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="teacher cache dir with teacher_sem.npy")
    ap.add_argument("--ks", type=int, nargs="+", default=[50, 200, 800, 3200])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sem = np.load(os.path.join(args.cache, "teacher_sem.npy"))
    norms = np.linalg.norm(sem, axis=1)
    idx = np.where(norms > 0)[0]                       # non-PAD questions
    X = sem[idx]
    print(f"[degrade] sem {sem.shape}, non-pad rows={len(idx)}")

    for K in args.ks:
        out = np.zeros_like(sem)
        if K >= len(idx):
            out = sem.copy()
        else:
            km = MiniBatchKMeans(n_clusters=K, random_state=args.seed, n_init=3,
                                 batch_size=2048, max_iter=200)
            lab = km.fit_predict(X)
            out[idx] = km.cluster_centers_[lab]
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1.0
        out = (out / n).astype(np.float32)
        distinct = np.unique(np.round(out[idx], 4), axis=0).shape[0]
        path = os.path.join(args.cache, f"sem_k{K}.npy")
        np.save(path, out)
        print(f"[degrade] K={K:<5} distinct={distinct:<5} -> {path}")


if __name__ == "__main__":
    main()
