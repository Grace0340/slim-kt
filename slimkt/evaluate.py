"""Evaluation entry point: predictive metrics, cold-start AUC, efficiency, and
interpretability artifacts for a trained checkpoint.

  python -m slimkt.evaluate --config configs/default.yaml \
    --dataset-config configs/dataset/eedi.yaml --set dataset.name=eedi \
    --checkpoint /root/autodl-tmp/slimkt_runs/eedi_sakt_fold0/best.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .config import add_config_args, load_config
from .utils import get_device, get_logger, set_seed

log = get_logger("slimkt.evaluate")


def main() -> None:
    import torch

    from .data.datasets import build_dataloaders
    from .metrics import benchmark_efficiency, kt_metrics
    from .models.factory import build_model
    from .teacher.llm_teacher import TeacherCache

    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()
    cfg = load_config(args.config, args.dataset_config, args.set)
    set_seed(cfg.seed)
    device = get_device()

    train_dl, val_dl, test_dl, stats = build_dataloaders(cfg)
    cfg.model["num_kcs"] = stats.num_kcs

    cache = TeacherCache.load(os.path.join(cfg.paths.teacher_cache, cfg.dataset.name))
    sem_override = cfg.get_dotted("model.sem_override_path", None)
    sem_arr = np.load(sem_override) if sem_override else cache.sem
    teacher_sem = torch.as_tensor(sem_arr, dtype=torch.float32)
    model = build_model(cfg, stats, teacher_sem).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # cold-start positions come straight from the dataset's held-out items,
    # identical to what training excluded (see data.cold_start.select_new_items).
    ys, ps, colds = [], [], []
    with torch.no_grad():
        for batch in test_dl:
            b = {k: v.to(device) for k, v in batch.items()}
            prob = torch.sigmoid(model(b)["logit"]).cpu().numpy()
            mask = batch["mask"].numpy().astype(bool)
            ys.append(batch["r"].numpy()[mask])
            ps.append(prob[mask])
            colds.append(batch["cold"].numpy().astype(bool)[mask])

    y = np.concatenate(ys)
    pr = np.concatenate(ps)
    cold_mask = np.concatenate(colds) if colds else None
    metrics = kt_metrics(y, pr, cold_mask=cold_mask)
    log.info("TEST auc %.4f acc %.4f | cold_auc %s", metrics["auc"], metrics["acc"],
             f'{metrics.get("cold_auc", float("nan")):.4f}')

    # efficiency (H1)
    if cfg.eval.benchmark_efficiency:
        sample = next(iter(test_dl))
        eff = benchmark_efficiency(model, sample, device,
                                   warmup=cfg.eval.latency_warmup, iters=cfg.eval.latency_iters)
        metrics.update(eff)
        log.info("EFFICIENCY %s", {k: round(v, 3) for k, v in eff.items()})

    out_dir = os.path.dirname(args.checkpoint)
    with open(os.path.join(out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # interpretability (H3): dump attention for a few sequences + note LIME hook
    if cfg.eval.interpretability:
        _dump_interpretability(model, test_dl, device, out_dir)
    log.info("eval artifacts saved to %s", out_dir)


def _dump_interpretability(model, loader, device, out_dir):
    import torch

    batch = next(iter(loader))
    b = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = model(b)
    attn = out.get("attn")
    if attn is not None:
        np.save(os.path.join(out_dir, "attention_sample.npy"), attn.cpu().numpy())
    mastery = out.get("mastery")  # baselines may not expose a per-KC mastery head
    if mastery is not None:
        np.save(os.path.join(out_dir, "mastery_sample.npy"), mastery.cpu().numpy())
    # TODO(slim-kt): LIME on the response head treating per-item attributes as
    # interpretable features (lime.lime_tabular). Requires an item-attribute matrix.


if __name__ == "__main__":
    main()
