"""Training entry point for the SLIM-KT student (no LLM in the loop).

  python -m slimkt.train --config configs/default.yaml \
    --dataset-config configs/dataset/eedi.yaml --set dataset.name=eedi
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .config import add_config_args, load_config
from .utils import count_params, get_device, get_logger, set_seed

log = get_logger("slimkt.train")


def _collect_preds(model, loader, device):
    import torch

    model.eval()
    ys, ps, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            prob = torch.sigmoid(out["logit"])
            ys.append(batch["r"].cpu().numpy())
            ps.append(prob.cpu().numpy())
            masks.append(batch["mask"].cpu().numpy())
    y = np.concatenate([a.reshape(-1) for a in ys])
    p = np.concatenate([a.reshape(-1) for a in ps])
    mk = np.concatenate([a.reshape(-1) for a in masks]).astype(bool)
    return y[mk], p[mk]


def main() -> None:
    import torch

    from .data.datasets import build_dataloaders
    from .metrics import kt_metrics
    from .models.factory import build_model
    from .models.losses import SlimKTLoss
    from .teacher.llm_teacher import TeacherCache

    p = argparse.ArgumentParser()
    add_config_args(p)
    args = p.parse_args()
    cfg = load_config(args.config, args.dataset_config, args.set)
    set_seed(cfg.seed)
    device = get_device()
    log.info("device=%s dataset=%s backbone=%s", device, cfg.dataset.name, cfg.model.backbone)

    train_dl, val_dl, test_dl, stats = build_dataloaders(cfg)
    cfg.model["num_kcs"] = stats.num_kcs

    cache_dir = os.path.join(cfg.paths.teacher_cache, cfg.dataset.name)
    teacher = TeacherCache.load(cache_dir)
    # optional: swap in a degraded/alternative semantic table (e.g. the resolution
    # sweep uses K-means-quantized embeddings) without touching the teacher cache.
    sem_override = cfg.get_dotted("model.sem_override_path", None)
    if sem_override:
        sem_arr = np.load(sem_override)
        log.info("semantic embeddings overridden from %s (shape %s)", sem_override, tuple(sem_arr.shape))
    else:
        sem_arr = teacher.sem
    teacher_sem = torch.as_tensor(sem_arr, dtype=torch.float32)

    model = build_model(cfg, stats, teacher_sem).to(device)
    criterion = SlimKTLoss(cfg, teacher).to(device)
    log.info("student trainable params: %s", f"{count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and device == "cuda")

    base = f"{cfg.dataset.name}_{cfg.model.backbone}_fold{cfg.dataset.fold}"
    run_name = cfg.get_dotted("train.run_name", None)
    out_dir = os.path.join(cfg.paths.output_dir, base if not run_name else f"{base}_{run_name}")
    os.makedirs(out_dir, exist_ok=True)
    log.info("output dir: %s", out_dir)
    best_auc, best_epoch, patience = -1.0, -1, 0

    for epoch in range(cfg.train.epochs):
        model.train()
        running = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.train.amp and device == "cuda"):
                out = model(batch)
                losses = criterion(out, batch)
                loss = losses["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach())

        if epoch % cfg.train.eval_every == 0:
            yv, pv = _collect_preds(model, val_dl, device)
            vm = kt_metrics(yv, pv)
            log.info("epoch %d | train_loss %.4f | val_auc %.4f val_acc %.4f",
                     epoch, running / max(len(train_dl), 1), vm["auc"], vm["acc"])
            if vm["auc"] > best_auc:
                best_auc, best_epoch, patience = vm["auc"], epoch, 0
                torch.save({"model": model.state_dict(), "cfg": dict(cfg),
                            "stats_num_kcs": stats.num_kcs}, os.path.join(out_dir, "best.pt"))
            else:
                patience += 1
                if patience >= cfg.train.early_stop_patience:
                    log.info("early stop at epoch %d (best epoch %d, val_auc %.4f)",
                             epoch, best_epoch, best_auc)
                    break

    # final test with the best checkpoint
    ckpt = torch.load(os.path.join(out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    yt, pt = _collect_preds(model, test_dl, device)
    tm = kt_metrics(yt, pt)
    log.info("TEST | auc %.4f acc %.4f rmse %.4f", tm["auc"], tm["acc"], tm["rmse"])
    with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_val_auc": best_auc, "best_epoch": best_epoch, **tm}, f, indent=2)
    log.info("checkpoint + metrics saved to %s", out_dir)


if __name__ == "__main__":
    main()
