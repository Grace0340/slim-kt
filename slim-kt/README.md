# SLIM-KT

**S**emantic-distilled **L**ightweight **I**nterpretable **M**odel for
**K**nowledge **T**racing.

> **Idea in one line.** A frozen text encoder (and, optionally, a frozen LLM)
> labels every item **once, offline** — a semantic embedding plus structured
> difficulty / knowledge-concept / option-adequacy signals. These are distilled
> into a small SAKT/AKT student. At inference the teacher is gone: the student
> runs at deep-KT cost, is cold-start robust (represents new items by semantics,
> not IDs), and exposes concept-level interpretable heads.

- **Data & one-click download:** see [`DATA.md`](DATA.md).
- **Full protocol** (splits, metrics, ablations, compute budget):
  [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).
- **License:** MIT (see [`LICENSE`](LICENSE)).

---

## Repository layout

```
slimkt/                       # the Python package
  config.py                   # YAML config dataclasses
  utils.py                    # seeding, logging, device, timers
  metrics.py                  # AUC / ACC / RMSE, cold-start AUC, latency & memory
  train.py                    # training entry point (config-driven)
  evaluate.py                 # evaluation / efficiency benchmarking
  data/
    datasets.py               # pyKT-style sequence dataset + dataloaders
    cold_start.py             # cold-start / few-shot (rho) hold-out splitting
    preprocess/               # raw dump -> uniform CSV contract
      xes3g5m.py  dbe_kt22.py  eedi.py  ednet.py  make_synthetic.py
      validate.py  common.py
  models/
    backbones.py              # SAKT + monotonic-attention AKT (both real)
    student.py                # semantic item encoder + backbone + mastery/option heads
    losses.py                 # L_KT, L_sem, L_attr, L_opt + combined loss
    baselines.py  factory.py  # DKT / DKVMN / AKT / SAKT-ID baselines + builder
  teacher/
    prompts.py                # LLM prompt templates (attributes, option weights)
    llm_teacher.py            # frozen-LLM extraction (embeddings / attrs / opts)
configs/                      # default.yaml + one override per dataset
scripts/                      # download, preprocess, train, analysis, plotting
docs/                         # EXPERIMENT_PROTOCOL.md (reproducibility protocol)
results/                      # small reproducible JSON/CSV that back the result tables
requirements.txt  pyproject.toml  DATA.md  LICENSE
```

Everything data-shaped (raw dumps, derived CSVs, `teacher_cache/`, `runs/`,
checkpoints, logs) is `.gitignore`d and regenerated from public sources.

## Install

```bash
git clone <this-repo> slim-kt && cd slim-kt
# On a GPU box, install torch matching your CUDA FIRST (see scripts/autodl_setup.sh),
# then:
pip install -r requirements.txt
pip install -e .          # exposes the `slimkt` package
```

## Reproduce (5 stages)

```bash
export RAW_ROOT=/root/autodl-tmp/raw
export DATA_ROOT=/root/autodl-tmp/slimkt_data

# 1) DATA — download + preprocess public datasets (see DATA.md for details)
bash scripts/download_all.sh
python -m slimkt.data.preprocess.eedi     --raw $RAW_ROOT/eedi     --out $DATA_ROOT/eedi
python -m slimkt.data.preprocess.dbe_kt22 --raw $RAW_ROOT/dbe_kt22 --out $DATA_ROOT/dbe_kt22
python -m slimkt.data.preprocess.xes3g5m  --raw $RAW_ROOT/xes3g5m  --out $DATA_ROOT/xes3g5m

# 2) TEACHER (optional; only for the attribute/option heads — the LLM is offline)
bash scripts/run_teacher.sh xes3g5m

# 3) TRAIN the student (no LLM in the loop)
bash scripts/run_core.sh              # main headline runs (SLIM-KT vs SAKT-ID, 3 seeds)

# 4) EVALUATE — AUC/ACC, cold-start AUC (rho=0.001), latency & GPU memory
bash scripts/run_eval.sh xes3g5m

# 5) ANALYSIS — regenerate the supplementary experiment tables
bash scripts/run_supplementary.sh     # bootstrap + backbone + option recovery + folds
python scripts/export_table_s1.py     # -> results/table_s1.csv (supplementary experiments)
```

## Reproducing specific results

| Result | Script | Output |
|---|---|---|
| Main + cold-start performance | `scripts/run_core.sh`, `scripts/run_eedi_core.sh` | `runs/…/metrics.json` |
| Semantic-resolution sweep | `scripts/run_resolution_xes.sh` → `summarize_resolution.py` | `results/…` |
| Paired-bootstrap significance | `scripts/bootstrap_significance.py` | `results/*_bootstrap_*.json` |
| AKT + semantic backbone | `scripts/_run_sem_akt.sh` | `runs/…` |
| Option recovery (interpretability) | `scripts/option_recovery.py` | stdout / `results/` |
| 5-fold cross-validation | `scripts/_run_folds.sh` → `agg_folds.py` | `results/folds_summary.json` |
| Teacher preprocessing cost | `scripts/teacher_cost.py` | `results/teacher_cost.json` |

## Datasets

Three public datasets are used — **XES3G5M** and **DBE-KT22** (text-rich) and
**Eedi** (image-stem, coarse-semantics boundary case) — plus optional EdNet-KT1.
Sources, licenses, and download/preprocess commands are in [`DATA.md`](DATA.md).
No dataset content is redistributed here.

## License

Released under the MIT License (see [`LICENSE`](LICENSE)).
