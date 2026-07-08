# Data availability & reproduction

SLIM-KT is trained and evaluated on **public** knowledge-tracing datasets. This
repository ships **no data** — only the code, configs, and the download /
preprocessing scripts needed to rebuild everything from the original sources.
Raw dumps, derived CSVs, teacher caches, and training runs are all
`.gitignore`d (see `.gitignore`).

Everything below reproduces the three datasets used by SLIM-KT
(XES3G5M, DBE-KT22, Eedi); EdNet-KT1 is optional and not used in the main tables.

---

## TL;DR — one-click download

```bash
# raw dumps -> $RAW_ROOT (default /root/autodl-tmp/raw)
bash scripts/download_all.sh                 # XES3G5M + DBE-KT22 + Eedi
# add EdNet-KT1 as well (optional, large):
bash scripts/download_all.sh --with-ednet

# then build model-ready CSVs -> $DATA_ROOT (default /root/autodl-tmp/slimkt_data)
export DATA_ROOT=/root/autodl-tmp/slimkt_data
python -m slimkt.data.preprocess.eedi     --raw $RAW_ROOT/eedi     --out $DATA_ROOT/eedi
python -m slimkt.data.preprocess.dbe_kt22 --raw $RAW_ROOT/dbe_kt22 --out $DATA_ROOT/dbe_kt22
python -m slimkt.data.preprocess.xes3g5m  --raw $RAW_ROOT/xes3g5m  --out $DATA_ROOT/xes3g5m
```

In-China users: the scripts default to the `https://hf-mirror.com` HuggingFace
mirror. Set `HF_ENDPOINT=https://huggingface.co` to use the official endpoint.

---

## Datasets

### 1. XES3G5M (text-rich, Chinese math)
- **Content.** 7 652 questions, ~5.5 M interactions, 18 066 learners, 798 KCs.
  Ships **precomputed RoBERTa question embeddings** (768-d), which SLIM-KT uses
  as the frozen text-encoder teacher signal.
- **Source.** Community HuggingFace mirror (reachable where Google Drive is not):
  - `Atomi/XES3G5M_interaction_sequences` — pyKT sequences (train/test, folds, timestamps)
  - `Atomi/XES3G5M_content_metadata` — precomputed embeddings (question + concept splits)
- **Download.** `python scripts/download_xes3g5m.py $RAW_ROOT/xes3g5m`
- **Raw question TEXT (optional).** Needed only to run the LLM attribute/option
  teacher (not for the semantic / cold-start experiments). The Chinese stem text
  lives only in the authors' Google Drive package; run
  `python scripts/download_xes3g5m.py --how` for the exact steps to add
  `metadata/questions.json` and re-run the preprocessor with `--questions`.
- **Citation.** Liu et al., *XES3G5M: A Knowledge Tracing Benchmark Dataset with
  Auxiliary Information*, NeurIPS 2023 (Datasets & Benchmarks).

### 2. DBE-KT22 (text-rich, database course)
- **Content.** 212 questions, ~162 K interactions, 1 264 learners, 68 KCs, with
  full question text, multiple-choice options, and an answer key (used for the
  option-recovery / interpretability diagnostics).
- **Source.** HuggingFace dataset `Unggi/dbe-kt22_raw_data` (public mirror of the
  ANU / ADA Dataverse release). Files: `Questions.csv`, `Question_Choices.csv`,
  `KCs.csv`, `Question_KC_Relationships.csv`, `Transaction.csv`, plus metadata.
- **Download.** `python scripts/download_dbe_kt22.py --out $RAW_ROOT/dbe_kt22`
- **Citation.** Abdelrahman et al., *DBE-KT22: A Knowledge Tracing Dataset Based
  on Online Student Evaluation*, 2022 (arXiv:2208.12651).

### 3. Eedi (image-stem, NeurIPS 2020 Education Challenge)
- **Content.** 27 613 questions, ~15.9 M interactions, 118 971 learners. Question
  stems are **images**, so the only text-side signal is a coarse subject-construct
  hierarchy — the deliberate low-resolution boundary case in this study.
- **Source.** Official public Azure blob from the challenge starter kit
  (no login): `https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip`
  → `train_task_1_2.csv` + `{question,answer,student,subject}_metadata*.csv`.
- **Download.** `python scripts/download_eedi.py $RAW_ROOT/eedi` (pure stdlib).
- **Citation.** Wang et al., *Instructions and Guide for Diagnostic Questions:
  The NeurIPS 2020 Education Challenge*, 2020 (arXiv:2007.12061).

### 4. EdNet-KT1 (optional, not in the main tables)
- **Source.** Riiid official links (`github.com/riiid/ednet`): KT1 (~1.2 GB) and
  question contents. The downloader extracts only the first `--max-users` files
  to avoid unpacking 784 K tiny CSVs.
- **Download.** `python scripts/download_ednet.py $RAW_ROOT/ednet --max-users 20000`
  (`--how` prints a HuggingFace-mirror fallback if the bit.ly links are blocked).

---

## Preprocessing → model-ready contract

Each preprocessor (`python -m slimkt.data.preprocess.<dataset>`) converts a raw
dump into a small, uniform set of files under `$DATA_ROOT/<dataset>/`:

| file | purpose |
|------|---------|
| `interactions.csv` | `learner_id, q_idx, correct, timestamp` (contiguous `q_idx`) |
| `items.csv`        | per-item text / options / KC ids (semantic teacher input) |
| `splits/`          | learner-level train/val/test folds + cold-start hold-out |
| `embeddings.npy`   | precomputed text-encoder vectors (XES3G5M ships these) |

Validate a build with:

```bash
python -m slimkt.data.preprocess.validate --data $DATA_ROOT/<dataset>
```

## Teacher cache (offline LLM, optional)

The semantic / cold-start results need only the text-encoder embeddings above.
To additionally reproduce the attribute/option-distillation heads, build the
frozen-LLM teacher cache once (never called at inference):

```bash
bash scripts/run_teacher.sh <dataset>     # -> $DATA_ROOT/<dataset>/teacher_cache/
```

## Directory layout (all git-ignored)

```
$RAW_ROOT/                 # raw downloads (this file's step 1)
  xes3g5m/  dbe_kt22/  eedi/  ednet/
$DATA_ROOT/                # model-ready CSVs (step 2)
  xes3g5m/  dbe_kt22/  eedi/
    interactions.csv  items.csv  splits/  embeddings.npy  teacher_cache/
runs/                      # training outputs / checkpoints / tensorboard
```

## Licensing note

Each dataset retains its **original license and terms of use**; cite the sources
above and consult each provider before redistribution. This repository does not
redistribute any dataset content.
