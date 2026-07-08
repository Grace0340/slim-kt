# SLIM-KT — Experiment & Reproducibility Protocol

This document specifies exactly what to run to reproduce the reported results,
in an order that fails fast and cheap.

---

## 0. Research hypotheses (what the experiments must decide)

| ID | Hypothesis | Primary evidence | Section |
|----|-----------|------------------|-----------|
| **H1** | Zero-LLM inference gives DKT-level latency & memory. | latency (ms/interaction), peak GPU MB vs. CLST/LOKT | Results §Efficiency |
| **H2** | Semantic-distilled ID-free items beat ID-based KT on new items. | cold-start AUC, ρ=0.001 AUC vs. DKT/AKT and vs. CLST/LOKT | Results §Cold-Start |
| **H3** | Attribute/option distillation adds accuracy + readable explanations. | ablation deltas, attention/LIME faithfulness | Results §Interpretability + §Ablation |

If **H1 or H2 fails on the first dataset, stop** and revise the method before
scaling to the full matrix.

---

## 1. Datasets

| Dataset | Text? | Options? | Use | Citation key |
|--------|-------|----------|-----|--------------|
| Eedi (NeurIPS'20) | yes | yes (MCQ) | primary (H1–H3) | `wang2020eedi` |
| EdNet | yes | yes | scale check | `choi2020ednet` |
| ASSISTments 2009 | partial | no | classic baseline | `feng2009assistments` |
| ASSISTments 2017 | partial | no | classic baseline | `feng2009assistments` |
| Junyi | partial | no | breadth | **TODO: verified citation** |

**Preprocessing.** Converters produce two files per dataset under
`$DATA_ROOT/<name>/`:

- `interactions.csv`: `uid, order, question_id, kc_id, correct[, option_id]`
- `items.csv`: `question_id, text[, option_0..option_k][, kc_id]`

```bash
# Eedi
python -m slimkt.data.preprocess.eedi  --raw /path/to/eedi_raw       --out ./slimkt_data/eedi
# EdNet-KT1 (cap users while prototyping)
python -m slimkt.data.preprocess.ednet --raw /path/to/EdNet-KT1 \
    --questions /path/to/contents/questions.csv --out ./slimkt_data/ednet --max-users 20000
# always validate before training / teacher extraction
python -m slimkt.data.preprocess.validate --dir ./slimkt_data/eedi
```

> **Item-text caveat (important for the semantic teacher).** Neither Eedi nor
> EdNet releases raw question stems (Eedi ships images; EdNet withholds stems for
> copyright). The converters therefore build `text` as a **metadata proxy**:
> Eedi = subject/construct-name hierarchy; EdNet = `TOEIC Part n` + skill tags.
> This still gives the teacher a real semantic signal for cold start, but is a
> stated limitation. Pass `--question-text CSV(question_id,text)` to use real
> stems if you obtain them (e.g. via OCR). Datasets with no usable text can only
> run the ablation without `L_sem`.

For sequence splits comparable to the pyKT benchmark (`liu2025pykt`), you may
alternatively reuse `pykt-toolkit` preprocessing and then attach `items.csv`
(text/options) separately — pyKT does not carry item text, which is the extra
piece SLIM-KT needs.

**Splits.** 5-fold CV by learner (`num_folds: 5`), report mean ± std. Put fold
files at `$DATA_ROOT/<name>/splits/fold{k}.json`; otherwise a 70/10/20 random
split is used.

---

## 2. Stage 1 — Teacher cache (offline, LLM used once)

```bash
# serve the extraction LLM locally on the AutoDL GPU (recommended, avoids token cost):
python -m vllm.entrypoints.openai.api_server --model Qwen2.5-7B-Instruct --port 8000 &
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY
bash scripts/run_teacher.sh eedi
```

Produces `$DATA_ROOT/teacher_cache/eedi/{teacher_sem,teacher_difficulty,teacher_kc,teacher_options}.npy`
and `teacher_raw.jsonl`. Record: teacher model, prompt version, wall-clock, and
total tokens (report as one-time offline cost, **not** inference cost).

**Open integration items** (`TODO(slim-kt)` in `teacher/llm_teacher.py`):
- pass real `option_0..k` strings into `extract_option_weights`;
- map `required_kcs` names to the dataset's KC vocabulary to fill `teacher_kc`.

---

## 3. Stage 2 — Minimal-viable experiment (do this first)

One dataset (Eedi), backbone AKT/SAKT, verify **H1 + H2** only.

```bash
# warm baseline: ID-based student (turn distillation off, ID on)
bash scripts/run_train.sh eedi --set model.use_id_embedding=true \
     model.lambda_sem=0 model.lambda_attr=0 model.lambda_opt=0
# SLIM-KT: ID-free, full distillation
bash scripts/run_train.sh eedi --set model.use_id_embedding=false
# evaluate both (adds cold-start AUC + latency/memory)
bash scripts/run_eval.sh eedi $OUTPUT_ROOT/eedi_sakt_fold0/best.pt
```

Decision gate: SLIM-KT should (a) match warm AUC within noise, (b) beat the
ID-based model on `cold_auc`, and (c) show latency/memory on par with DKT and
far below any LLM-at-inference baseline. Only then proceed to §4.

---

## 4. Stage 3 — Full matrix

- **Datasets:** all five. **Folds:** 5. **Seeds:** 42, 1, 2 (report mean±std).
- **Baselines:** DKT `piech2015dkt`, DKVMN `zhang2017dkvmn`, AKT `ghosh2020akt`,
  pyKT configs `liu2025pykt`, CLST `jung2024clst`, LOKT `kim2025lokt`. Run DKT/
  DKVMN/AKT via pyKT for fairness; run CLST/LOKT from their released code.
- **Metrics:** AUC, ACC (main); RMSE (secondary); `cold_auc`; latency
  (ms/interaction) and peak GPU MB (H1); interpretability (H3).
- **Cold-start protocol:** `cold_start.new_item_frac=0.2`, sweep
  `rho ∈ {0.0, 0.001, 0.01, 0.1}` (0.0 = fully unseen; 0.001 = extreme few-shot).

### Ablations (isolate each distilled signal)
| Variant | Override |
|---------|----------|
| full SLIM-KT | (default) |
| − distillation | `model.lambda_sem=0 model.lambda_attr=0 model.lambda_opt=0` |
| − attributes | `model.lambda_attr=0` |
| − option weights | `model.lambda_opt=0` |
| + ID (hybrid) | `model.use_id_embedding=true` |
| backbone swap | `model.backbone=akt` (after AKT stub is implemented) |

---

## 5. Compute budget (AutoDL)

- A single 24 GB GPU (e.g. RTX 3090/4090) is enough for the student (it is a
  small deep KT model). Teacher extraction with a 7B model fits on the same card
  via vLLM; a larger teacher may need a 40 GB+ card or API.
- Rough order of magnitude: teacher cache = one pass over the item bank
  (minutes–hours depending on item count); each student run = standard deep-KT
  training (tens of minutes). Full matrix = datasets × folds × seeds × variants —
  script it and log to `$OUTPUT_ROOT`.

---

## 6. Reproducibility checklist

- [ ] seed(s) fixed (`utils.set_seed`), cudnn.benchmark off.
- [ ] teacher model + prompt version + extraction date recorded.
- [ ] exact `items.csv`/`interactions.csv` provenance and preprocessing script.
- [ ] fold files released; splits reproducible.
- [ ] all hyperparameters from `configs/` dumped with each checkpoint (`best.pt`
      stores the resolved config).
- [ ] efficiency measured on a stated GPU, warmup + fixed iters (`metrics.benchmark_efficiency`).
- [ ] one-time offline teacher token/compute cost reported separately from
      inference cost.

---

## 7. Known gaps / extension points (flagged in code)

1. **Junyi** dataset needs a verified citation before use.
2. **AKT backbone**: monotonic distance-decayed attention is implemented in
   `models/backbones.py`; add further attention variants here if needed.
3. **`L_sem`** is a neutral placeholder because semantics enter through the item
   encoder input; add an explicit alignment term only if the encoder is decoupled
   from the teacher (`models/losses.py`).
4. **KC-name → KC-index mapping** for `teacher_kc` and **real option strings**
   for option-weight extraction (`teacher/llm_teacher.py`).
5. **LIME** hook in `evaluate._dump_interpretability` needs an item-attribute
   feature matrix.
