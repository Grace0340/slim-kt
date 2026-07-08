#!/usr/bin/env bash
cd /root/autodl-tmp/slim-kt
echo "=== all runs ==="
ls runs
echo
echo "=== checkpoint presence (sem_cs / id_cs / sem_akt) ==="
for ds in xes3g5m dbe_kt22 eedi; do
  for s in 42 1 7; do
    for v in sem_cs id_cs sem_akt; do
      d="runs/${ds}_sakt_fold0_${v}_s${s}"
      if [ -f "$d/best.pt" ]; then st="best.pt OK"; else st="MISSING"; fi
      ev="no-eval"; [ -f "$d/eval_metrics.json" ] && ev="eval OK"
      printf "%-45s %-12s %s\n" "$d" "$st" "$ev"
    done
  done
done
echo
echo "=== teacher_options present? ==="
for ds in xes3g5m dbe_kt22 eedi synth; do
  echo -n "$ds: "; ls data/teacher_cache/$ds/teacher_options.npy 2>/dev/null || echo "no options"
done
echo
echo "=== raw dbe? ==="
ls /root/autodl-tmp/raw 2>/dev/null
