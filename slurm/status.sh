#!/bin/bash
# One-shot status for the exp28-31 verification campaign, with each run's key metric
# compared against its earlier-experiment reference.
#
# Usage:  bash slurm/status.sh
#
# Reads ONLY local job logs (no wandb, no network). Safe to run any time; changes nothing.
# The references below are the same ones in each group's README -- see those for WHY some
# comparisons are invalid (notably exp30-vs-exp24 mIoU, which differs by metric definition).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

hdr () { printf "\n\033[1m%s\033[0m\n" "$1"; }
# NB: metrics are scraped across ALL of a run's logs (grep -h over *.log) and reduced with
# sort -g, never from a single log picked by glob order -- `ls *.log | tail -1` sorts
# ALPHABETICALLY (job-X_9 outranks job-X_48) and silently yields mid-array values.
# Also: extract values with `sed "s/.*: //"`, not `grep -oE "[0-9.]+"` -- key names like
# `top1` contain a digit and would be captured as the value.

hdr "QUEUE"
r=$(squeue -u "$USER" -h -t RUNNING -r 2>/dev/null | grep -c .)
p=$(squeue -u "$USER" -h -t PENDING -r 2>/dev/null | grep -c .)
echo "  running=$r  pending=$p"
bad=$(sacct -u "$USER" -S now-14days -n -P -o JobID,State -X 2>/dev/null \
      | awk -F'|' '$1 ~ /^151133(5[89]|6[0-9]|7[0-9]|80)/ && $2 !~ /PENDING|RUNNING|COMPLETED/ {print "  "$1" "$2}')
[ -n "$bad" ] && { echo "  NON-CLEAN TERMINAL STATES:"; echo "$bad"; } || echo "  no failed/cancelled/timeout tasks"

hdr "exp28 — pretraining (ref = exp22 train_loss @ step 8192)"
printf "  %-30s %-10s %-12s %-10s %s\n" ARM STEP LOSS "REF@8192" DELTA@8192
for pair in exp28-uniform16:1.8945 exp28-uniform16-teacherinit:1.6106 exp28-fovi:1.8885 exp28-fovi-teacherinit:1.8377; do
  arm=${pair%%:*}; ref=${pair##*:}; d=logs/exp28_pretrain_lrdrop/$arm
  [ -d "$d" ] || { printf "  %-30s not started\n" "$arm"; continue; }
  last=$(grep -hoE "step [0-9]+ +loss=[0-9.]+" "$d"/log/*.log 2>/dev/null | tail -1)
  st=$(echo "$last" | awk '{print $2}'); ls_=$(echo "$last" | grep -oE "[0-9.]+$")
  at8=$(grep -hoE "step 8192 +loss=[0-9.]+" "$d"/log/*.log 2>/dev/null | head -1 | grep -oE "[0-9.]+$")
  dl=$([ -n "$at8" ] && awk -v a="$at8" -v b="$ref" 'BEGIN{printf "%+.4f", a-b}' || echo "-")
  printf "  %-30s %-10s %-12s %-10s %s\n" "$arm" "${st:-?}" "${ls_:-?}" "$ref" "${at8:+$at8 ($dl)}${at8:-not yet}"
done
echo "  phase B (x0.1 drop) is SEPARATE and gated on step-<N>.pt existing:"
for a in exp28-fovi-teacherinit:1130496 exp28-uniform16:1441792 exp28-uniform16-teacherinit:630784; do
  arm=${a%%:*}; s=${a##*:}
  [ -f "logs/exp28_pretrain_lrdrop/$arm/checkpoints/step-$s.pt" ] \
    && echo "    READY: bash slurm/runs/exp28_pretrain_lrdrop/$arm-lrdrop.sh" \
    || echo "    not ready: $arm needs step-$s.pt"
done

hdr "exp29 — in1k finetune (ref = exp25 BEST eval/top1)"
printf "  %-24s %-10s %-14s %-10s %s\n" RUN STEP BEST_TOP1 REF NOTE
for pair in in1k-uni16ti-803k:0.84954: in1k-fovi-ti-1196k:0.83692:ref_incomplete_320k_of_401408 in1k-uni16-1516k:0.83522: in1k-fovi-1901k:none:new_arm_no_reference; do
  IFS=: read -r run ref note <<<"$pair"; d=logs/exp29_in1k_finetune/$run
  [ -d "$d" ] || { printf "  %-24s not started\n" "$run"; continue; }
  st=$(grep -hoE "step [0-9]+ +loss=" "$d"/log/*.log 2>/dev/null | tail -1 | awk '{print $2}')
  b=$(grep -hoE "'top1': [0-9.]+" "$d"/log/*.log 2>/dev/null | sed "s/.*: //" | sort -g | tail -1)
  [ -n "$b" ] && b=$(printf "%.5f" "$b")
  printf "  %-24s %-10s %-14s %-10s %s\n" "$run" "${st:-?}" "${b:-none yet}" "$ref" "$note"
done

hdr "exp30 — ade20k probe (ref = exp24 miou_final; NOT directly comparable)"
echo "  exp24 predates 68b635f (mIoU reduction-order fix) -> exp30 reads ~+0.2pp HIGHER by"
echo "  metric definition, not improvement. Judge ORDERING: uni16ti > fovi-ti > uni16."
printf "  %-24s %-10s %-14s %s\n" RUN STEP BEST_MIOU "REF(stale basis)"
for pair in ade20k-uni16ti-803k:0.44479 ade20k-fovi-ti-1196k:0.43997 ade20k-uni16-1516k:0.42321 ade20k-fovi-1901k:none; do
  run=${pair%%:*}; ref=${pair##*:}; d=logs/exp30_ade20k_probe/$run
  [ -d "$d" ] || { printf "  %-24s not started\n" "$run"; continue; }
  st=$(grep -hoE "step [0-9]+ +eval" "$d"/log/*.log 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1)
  b=$(grep -hoE "'miou_final': [0-9.]+" "$d"/log/*.log 2>/dev/null | sed "s/.*: //" | sort -g | tail -1)
  [ -n "$b" ] && b=$(printf "%.5f" "$b")
  printf "  %-24s %-10s %-14s %s\n" "$run" "${st:-?}" "${b:-none yet}" "$ref"
done

hdr "exp31 — policy qreg, 10 seeds (ref = exp27 lossfix; qband 0.6853+-0.0007)"
echo "  A seed BEATING the band is evidence of a broken protocol, not success."
printf "  %-28s %-10s %-12s %-12s %s\n" SEED STEP BEST_CE_MEAN MIOU_FINAL IN_BAND
for s in 0 1 2 3 4 5 6 7 8 9; do
  d=logs/exp31_policy_qreg_10seed/exp31-policy-qreg-s$s
  [ -d "$d" ] || { printf "  %-28s not started\n" "s$s"; continue; }
  st=$(grep -hoE "step [0-9]+ +eval" "$d"/log/*.log 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1)
  ce=$(grep -hoE "'ce_mean': [0-9.]+" "$d"/log/*.log 2>/dev/null | sed "s/.*: //" | sort -g | head -1)
  mi=$(grep -hoE "'miou_final': [0-9.]+" "$d"/log/*.log 2>/dev/null | sed "s/.*: //" | sort -g | tail -1)
  [ -n "$ce" ] && ce=$(printf "%.5f" "$ce"); [ -n "$mi" ] && mi=$(printf "%.5f" "$mi")
  band="-"
  [ -n "$ce" ] && band=$(awk -v c="$ce" 'BEGIN{print (c>=0.6846 && c<=0.6860) ? "yes" : (c<0.6846 ? "BELOW - SUSPECT" : "above")}')
  printf "  %-28s %-10s %-12s %-12s %s\n" "s$s" "${st:-?}" "${ce:-none yet}" "${mi:-none yet}" "$band"
done
echo
echo "References + why some comparisons are invalid: slurm/runs/<group>/README.md"
