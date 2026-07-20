#!/bin/bash
set -e
for mode in clip gp_standard gp_domain_aware; do
  echo "=== START $mode $(date) ==="
  t0=$(date +%s)
  CRITIC_CONSTRAINT=$mode python3 gen_gan.py > "run_log_${mode}.txt" 2>&1
  t1=$(date +%s)
  echo "=== DONE $mode in $((t1-t0))s ==="
done
echo "ALL DONE"
