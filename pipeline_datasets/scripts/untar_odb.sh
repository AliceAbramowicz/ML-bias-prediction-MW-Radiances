#!/usr/bin/env bash
set -euo pipefail

# Inputs from env:
# RAW_DATA_DIR, CCMA_OUTPUT_DIR, EXP

RAW=${RAW_DATA_DIR}
OUT_DIR=${CCMA_OUTPUT_DIR}
YEAR=${YEAR}
EXP=${EXP}

echo "Untarring CCMA files from $RAW to $OUT_DIR"
mkdir -p "$OUT_DIR"

DAYS_JSON=${DAYS_JSON:-"{}"}


for mm in $MONTHS; do
  # Extract days for this month using python (to parse the JSON properly)
  for dd in $(python3 -c "import json; d=json.loads('$DAYS_JSON'); print(' '.join(d.get('$mm', [])))"); do
    for cy in $CYCLES; do
      tarfile="$RAW/odb_ccma$YEAR$mm$dd$cy.tar"      
      target_dir="$OUT_DIR/$YEAR/$mm/$dd/$cy"
      mkdir -p "$target_dir"

      # Skip extraction if target directory already contains extracted ODB files
      if [ -d "$target_dir" ] && [ "$(ls -A "$target_dir" 2>/dev/null)" ]; then
        echo "Skipping $tarfile — already extracted in $target_dir"
        continue
      fi

      if [ -f $tarfile ]; then
        echo "Extracting $tarfile"
        tar -xf $tarfile -C "$OUT_DIR/$YEAR/$mm/$dd/$cy"
      else
        echo "Missing file: $tarfile"
      fi
    done
  done
done