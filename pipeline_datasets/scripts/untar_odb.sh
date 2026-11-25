#!/usr/bin/env bash
set -euo pipefail

# Extract CCMA tarballs into CCMA_OUTPUT_DIR/YYYY/mm/dd/cy.
#
# Environment (from pipeline_dataset.py):
#   RAW_DATA_DIR, CCMA_OUTPUT_DIR, DATETIMES

: "${RAW_DATA_DIR:?RAW_DATA_DIR is not set}"
: "${CCMA_OUTPUT_DIR:?CCMA_OUTPUT_DIR is not set}"
: "${DATETIMES:?DATETIMES is not set}"

echo "Untarring CCMA files from $RAW_DATA_DIR to $CCMA_OUTPUT_DIR"

n_done=0
n_skipped=0
n_missing=0

for stamp in $DATETIMES; do
  yyyy=${stamp:0:4}
  mm=${stamp:4:2}
  dd=${stamp:6:2}
  cy=${stamp:8:2}

  tarfile="$RAW_DATA_DIR/odb_ccma${stamp}.tar"
  target_dir="$CCMA_OUTPUT_DIR/$yyyy/$mm/$dd/$cy"

  # Skip if already extracted
  if [ -d "$target_dir" ] && [ -n "$(ls -A "$target_dir" 2>/dev/null)" ]; then
    echo "Skipping $tarfile - already extracted in $target_dir"
    n_skipped=$((n_skipped + 1))
    continue
  fi

  if [ ! -f "$tarfile" ]; then
    echo "Missing file: $tarfile"
    n_missing=$((n_missing + 1))
    continue
  fi

  mkdir -p "$target_dir"
  echo "Extracting $tarfile"
  tar -xf "$tarfile" -C "$target_dir"
  n_done=$((n_done + 1))
done

echo "untar_odb: $n_done extracted, $n_skipped already present, $n_missing missing"
