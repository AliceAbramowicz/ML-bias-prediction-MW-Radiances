#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# make_directories_varbc.sh
#
# Copies VarBC.cycle files into date-organized directories
# using config-provided environment variables.
#
# Inputs (from environment variables):
#   ORIGIN_VARBC_DIR   - path to VarBC source directory
#   DEST_VARBC_DIR     - path to destination directory
#   YEAR               - year 
#   MONTHS             - list of months 
#   CYCLES             - list of cycles 
#   DAYS_JSON          - JSON dict of {month: [days]}
# ============================================================


YEAR=${YEAR:-2021}
MONTHS=${MONTHS}
CYCLES=${CYCLES}
DAYS_JSON=${DAYS_JSON:-"{}"}

echo "Origin: $ORIGIN_VARBC_DIR"
echo "Destination: $DEST_VARBC_DIR"

parse_days() {
python3 - <<'PYCODE'
import json, os
days_json = json.loads(os.environ.get("DAYS_JSON", "{}"))
for mm, days in days_json.items():
    print(mm + ":" + ",".join(str(d).zfill(2) for d in days))
PYCODE
}

days_list=$(parse_days)

for mm in $MONTHS; do
    days=$(echo "$days_list" | grep "^$mm:" | cut -d: -f2 | tr ',' ' ')
    for dd in $days; do
        for cy in $CYCLES; do
            src_file="$ORIGIN_VARBC_DIR/VARBC.cycle.${YEAR}${mm}${dd}${cy}"
            dest_dir="$DEST_VARBC_DIR/$mm/$dd/$cy"
            dest_file="$dest_dir/VARBC.cycle"

            if [[ ! -f "$src_file" ]]; then
                echo "Missing source file: $src_file"
                continue
            fi

            mkdir -p "$dest_dir"
            cp "$src_file" "$dest_file"
            echo "Copied: $src_file to $dest_file"
        done
    done
done


