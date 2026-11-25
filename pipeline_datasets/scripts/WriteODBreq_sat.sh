#!/usr/bin/env bash
set -euo pipefail

# Extract satellite data from the CCMAs into SAT_TABLES_DIR/sat_$YYYYMMDDHH.txt
# Environment from pipeline_dataset.py:
# CCMA_OUTPUT_DIR, SAT_TABLES_DIR, DATETIMES, SENSORS, SATELLITES

: "${CCMA_OUTPUT_DIR:?CCMA_OUTPUT_DIR is not set}"
: "${SAT_TABLES_DIR:?SAT_TABLES_DIR is not set}"
: "${DATETIMES:?DATETIMES is not set}"
: "${SENSORS:?SENSORS is not set}"
: "${SATELLITES:?SATELLITES is not set}"

VARNO=119          # 119 == brightness temperature
OBSTYPE=7          # 7 == SATEM
ODB_SUBDIR="odb_ccma/CCMA"

n_done=0
n_skipped=0
n_failed=0

for stamp in $DATETIMES; do
  yyyy=${stamp:0:4}
  mm=${stamp:4:2}
  dd=${stamp:6:2}
  cy=${stamp:8:2}

  odb_path="$CCMA_OUTPUT_DIR/$yyyy/$mm/$dd/$cy/$ODB_SUBDIR"
  outfile="$SAT_TABLES_DIR/sat_${stamp}.txt"
  tmpfile="$outfile.tmp"

  # empty files will still be re-run and take time when pipeline is rerun, even when this script was already fully run
  if [ -s "$outfile" ]; then
    echo "Skipping $outfile - already exists."
    n_skipped=$((n_skipped + 1))
    continue
  fi

  if [ ! -d "$odb_path" ]; then
    echo "Directory not found: $odb_path"
    n_skipped=$((n_skipped + 1))
    continue
  fi

  echo "Processing $stamp"
  rm -f "$tmpfile"


  (
    cd "$odb_path"
    sqlfile="sql_sat_${stamp}.sql"
    trap 'rm -f "$sqlfile"' EXIT

    cat > "$sqlfile" <<EOF
SELECT
  date,
  time,
  lat,
  lon,
  satellite_identifier@sat,
  sensor@hdr,
  vertco_reference_1@body,
  fg_depar,
  an_depar,
  biascorr_fg@body,
  biascorr@body,
  obsvalue@body,
  zenith@sat,
  azimuth@sat
FROM hdr, body, sat
WHERE (varno@body = $VARNO)
  AND (obstype@hdr = $OBSTYPE)
  AND satellite_identifier@sat IN ($SATELLITES)
  AND sensor@hdr IN ($SENSORS);
EOF

    dcagen
    odbsql -v "$sqlfile" -k -o "$tmpfile"
  ) || {
    echo "WARNING: extraction failed for $stamp"
    rm -f "$tmpfile"
    n_failed=$((n_failed + 1))
    continue
  }

  mv "$tmpfile" "$outfile"
  echo "Finished $outfile"
  n_done=$((n_done + 1))
done

echo "WriteODBreq_sat: $n_done written, $n_skipped skipped, $n_failed failed"
