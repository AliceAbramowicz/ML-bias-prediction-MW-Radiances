#!/usr/bin/env bash
set -euo pipefail

# ENV expected from run_pipeline.py:
#   YEAR, HM_DIR, EXP, MONTHS, DAYS_JSON, CYCLES
#   (optional: ODB_DIR, VARNO, SAT_IDS, SENSOR_IDS)

#  default values given here in case they are absent in config:
YEAR=${YEAR}
CCMA_OUTPUT_DIR=${CCMA_OUTPUT_DIR}
HM_DIR=${HM_DIR:-/perm/nld3863/pipeline_datasets}
DATA_DIR=${DATA_DIR:-/perm/nld3863/pipeline_datasets/datasets}
DEST_DIR="$DATA_DIR/sat_${YEAR}_tables"
mkdir -p "$DEST_DIR"

ODB_DIR="odb_ccma/CCMA"
VARNO="119" # 119 == Brightness temperature
OBSTYPE="7" # 7 == SATEM

# Parse the JSON date dictionary
for mm in $MONTHS; do
  for dd in $(python3 -c "import json; d=json.loads('$DAYS_JSON'); print(' '.join(d.get('$mm', [])))"); do
    for cy in $CYCLES; do
      echo "Processing ${YEAR}${mm}${dd}${cy}"
      odb_path="$CCMA_OUTPUT_DIR/$YEAR/$mm/$dd/$cy/$ODB_DIR"
      outfile_final="$DEST_DIR/sat_${YEAR}${mm}${dd}${cy}.txt"

      # Skip processing if the final txt file already exists (-s if you want to skip only when file exists AND not empty)
      if [ -f "$outfile_final" ]; then
        echo "Skipping ${outfile_final} — already exists and non-empty."
        continue
      fi

      if [ ! -d "$odb_path" ]; then
        echo "Directory not found: $odb_path"
        continue
      fi

      cd "$odb_path"

      sqlfile="sql_temp_${YEAR}${mm}${dd}${cy}.sql"
      outfile_temp="temp_${YEAR}${mm}${dd}${cy}.txt"

cat > $sqlfile <<EOF
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
  biascorr_fg,
  obsvalue@body
FROM hdr, body, sat
WHERE (varno@body = $VARNO)
  AND (obstype@hdr = $OBSTYPE)
  AND satellite_identifier@sat IN (3, 5, 209, 223, 523)
  AND sensor@hdr IN (3, 15, 73);
EOF
# Run dcagen and odbsql safely
if ! dcagen; then
  echo "Warning: dcagen failed for ${YEAR}${mm}${dd}${cy}"
  rm -f "$sqlfile"
  continue
fi

dcagen 
# execute SQL query from TempFile
odbsql -v $sqlfile -k -o $outfile_temp
pwd
rm -f $sqlfile
cat $outfile_temp >> $outfile_final
pwd
rm -f $outfile_temp
echo "Finished ${outfile_final}"

		done
	done     
done      

