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
DEST_DIR="$DATA_DIR/anchors_${YEAR}_tables"
mkdir -p "$DEST_DIR"

VARNO="2" # 119 == Brightness temperature
OBSTYPE="5,2" # 7 == SATEM
ODB_DIR="odb_ccma/CCMA"

# Parse the JSON date dictionary
for mm in $MONTHS; do
  for dd in $(python3 -c "import json; d=json.loads('$DAYS_JSON'); print(' '.join(d.get('$mm', [])))"); do
    for cy in $CYCLES; do
      echo "Processing ${YEAR}${mm}${dd}${cy}"
      odb_path="$CCMA_OUTPUT_DIR/$YEAR/$mm/$dd/$cy/$ODB_DIR"
      outfile_final="$DEST_DIR/anchors_${YEAR}${mm}${dd}${cy}.txt"

      # Skip processing if the final txt file already exists
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

cat > "$sqlfile" <<EOF
select 
        date, 
        time, 
        codetype@hdr, 
        obstype@hdr, 
        varno@body, 
        obsvalue@body, 
        vertco_reference_1@body 
FROM hdr,body 
WHERE obstype@hdr IN (2, 5) AND (varno@body=2)
EOF
# EOF is a delimiter (End Of File)
dcagen
# execute SQL query from TempFile
odbsql -v $sqlfile -k -o $outfile_temp
rm -f $sqlfile
cat $outfile_temp >> $outfile_final
rm -f $outfile_temp
echo "Finished ${outfile_final}"

		done
	done     
done   
