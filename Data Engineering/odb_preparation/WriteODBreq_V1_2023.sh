#!/bin/bash 
#
##################################################
### User settings:
#
#  OBSTYPE and VARNO should be chosen by the user.
## a complete description can be found in 
### https://apps.ecmwf.int/odbgov/varno/
##
####   some examples:
##   OBSTYPE = 1 <> SYNOP  VARNO = 1 geopiotential (z)
##   OBSTYPE = 2 <> AIREP  VARNO = 2 upper air temperature (t)
##   OBSTYPE = 7 <> SATEM  VARNO = 119 <> brightness temperature (rawbt) 
##   OBSTYPE = 9 <> Scatterometer VARNO = 125 <> ambiguous u component (scatff) 
##   OBSTYPE = 13 <> Ground-based weather radar(14 in ECMWF)  VARNO = 195 <> radar doppler wind (dopp) 
##
SAT="3:5:209:223:523"
SENSOR="3:15:73"
CODETYPE="210"
OBSTYPE="7"
VARNO="119"
HM_DIR="/hpcperm/nld3863"
ODB_DIR="odb_ccma/CCMA"
EXP="ccma_dini_2023"
###########
#
# Tasks written as loop1: FGdep and 4DVminim
#     high resolution (FGdep)  |  low resolution (4DVminim1) lores update_1 
# Tasks written as loop2: 4DVminim and 4DVtraj
#      high resolution (4DVtraj1) hires update_2 | low resolution (4DVminim2) lores update_2 
# Tasks writtenm as loop3 :
#      high resolution (4DVtraj2) hires update_3
##################################################

for yy in 2023; do
        for mm in 02 03 04 05 06; do
                if [ "$mm" = "02" ]; then
                        days="09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28"
                elif [ "$mm" = "03" ]; then
                        days="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"
                elif [ "$mm" = "04" ]; then
                        days="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30"
                elif [ "$mm" = "05" ]; then
                        days="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"
                elif [ "$mm" = "06" ]; then
                        days="01 02 03 04 05"
                fi

                for dd in $days; do
                        for cy in 00 03 06 12 15 18 21; do
                                echo $yy$mm$dd$cy
                                cd $HM_DIR/$EXP/$yy/$mm/$dd/$cy/$ODB_DIR
				pwd

cat >> sql_TempFileLoop1 <<EOF
# I removed the selection of biascorr a posteriori. Keep it in mind if you re-run everything!
select date, time, lat, lon, satellite_identifier@sat,sensor@hdr,vertco_reference_1@body,update_1.offset@hdr, fg_depar, an_depar, biascorr_fg, obsvalue@body FROM hdr,desc,body,errstat,sat WHERE (obstype== $OBSTYPE) AND (fg_depar is not NULL) AND (datum_event1.fg2big@body = 0) AND (varno@body = $VARNO) AND (codetype@hdr = $CODETYPE)
EOF
# EOF is a delimiter (End Of File)
dcagen
# execute SQL query from TempFile
odbsql -v sql_TempFileLoop1 -k -o loop1_$yy$mm$dd$cy
rm -f sql_TempFileLoop1
# concatenate results of SQL query to a file "loop1_OBTY..."
cat loop1_$yy$mm$dd$cy >> $HM_DIR/"loop_DINI_ALL_"$yy$mm$dd$cy".txt"
			done ##cy	
		done ##dd
	done      ###mm
done        ###yy

