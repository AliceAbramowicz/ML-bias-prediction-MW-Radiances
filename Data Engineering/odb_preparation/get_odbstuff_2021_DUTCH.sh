#! /bin/bash

mkdir -p ODB_2021
cd ODB_2021

for yy in 2021; do
	for mm in 03 04; do
		if [ "$mm" = "03" ]; then         
			days="14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"
                elif [ "$mm" = "04" ]; then
                        days="01 02 03 04 05 06 07 08 09 10"
		fi
                
		for dd in $days; do
                        for cy in 09 21; do
                                echo $yy$mm$dd$cy
				ecp "ec:/nkim/harmonie/knmi43h22tg3_mw_JanBugFix/$yy/$mm/$dd/$cy/odb_stuff.tar" .
				# extract tar file
				tar -xvf "odb_stuff.tar"
				# check for relevant files and remove unnecessary files
				if [ ! -f "VARBC.cycle" ] || [ ! -f "odb_ccma.tar" ]; then
					echo "VARBC.cycle or odb_ccma.tar not found, skipping to the next file".
					rm -f "odb_can_merge.tar" "odb_can_ori.tar" "odb_can.tar" "odb.tar" "odbvar.tar" "bdstrategy"
					continue
				fi
				# remove unnecessary files
				rm -f "odb_can_merge.tar" "odb_can_ori.tar" "odb_can.tar" "odb.tar" "odbvar.tar" "bdstrategy"
				mv "odb_ccma.tar" "odb_ccma"$yy$mm$dd$cy".tar"
				mv "VARBC.cycle" "VARBC.cycle."$yy$mm$dd$cy
				rm -f "odb_stuff.tar" 
			done
		 done
	 done
done
exit

