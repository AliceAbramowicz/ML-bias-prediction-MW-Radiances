#! /bin/bash
cd /hpcperm/nld3863
mkdir -p ODB_2023
cd ODB_2023

for yy in 2023; do
	for mm in 02 03 04 05 06; do
		if [ "$mm" = "02" ]; then
			days="09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28"
		elif [ "$mm" = "03" ]; then         
			days="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"
                elif [ "$mm" = "04" ]; then
                        days="18 19 20 21 22 23 24 25 26 27 28 29 30"
		elif [ "$mm" = "05" ]; then
			days="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"
		elif [ "$mm" = "06" ]; then
			days="01 02 03 04 05"
		fi
                
		for dd in $days; do
                        for cy in 00 03 06 09 12 15 18 21; do
                                echo $yy$mm$dd$cy
                                ecp "ec:/nlf/harmonie/4dvar_v22/$yy/$mm/$dd/$cy/odb_stuff.tar" .
				tar -xvf "odb_stuff.tar"
				
				if [ ! -f "VARBC.cycle" ] || [ ! -f "odb_ccma.tar" ]; then
                                        echo "VARBC.cycle or odb_ccma.tar not found, skipping to the next file".
                                        rm -f "odb_can_merge.tar" "odb_can_ori.tar" "odb_can.tar" "odb.tar" "odbvar.tar" "bdstrategy"
                                        continue
                                fi

				rm -f "odb_can_merge.tar" "odb_can_ori.tar" "odb_can.tar" "odb.tar" "odbvar.tar" "bdstrategy"
				mv "odb_ccma.tar" "odb_ccma"$yy$mm$dd$cy".tar"
				mv "VARBC.cycle" "VARBC.cycle."$yy$mm$dd$cy
			 done
		 done
	 done
done
exit

