#! /bin/bash
ui="nld3863"
EXP="/hpcperm/nld3863/ODB_2023"
output_dir="/hpcperm/nld3863/ccma_dini_2023"


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
                        for cy in 00 03 06 09 12 15 18 21; do
                                echo $yy$mm$dd$cy
				mkdir -p "$output_dir/$yy/$mm/$dd/$cy"
				tar -xvf "$EXP/odb_ccma$yy$mm$dd$cy.tar" -C "$output_dir/$yy/$mm/$dd/$cy"
			done
		done
	done
done
exit
# create different directories per time
