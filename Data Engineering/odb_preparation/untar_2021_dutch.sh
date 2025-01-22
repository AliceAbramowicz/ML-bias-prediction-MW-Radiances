#! /bin/bash
ui="nld3863"
EXP="/hpcperm/nld3863/ODB_2021"
output_dir="/hpcperm/nld3863/ccma_dutch_2021"

# create different directories per time

for yy in 2021; do
	for mm in 03 04; do
		if [ "$mm" == "03" ]; then
                        days="14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"
                elif [ "$mm" == "04" ]; then
                        days="01 02 03 04 05 06 07 08 09 10"
                fi
		for dd in $days; do	
			for cy in 00 03 06 09 12 15 18 21; do
				mkdir -p "$output_dir/$yy/$mm/$dd/$cy"
				tar -xvf "$EXP/odb_ccma$yy$mm$dd$cy.tar" -C "$output_dir/$yy/$mm/$dd/$cy"
			done
		done
	done
done
exit
