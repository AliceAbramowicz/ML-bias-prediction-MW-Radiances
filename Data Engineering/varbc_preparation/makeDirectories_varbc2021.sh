#!/bin/bash
origin="/hpcperm/nld3863/ODB_2021"
dest="/hpcperm/nld3863/VarBC_2021"
cd /hpcperm/nld3863/

if [ ! -d "$origin" ]; then
	echo "Origin folder doesn't exist: $origin"
	exit 1
fi

for yy in 2021; do
        for mm in 03 04; do
                if [ "$mm" = "03" ]; then
                        days="14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"
                elif [ "$mm" = "04" ]; then
                        days="01 02 03 04 05 06 07 08 09 10"
                fi
                for dd in $days; do
                        for cy in 00 03 06 09 12 15 18 21; do
                                echo $yy$mm$dd$cy

				mkdir -p "$dest/$mm/$dd/$cy"
				new_filename="VARBC.cycle"
				mv "$origin/VARBC.cycle.2021${mm}${dd}${cy}" "$dest/$mm/$dd/$cy/$new_filename"
			done
		done
	done
done

