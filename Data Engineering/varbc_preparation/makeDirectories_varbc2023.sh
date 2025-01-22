#!/bin/bash
origin="/hpcperm/nld3863/ODB_2023"
dest="/hpcperm/nld3863/VarBC_2023"
cd /hpcperm/nld3863

if [ ! -d "$dest" ]; then
	echo "Destination doesn't exist: $dest"
	exit 1
fi
for yy in 2023; do
        for mm in 03 04 05; do
                if [ "$mm" = "03" ]; then
                        days="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"
                elif [ "$mm" = "04" ]; then
                        days="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30"
                elif [ "$mm" = "05" ]; then
                        days="01 02 03 04 05 06 07 08 09 10 11 12 13"
                fi

                for dd in $days; do
                        for cy in 00 03 06 09 12 15 18 21; do
                                echo $yy$mm$dd$cy
				mkdir -p "$dest/$mm/$dd/$cy"
				new_filename="VARBC.cycle"
				mv "$dest/VARBC.cycle.2021${mm}${dd}${cy}" "$dest/$mm/$dd/$cy/$new_filename"
			done
		done
	done
done

