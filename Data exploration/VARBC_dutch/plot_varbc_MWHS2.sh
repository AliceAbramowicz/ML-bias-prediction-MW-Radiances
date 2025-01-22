#!/bin/bash

# plot params evolution over time in the dutch domain. AccordDaTools script plots param0+params, while the script plotvarbccoeff_modified.py plots only params over time

mkdir -p plot_coef_MWHS2_03
cd plot_coef_MWHS2_03
for file in /perm/nld3863/VARBC/varbcout_dutch2021/VARBC_523_73_*_030000; do
	/perm/nld3863/VARBC/plotvarbccoeff_modified.py -i "$file" -b
done

cd /perm/nld3863/VARBC/

mkdir -p plot_coef_MWHS2_12
cd plot_coef_MWHS2_12
for file in /perm/nld3863/VARBC/varbcout_dutch2021/VARBC_523_73_*_120000; do
	/perm/nld3863/VARBC/plotvarbccoeff_modified.py -i "$file" -b
done
