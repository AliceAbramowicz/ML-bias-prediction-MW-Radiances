#!/bin/bash

# plot params evolution over time in the dutch domain. 
# AccordDaTools script plots param0+params, while the script plotvarbccoeff_modified.py plots only params over time

mkdir -p plot_coef_MHS_06
cd plot_coef_MHS_06
for file in /perm/nld3863/VARBC_dutch/varbcout_dutch2021/VARBC_*_15_*_060000; do
	/perm/nld3863/VARBC_dutch/plotvarbccoeff_modified.py -i "$file" -b
done

cd /perm/nld3863/VARBC_dutch/

mkdir -p plot_coef_MHS_09
cd plot_coef_MHS_09
for file in /perm/nld3863/VARBC_dutch/varbcout_dutch2021/VARBC_*_15_*_090000; do
        /perm/nld3863/VARBC_dutch/plotvarbccoeff_modified.py -i "$file" -b
done

cd /perm/nld3863/VARBC_dutch/

mkdir -p plot_coef_MHS_18
cd plot_coef_MHS_18
for file in /perm/nld3863/VARBC_dutch/varbcout_dutch2021/VARBC_*_15_*_180000; do
	/perm/nld3863/VARBC_dutch/plotvarbccoeff_modified.py -i "$file" -b
done

cd /perm/nld3863/VARBC_dutch/

mkdir -p plot_coef_MHS_21
cd plot_coef_MHS_21
for file in /perm/nld3863/VARBC_dutch/varbcout_dutch2021/VARBC_*_15_*_210000; do
        /perm/nld3863/VARBC_dutch/plotvarbccoeff_modified.py -i "$file" -b
done
