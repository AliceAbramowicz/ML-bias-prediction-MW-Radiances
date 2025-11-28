#! /bin/bash
ui="nld3863"
model="AB"
output_dir="/ec/res4/scratch/nld3863/hm_home/${model}_cy43_exp/my_varbc"
mkdir -p $output_dir

for yy in 2021; do
    for mm in 03 04; do
	if [ "$mm" == "03" ]; then
                        days="14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"
        elif [ "$mm" == "04" ]; then
                        days="01 02 03 04 05 06 07 08 09 10 11 12 13 14"
        fi
	for dd in $days; do	
	    for cy in 00 03 06 09 12 15 18 21; do
		    mkdir -p $output_dir/$yy/$mm/$dd/$cy
		    cp /perm/nld3863/FINAL_ML_MODELS/update_VARBC/VarBC_2021/$mm/$dd/$cy/VARBC.cycle_$model $output_dir/$yy/$mm/$dd/$cy/VARBC.cycle
                done
    	done
    done
done
exit
