#!/bin/bash

#SBATCH --job-name=pipeline_dini
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/perm/nld3863/pipeline_no_anchor/outputs/output-%j.out

cd /perm/nld3863/pipeline_no_anchor

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Started:    $(date)"

python3 pipeline_dataset.py --config config/dini.yaml

echo "Finished:   $(date)"

