#!/bin/bash
#$ -N emb_model2
#$ -o /groups/ecumming/rrazavi/twitter_project/logs/model2_$JOB_ID.out
#$ -e /groups/ecumming/rrazavi/twitter_project/logs/model2_$JOB_ID.err
#$ -q long
#$ -l h_vmem=48G
#$ -l h_rt=48:00:00
#$ -M rrazavi@nd.edu
#$ -m bea

# Load conda and activate environment
source /software/c/conda/25.9.1/etc/profile.d/conda.sh
conda activate /groups/ecumming/rrazavi/twitter_project/myenv2

# Run embedding script
cd /groups/ecumming/rrazavi/twitter_project
python get_embeddings_model2.py
