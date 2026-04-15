#!/bin/bash
#$ -N cluster_analysis
#$ -o /groups/ecumming/rrazavi/twitter_project/logs/cluster_analysis_$JOB_ID.out
#$ -e /groups/ecumming/rrazavi/twitter_project/logs/cluster_analysis_$JOB_ID.err
#$ -q long
#$ -l h_vmem=32G
#$ -M rrazavi@nd.edu
#$ -m bea

# Load conda and activate environment
source /software/c/conda/25.9.1/etc/profile.d/conda.sh
conda activate /groups/ecumming/rrazavi/twitter_project/myenv2

# Run clustering analysis
cd /groups/ecumming/rrazavi/twitter_project
python cluster_analysis_8.py
