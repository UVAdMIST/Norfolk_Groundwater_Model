#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH -t 06:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:p100:1
#SBATCH -A uvahydroinformatics
#SBATCH --array=0-19
#SBATCH -o keras_%A_%a.out
#SBATCH -e keras_%A_%a.err

module purge
module load singularity tensorflow/1.6.0-py36

inputdir=/scratch/$USER/mmps153_bootstraps_shifted
# single output dir for all tasks in this array
outputdir=/scratch/$USER/mmps153_results_full_bootstrap_rnn-$SLURM_ARRAY_JOB_ID
mkdir -p $outputdir

fcstoutdir=/scratch/$USER/mmps153_results_full_bootstrap_fcst_rnn-$SLURM_ARRAY_JOB_ID
mkdir -p $fcstoutdir

# get all files
shopt -s nullglob
allfiles=($inputdir/*.csv)

# each task handles a distint group of the csv files
fileindex=$SLURM_ARRAY_TASK_ID
while [[ $fileindex -lt ${#allfiles[@]} ]]; do
    # pass the csv file to be processed as first argument
    file=${allfiles[fileindex]}
    echo "processing file # $fileindex: $file, writing to $outputdir"
    singularity-gpu exec /scratch/$USER/tensorflow-1.6.0-py36.simg python /scratch/$USER/keras_mmps153_fullBS_fcst_rnn_rivanna.py "$file" "$outputdir" "$fcstoutdir" #$count
    fileindex=$((fileindex + $SLURM_ARRAY_TASK_COUNT))
done
