#!/bin/bash
# sbatch /fsx/home-laura/trlx-pythia/trlx-pythia/launch.sh
#Resource Request 
#SBATCH --account=eleuther
#SBATCH --job-name=sft-pythia
#SBATCH --output=%x_%j.out   ## filename of the output; the %j is equivalent to jobID; default is slurm-[jobID].out 
##SBATCH --error=%x_%jerror.out    # Set this dir where you want slurm outs to go
#SBATCH --partition=g40x ## the partitions to run in (comma seperated) 

#SBATCH --gpus=1 # number of gpus per task 
#SBATCH --cpus-per-gpu=12 
#SBATCH --nodes=1

##SBATCH --ntasks=1  ## number of tasks (analyses) to run 
##SBATCH --gpus-per-task=1 # number of gpus per task 
##SBATCH --mail-type=ALL
##SBATCH --ntasks-per-node=8

module load cuda/11.7

export HYDRA_FULL_ERROR=1

source /fsx/home-laura/trlx-venv/bin/activate

python upload.py .checkpoints/sft_hh/pythia-70m/checkpoint_430/hf_model lomahony/pythia-70m-helpful-sft main
# python upload.py .checkpoints/sft_hh/pythia-70m/checkpoint_430/hf_model lomahony/pythia-70m-helpful-sft checkpoints
