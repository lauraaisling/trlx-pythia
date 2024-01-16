#!/bin/bash
# sbatch bash_scripts/upload.sh
#Resource Request 
#SBATCH --account=eleuther
#SBATCH --job-name=sft-pythia
#SBATCH --output=%x_%j.out   ## filename of the output; the %j is equivalent to jobID; default is slurm-[jobID].out 
#SBATCH --partition=g40x ## the partitions to run in (comma seperated) 

#SBATCH --nodes=1
#SBATCH --gpus=1 # number of gpus per task 
#SBATCH --cpus-per-gpu=16 

##SBATCH --ntasks=1  ## number of tasks (analyses) to run 
##SBATCH --gpus-per-task=1 # number of gpus per task 
#SBATCH --mail-type=ALL
##SBATCH --ntasks-per-node=8
##SBATCH --error=%x_%jerror.out    # Set this dir where you want slurm outs to go

module load cuda/11.7

export HYDRA_FULL_ERROR=1

source /fsx/home-laura/trlx-venv/bin/activate

# python scripts/upload.py checkpoints/sft_hh/pythia-70m/checkpoint_1504/hf_model lomahony/pythia-70m-helpful-sft main
# python scripts/upload.py checkpoints/sft_hh/pythia-70m/checkpoint_0200/hf_model lomahony/pythia-70m-helpful-sft 200 
# python scripts/upload.py checkpoints/sft_hh/pythia-70m/checkpoint_0400/hf_model lomahony/pythia-70m-helpful-sft 400 
# python scripts/upload.py checkpoints/sft_hh/pythia-70m/checkpoint_0600/hf_model lomahony/pythia-70m-helpful-sft 600 
# python scripts/upload.py checkpoints/sft_hh/pythia-70m/checkpoint_0800/hf_model lomahony/pythia-70m-helpful-sft 800 
# python scripts/upload.py checkpoints/sft_hh/pythia-70m/checkpoint_1000/hf_model lomahony/pythia-70m-helpful-sft 1000 
# python scripts/upload.py checkpoints/sft_hh/pythia-70m/checkpoint_1200/hf_model lomahony/pythia-70m-helpful-sft 1200 
# python scripts/upload.py checkpoints/sft_hh/pythia-70m/checkpoint_1400/hf_model lomahony/pythia-70m-helpful-sft 1400 

# python scripts/upload.py checkpoints/sft_hh/pythia-160m/checkpoint_1504/hf_model lomahony/pythia-160m-helpful-sft main
# python scripts/upload.py checkpoints/sft_hh/pythia-160m/checkpoint_0200/hf_model lomahony/pythia-160m-helpful-sft 200 
# python scripts/upload.py checkpoints/sft_hh/pythia-160m/checkpoint_0400/hf_model lomahony/pythia-160m-helpful-sft 400 
# python scripts/upload.py checkpoints/sft_hh/pythia-160m/checkpoint_0600/hf_model lomahony/pythia-160m-helpful-sft 600 
# python scripts/upload.py checkpoints/sft_hh/pythia-160m/checkpoint_0800/hf_model lomahony/pythia-160m-helpful-sft 800 
# python scripts/upload.py checkpoints/sft_hh/pythia-160m/checkpoint_1000/hf_model lomahony/pythia-160m-helpful-sft 1000 
# python scripts/upload.py checkpoints/sft_hh/pythia-160m/checkpoint_1200/hf_model lomahony/pythia-160m-helpful-sft 1200 
# python scripts/upload.py checkpoints/sft_hh/pythia-160m/checkpoint_1400/hf_model lomahony/pythia-160m-helpful-sft 1400 

# python scripts/upload.py checkpoints/sft_hh/pythia-410m/checkpoint_1504/hf_model lomahony/pythia-410m-helpful-sft main
# python scripts/upload.py checkpoints/sft_hh/pythia-410m/checkpoint_0200/hf_model lomahony/pythia-410m-helpful-sft 200 
# python scripts/upload.py checkpoints/sft_hh/pythia-410m/checkpoint_0400/hf_model lomahony/pythia-410m-helpful-sft 400 
# python scripts/upload.py checkpoints/sft_hh/pythia-410m/checkpoint_0600/hf_model lomahony/pythia-410m-helpful-sft 600 
# python scripts/upload.py checkpoints/sft_hh/pythia-410m/checkpoint_0800/hf_model lomahony/pythia-410m-helpful-sft 800 
# python scripts/upload.py checkpoints/sft_hh/pythia-410m/checkpoint_1000/hf_model lomahony/pythia-410m-helpful-sft 1000 
# python scripts/upload.py checkpoints/sft_hh/pythia-410m/checkpoint_1200/hf_model lomahony/pythia-410m-helpful-sft 1200 
# python scripts/upload.py checkpoints/sft_hh/pythia-410m/checkpoint_1400/hf_model lomahony/pythia-410m-helpful-sft 1400 

# python scripts/upload.py checkpoints/sft_hh/pythia-1b/checkpoint_1504/hf_model lomahony/pythia-1b-helpful-sft main
# python scripts/upload.py checkpoints/sft_hh/pythia-1b/checkpoint_0200/hf_model lomahony/pythia-1b-helpful-sft 200 
# python scripts/upload.py checkpoints/sft_hh/pythia-1b/checkpoint_0400/hf_model lomahony/pythia-1b-helpful-sft 400 
# python scripts/upload.py checkpoints/sft_hh/pythia-1b/checkpoint_0600/hf_model lomahony/pythia-1b-helpful-sft 600 
# python scripts/upload.py checkpoints/sft_hh/pythia-1b/checkpoint_0800/hf_model lomahony/pythia-1b-helpful-sft 800 
# python scripts/upload.py checkpoints/sft_hh/pythia-1b/checkpoint_1000/hf_model lomahony/pythia-1b-helpful-sft 1000 
# python scripts/upload.py checkpoints/sft_hh/pythia-1b/checkpoint_1200/hf_model lomahony/pythia-1b-helpful-sft 1200 
# python scripts/upload.py checkpoints/sft_hh/pythia-1b/checkpoint_1400/hf_model lomahony/pythia-1b-helpful-sft 1400 

python scripts/upload.py checkpoints/sft_hh/pythia-1.4b/checkpoint_3008/hf_model lomahony/pythia-1.4b-helpful-sft main 
python scripts/upload.py checkpoints/sft_hh/pythia-1.4b/checkpoint_0500/hf_model lomahony/pythia-1.4b-helpful-sft 500 
python scripts/upload.py checkpoints/sft_hh/pythia-1.4b/checkpoint_1000/hf_model lomahony/pythia-1.4b-helpful-sft 1000 
python scripts/upload.py checkpoints/sft_hh/pythia-1.4b/checkpoint_1500/hf_model lomahony/pythia-1.4b-helpful-sft 1500 
python scripts/upload.py checkpoints/sft_hh/pythia-1.4b/checkpoint_2000/hf_model lomahony/pythia-1.4b-helpful-sft 2000 
python scripts/upload.py checkpoints/sft_hh/pythia-1.4b/checkpoint_2500/hf_model lomahony/pythia-1.4b-helpful-sft 2500 
python scripts/upload.py checkpoints/sft_hh/pythia-1.4b/checkpoint_3000/hf_model lomahony/pythia-1.4b-helpful-sft 3000 

python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_6016/hf_model lomahony/pythia-2.8b-helpful-sft main
python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_1000/hf_model lomahony/pythia-2.8b-helpful-sft 1000 
python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_2000/hf_model lomahony/pythia-2.8b-helpful-sft 2000 
python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_3000/hf_model lomahony/pythia-2.8b-helpful-sft 3000 
python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_4000/hf_model lomahony/pythia-2.8b-helpful-sft 4000 
python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_5000/hf_model lomahony/pythia-2.8b-helpful-sft 5000 
python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_6000/hf_model lomahony/pythia-2.8b-helpful-sft 6000 
