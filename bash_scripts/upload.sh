#!/bin/bash
# sbatch bash_scripts/upload.sh
#Resource Request 
#SBATCH --account=eleuther
#SBATCH --job-name=upload
#SBATCH --output=%x_%j.out   ## filename of the output; the %j is equivalent to jobID; default is slurm-[jobID].out 
#SBATCH --partition=a40x ## the partitions to run in (comma seperated) 

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

# source /admin/home-laura/trlx-venv/bin/activate
source /admin/home-laura/venvs/trlx310/bin/activate


# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_18048/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs main
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_01000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs 1000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_02000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs 2000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_03000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs 3000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_04000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs 4000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_05000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs 5000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_06000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs epoch1-6000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_07000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs 7000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_08000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs 8000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_09000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs 9000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_10000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs 10000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_11000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs 11000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_12000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs epoch2-12000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_13000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs 13000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_14000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs 14000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_15000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs 15000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_16000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs 16000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_17000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs 17000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_18000/hf_model lomahony/pythia-2.8b-helpful-sft-3epochs 18000


# python scripts/upload.py checkpoints/sft_hh/pythia-1b/checkpoint_1504/hf_model lomahony/pythia-1b-helpful-sft main
# python scripts/upload.py checkpoints/sft_hh/pythia-1b/checkpoint_0200/hf_model lomahony/pythia-1b-helpful-sft 200 
# python scripts/upload.py checkpoints/sft_hh/pythia-1b/checkpoint_0400/hf_model lomahony/pythia-1b-helpful-sft 400 
# python scripts/upload.py checkpoints/sft_hh/pythia-1b/checkpoint_0600/hf_model lomahony/pythia-1b-helpful-sft 600 
# python scripts/upload.py checkpoints/sft_hh/pythia-1b/checkpoint_0800/hf_model lomahony/pythia-1b-helpful-sft 800 
# python scripts/upload.py checkpoints/sft_hh/pythia-1b/checkpoint_1000/hf_model lomahony/pythia-1b-helpful-sft 1000 
# python scripts/upload.py checkpoints/sft_hh/pythia-1b/checkpoint_1200/hf_model lomahony/pythia-1b-helpful-sft 1200 
# python scripts/upload.py checkpoints/sft_hh/pythia-1b/checkpoint_1400/hf_model lomahony/pythia-1b-helpful-sft 1400 

# python scripts/upload.py checkpoints/sft_hh/pythia-1.4b/checkpoint_3008/hf_model lomahony/pythia-1.4b-helpful-sft main 
# python scripts/upload.py checkpoints/sft_hh/pythia-1.4b/checkpoint_0500/hf_model lomahony/pythia-1.4b-helpful-sft 500 
# python scripts/upload.py checkpoints/sft_hh/pythia-1.4b/checkpoint_1000/hf_model lomahony/pythia-1.4b-helpful-sft 1000 
# python scripts/upload.py checkpoints/sft_hh/pythia-1.4b/checkpoint_1500/hf_model lomahony/pythia-1.4b-helpful-sft 1500 
# python scripts/upload.py checkpoints/sft_hh/pythia-1.4b/checkpoint_2000/hf_model lomahony/pythia-1.4b-helpful-sft 2000 
# python scripts/upload.py checkpoints/sft_hh/pythia-1.4b/checkpoint_2500/hf_model lomahony/pythia-1.4b-helpful-sft 2500 
# python scripts/upload.py checkpoints/sft_hh/pythia-1.4b/checkpoint_3000/hf_model lomahony/pythia-1.4b-helpful-sft 3000 

# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_6016/hf_model lomahony/pythia-2.8b-helpful-sft main
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_1000/hf_model lomahony/pythia-2.8b-helpful-sft 1000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_2000/hf_model lomahony/pythia-2.8b-helpful-sft 2000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_3000/hf_model lomahony/pythia-2.8b-helpful-sft 3000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_4000/hf_model lomahony/pythia-2.8b-helpful-sft 4000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_5000/hf_model lomahony/pythia-2.8b-helpful-sft 5000 
# python scripts/upload.py checkpoints/sft_hh/pythia-2.8b/checkpoint_6000/hf_model lomahony/pythia-2.8b-helpful-sft 6000 

# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_00400/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 400 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_00800/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 800 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_01200/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 1200 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_01600/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 1600 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_02000/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 2000 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_02400/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 2400 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_02800/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 2800 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_03200/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 3200 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_03600/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs epoch1-3600 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_04000/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 4000 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_04400/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 4400 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_04800/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 4800 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_05200/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 5200 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_05600/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 5600 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_06000/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 6000 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_06400/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 6400 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_06800/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs epoch2-6800 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_07200/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 7200 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_07600/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 7600 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_08000/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 8000 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_08400/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 8400 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_08800/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 8800 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_09200/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 9200 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_09600/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 9600 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_10000/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs 10000 
# python scripts/upload.py checkpoints/ppo_hh/pythia-1.4b/checkpoint_10400/hf_model lomahony/pythia-1.4b-helpful-sfted1-ppo-3epochs main

