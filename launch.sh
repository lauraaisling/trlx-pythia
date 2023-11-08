#!/bin/bash
# sbatch launch.sh
#Resource Request 
#SBATCH --account=eleuther
#SBATCH --job-name=sft-pythia
#SBATCH --output=sft-pythia-70m%x_%j.out   ## filename of the output; the %j is equivalent to jobID; default is slurm-[jobID].out 
#SBATCH --partition=g40x ## the partitions to run in (comma seperated) 

#SBATCH --gpus=8 # number of gpus per task 
#SBATCH --cpus-per-gpu=12 
#SBATCH --nodes=1

#SBATCH --mail-type=ALL

##SBATCH --error=%x_%jerror.out    # Set this dir where you want slurm outs to go
##SBATCH --ntasks=1  ## number of tasks (analyses) to run 
##SBATCH --gpus-per-task=1 # number of gpus per task 
##SBATCH --ntasks-per-node=8

module load cuda/11.7

export HYDRA_FULL_ERROR=1

source /fsx/home-laura/trlx-venv/bin/activate

# accelerate launch --gradient_accumulation_steps 1 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="70m" train.batch_size=8 train.checkpoint_interval=200 train.eval_interval=100 train.run_name="70_adamw_1e-06_zero2" 
# accelerate launch --gradient_accumulation_steps 1 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="160m" train.batch_size=8 train.checkpoint_interval=200 train.eval_interval=100 train.run_name="160m_adamw_1e-06_zero2" 
# accelerate launch --gradient_accumulation_steps 1 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="410m" train.batch_size=8 train.checkpoint_interval=200 train.eval_interval=100 train.run_name="410m_adamw_1e-06_zero2" 
# accelerate launch --gradient_accumulation_steps 1 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="1b" train.batch_size=8 train.checkpoint_interval=200 train.eval_interval=100 train.run_name="1b_adamw_1e-06_zero2" 
##accelerate launch --gradient_accumulation_steps 2 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="1.4b" train.batch_size=4 train.checkpoint_interval=500 train.eval_interval=500 train.run_name="1.4b_adamw_1e-06_zero2" 
##accelerate launch --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="2.8b" train.batch_size=2 train.checkpoint_interval=1000 train.eval_interval=100 train.run_name="2.8b_adamw_1e-06_zero2" # 2 bs * 4 grad_acc * 8 gpus = 128 total_bs AND 96000/16 ~6000 steps

# accelerate launch --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="6.9b" train.batch_size=1 train.run_name="6.9b_adamw_8bit_bnb_1e-06_zero2" train.checkpoint_interval=10000 train.eval_interval=5000 
# accelerate launch --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="12b" train.batch_size=1 train.run_name="12b_adamw_1e-06_zero2" train.checkpoint_interval=10000 train.eval_interval=5000

#test capacity # method.max_new_tokens=40 method.gen_kwargs.top_k=0
# accelerate launch --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="12b" train.batch_size=1 train.run_name="12b_adamw_1e-06_zero2_test" train.checkpoint_interval=100 train.eval_interval=50 train.total_steps=110 

