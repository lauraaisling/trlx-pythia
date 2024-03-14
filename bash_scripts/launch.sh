#!/bin/bash
# sbatch bash_scripts/launch.sh
#Resource Request 
#SBATCH --account=eleuther
#SBATCH --job-name=sft3-finetune
#SBATCH --output=sft-2.8b-3epochs%x_%j.out   ## filename of the output; the %j is equivalent to jobID; default is slurm-[jobID].out 
#SBATCH --partition=a40x ## the partitions to run in (comma seperated) 

#SBATCH --gpus=8 # number of gpus per task 
#SBATCH --cpus-per-gpu=8 
#SBATCH --nodes=1

#SBATCH --mail-type=ALL

##SBATCH --error=%x_%jerror.out    # Set this dir where you want slurm outs to go
##SBATCH --ntasks=1  ## number of tasks (analyses) to run 
##SBATCH --gpus-per-task=1 # number of gpus per task 
##SBATCH --ntasks-per-node=8

# module load cuda/11.7
module load cuda/12.1

export HYDRA_FULL_ERROR=1
# export HF_HOME='~/admin/home-laura/.cache/huggingface/hub' ##########
# export HF_DATASETS_CACHE='~/admin/home-laura/.cache/huggingface/datasets' ##########

source /admin/home-laura/venvs/trlx310/bin/activate

##INFO ppo test equivalent to 3 epochs launches
## need to change model_path, checkpoint_dir, project_name in config!!! 
# 8 bs * 1 grad_acc * 8 gpus = 64 total_bs AND 96000/64 ~1800 steps - add total steps??? 
# accelerate launch --num_processes 7 --gradient_accumulation_steps 1 --config_file conf/zero2-bf16.yaml scripts/ppo_hh.py model.model_size="1b" train.batch_size=8 train.checkpoint_interval=200 train.eval_interval=100 train.run_name="ppo_1b_adamw_6e-6_3ep_zero2_test" train.total_steps=5200 # 8 bs * 1 grad_acc * 7 gpus = 56 total_bs AND 96200/56 ~5200 steps
# accelerate launch --num_processes 7 --gradient_accumulation_steps 2 --config_file conf/zero2-bf16.yaml scripts/ppo_hh.py model.model_size="1.4b" train.batch_size=4 train.checkpoint_interval=400 train.eval_interval=100 train.run_name="ppo_1.4b_adamw_6e-6_3ep_zero2_test" optimizer.kwargs.lr=6e-6 scheduler.kwargs.eta_min=6e-6 train.total_steps=10400 # 4 bs * 2 grad_acc * 7 gpus = 56 total_bs AND 96000/28 ~3400/epoch 10400 steps
# accelerate launch --num_processes 7 --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/ppo_hh.py model.model_size="2.8b" train.batch_size=1 train.checkpoint_interval=800 train.eval_interval=100 train.run_name="ppo_2.8b_adamw_5.4e-6_3ep_zero2_test" optimizer.kwargs.lr=1e-6 scheduler.kwargs.eta_min=1e-6 train.total_steps=800 # 2 bs * 4 grad_acc * 7 gpus = 56 total_bs AND 96000/14 ~20800 steps
# accelerate launch --num_processes 7 --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/ppo_hh.py model.model_size="2.8b" train.batch_size=1 train.checkpoint_interval=800 train.eval_interval=100 train.run_name="ppo_2.8b_adamw_5.4e-6_3ep_zero2_test" optimizer.kwargs.lr=2e-6 scheduler.kwargs.eta_min=2e-6 train.total_steps=800 # 2 bs * 4 grad_acc * 7 gpus = 56 total_bs AND 96000/14 ~20800 steps
# accelerate launch --num_processes 7 --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/ppo_hh.py model.model_size="2.8b" train.batch_size=1 train.checkpoint_interval=800 train.eval_interval=100 train.run_name="ppo_2.8b_adamw_5.4e-6_3ep_zero2_test" optimizer.kwargs.lr=2.3ee-6 scheduler.kwargs.eta_min=2.3e-6 train.total_steps=800 # 2 bs * 4 grad_acc * 7 gpus = 56 total_bs AND 96000/14 ~20800 steps
# accelerate launch --num_processes 7 --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/ppo_hh.py model.model_size="2.8b" train.batch_size=1 train.checkpoint_interval=800 train.eval_interval=100 train.run_name="ppo_2.8b_adamw_5.4e-6_3ep_zero2_test" optimizer.kwargs.lr=3e-6 scheduler.kwargs.eta_min=3e-6 train.total_steps=800 # 2 bs * 4 grad_acc * 7 gpus = 56 total_bs AND 96000/14 ~20800 steps
# accelerate launch --num_processes 7 --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/ppo_hh.py model.model_size="2.8b" train.batch_size=1 train.checkpoint_interval=800 train.eval_interval=100 train.run_name="ppo_2.8b_adamw_5.4e-6_3ep_zero2_test" optimizer.kwargs.lr=4e-6 scheduler.kwargs.eta_min=4e-6 train.total_steps=800 # 2 bs * 4 grad_acc * 7 gpus = 56 total_bs AND 96000/14 ~20800 steps
# accelerate launch --num_processes 7 --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/ppo_hh.py model.model_size="2.8b" train.batch_size=1 train.checkpoint_interval=800 train.eval_interval=100 train.run_name="ppo_2.8b_adamw_5.4e-6_3ep_zero2_test" optimizer.kwargs.lr=5e-6 scheduler.kwargs.eta_min=5e-6 train.total_steps=800 # 2 bs * 4 grad_acc * 7 gpus = 56 total_bs AND 96000/14 ~20800 steps
# accelerate launch --num_processes 7 --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/ppo_hh.py model.model_size="2.8b" train.batch_size=1 train.checkpoint_interval=800 train.eval_interval=100 train.run_name="ppo_2.8b_adamw_5.4e-6_3ep_zero2_test" optimizer.kwargs.lr=6e-6 scheduler.kwargs.eta_min=6e-6 train.total_steps=800 # 2 bs * 4 grad_acc * 7 gpus = 56 total_bs AND 96000/14 ~20800 steps
# accelerate launch --num_processes 7 --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/ppo_hh.py model.model_size="2.8b" train.batch_size=1 train.checkpoint_interval=800 train.eval_interval=100 train.run_name="ppo_2.8b_adamw_5.4e-6_3ep_zero2_test" optimizer.kwargs.lr=7e-6 scheduler.kwargs.eta_min=7e-6 train.total_steps=800 # 2 bs * 4 grad_acc * 7 gpus = 56 total_bs AND 96000/14 ~20800 steps
# accelerate launch --num_processes 7 --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/ppo_hh.py model.model_size="2.8b" train.batch_size=1 train.checkpoint_interval=800 train.eval_interval=100 train.run_name="ppo_2.8b_adamw_5.4e-6_3ep_zero2_test" optimizer.kwargs.lr=8e-6 scheduler.kwargs.eta_min=8e-6 train.total_steps=800 # 2 bs * 4 grad_acc * 7 gpus = 56 total_bs AND 96000/14 ~20800 steps
# accelerate launch --num_processes 7 --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/ppo_hh.py model.model_size="2.8b" train.batch_size=1 train.checkpoint_interval=800 train.eval_interval=100 train.run_name="ppo_2.8b_adamw_5.4e-6_3ep_zero2_test" optimizer.kwargs.lr=9e-6 scheduler.kwargs.eta_min=9e-6 train.total_steps=800 # 2 bs * 4 grad_acc * 7 gpus = 56 total_bs AND 96000/14 ~20800 steps


# find appropriate hyperhapameters like batchsize and accelerate/zer0 config setting for the larger models
# accelerate launch --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="6.9b" train.batch_size=1 train.run_name="6.9b_adamw_8bit_bnb_1e-06_zero2" train.checkpoint_interval=10000 train.eval_interval=5000 
# accelerate launch --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="12b" train.batch_size=1 train.run_name="12b_adamw_1e-06_zero2" train.checkpoint_interval=10000 train.eval_interval=5000

# test
# accelerate launch --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="12b" train.batch_size=1 train.run_name="12b_adamw_1e-06_zero2_test" train.checkpoint_interval=100 train.eval_interval=50 train.total_steps=110 

##INFO record of sft launches, 
##DO change epochs in script 1 or 3; project_name, checkpoint_dir, model_path (if starting from checkpoint) in config
# accelerate launch --gradient_accumulation_steps 1 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="70m" train.batch_size=8 train.checkpoint_interval=200 train.eval_interval=100 train.run_name="70_adamw_1e-06_3ep_zero2" # 8 bs * 1 grad_acc * 8 gpus = 64 total_bs AND 96000/64 ~1500 steps/epoch
# accelerate launch --gradient_accumulation_steps 1 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="160m" train.batch_size=8 train.checkpoint_interval=200 train.eval_interval=100 train.run_name="160m_adamw_1e-06_3ep_zero2" # 8 bs * 1 grad_acc * 8 gpus = 64 total_bs AND 96000/64 ~1500 steps/epoch
# accelerate launch --gradient_accumulation_steps 1 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="410m" train.batch_size=8 train.checkpoint_interval=200 train.eval_interval=100 train.run_name="410m_adamw_1e-06_3ep_zero2" # 8 bs * 1 grad_acc * 8 gpus = 64 total_bs AND 96000/64 ~1500 steps/epoch
# accelerate launch --gradient_accumulation_steps 1 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="1b" train.batch_size=8 train.checkpoint_interval=200 train.eval_interval=100 train.run_name="1b_adamw_1e-06_3ep_zero2" # 8 bs * 1 grad_acc * 8 gpus = 64 total_bs AND 96000/64 ~1500 steps/epoch
# accelerate launch --gradient_accumulation_steps 2 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="1.4b" train.batch_size=4 train.checkpoint_interval=500 train.eval_interval=500 train.run_name="1.4b_adamw_1e-06_3ep_zero2" # 4 bs * 2 grad_acc * 8 gpus = 64 total_bs AND 96000/32 ~3000 steps/epoch
accelerate launch --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="2.8b" optimizer.kwargs.lr=5e-7 scheduler.kwargs.eta_min=5e-7 train.batch_size=2 train.checkpoint_interval=1000 train.eval_interval=100 train.run_name="2.8b_adamw_5e-07_3ep_zero2" # 2 bs * 4 grad_acc * 8 gpus = 64 total_bs AND 96000/16 ~6000 steps/epoch

##INFO record of sft double launches
## need to change model_path!!! 
# accelerate launch --gradient_accumulation_steps 1 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="70m" train.batch_size=8 train.checkpoint_interval=200 train.eval_interval=100 train.run_name="70_2_adamw_1e-06_zero2" 
# accelerate launch --gradient_accumulation_steps 1 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="160m" train.batch_size=8 train.checkpoint_interval=200 train.eval_interval=100 train.run_name="160m_2_adamw_1e-06_zero2" 
# accelerate launch --gradient_accumulation_steps 1 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="410m" train.batch_size=8 train.checkpoint_interval=200 train.eval_interval=100 train.run_name="410m_2_adamw_1e-06_zero2" 
# accelerate launch --gradient_accumulation_steps 1 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="1b" train.batch_size=8 train.checkpoint_interval=200 train.eval_interval=100 train.run_name="1b_2_adamw_1e-06_zero2" 
# accelerate launch --gradient_accumulation_steps 2 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="1.4b" train.batch_size=4 train.checkpoint_interval=500 train.eval_interval=500 train.run_name="1.4b_2_adamw_1e-06_zero2" 
# accelerate launch --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml scripts/sft_hh.py model.model_size="2.8b" train.batch_size=2 train.checkpoint_interval=1000 train.eval_interval=100 train.run_name="2.8b_2_adamw_1e-06_zero2" 

