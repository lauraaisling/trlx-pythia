#!/bin/bash
# download https://github.com/EleutherAI/lm-evaluation-harness and run in lm-evaluation-harness directory
# sbatch evaluate.sh
#Resource Request 
#SBATCH --account=eleuther
#SBATCH --job-name=sft-pythia
#SBATCH --output=%x_%j.out   ## filename of the output; the %j is equivalent to jobID; default is slurm-[jobID].out 
##SBATCH --error=%x_%jerror.out    # Set this dir where you want slurm outs to go
#SBATCH --partition=g40x ## the partitions to run in (comma seperated) 

#SBATCH --gpus=8 # number of gpus per task 
#SBATCH --cpus-per-gpu=12 
#SBATCH --nodes=1

##SBATCH --ntasks=1  ## number of tasks (analyses) to run 
##SBATCH --gpus-per-task=1 # number of gpus per task 
##SBATCH --mail-type=ALL
##SBATCH --ntasks-per-node=8

module load cuda/11.7

source /fsx/home-laura/lm-evaluation-harness-venv/bin/activate 

accelerate launch main.py --model hf --model_args pretrained=lomahony/pythia-70m-helpful-sft --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path trlx-pythia/model_eval_files/pythia-70m-helpful-0shot > trlx-pythia/model_eval_files/pythia-70m-helpful-0shot-shelloutput.txt 
accelerate launch main.py --model hf --model_args pretrained=lomahony/pythia-70m-helpful-sft --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path trlx-pythia/model_eval_files/pythia-70m-helpful-5shot > trlx-pythia/model_eval_files/pythia-70m-helpful-5shot-shelloutput.txt 
# python main.py --model hf --model_args pretrained=lomahony/pythia-70m-helpful-sft,parallelize=True --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path trlx-pythia/model_eval_files/pythia-70m-helpful-0shot > trlx-pythia/model_eval_files/pythia-70m-helpful-0shot-shelloutput.txt 
# python main.py --model hf --model_args pretrained=lomahony/pythia-70m-helpful-sft,parallelize=True --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path trlx-pythia/model_eval_files/pythia-70m-helpful-5shot > trlx-pythia/model_eval_files/pythia-70m-helpful-5shot-shelloutput.txt 
