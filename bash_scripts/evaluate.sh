#!/bin/bash
# download https://github.com/EleutherAI/lm-evaluation-harness and run in lm-evaluation-harness directory
# sbatch bash_scripts/evaluate.sh
#Resource Request 
#SBATCH --account=eleuther
#SBATCH --job-name=sft-pythia
#SBATCH --output=%x_%j.out   ## filename of the output; the %j is equivalent to jobID; default is slurm-[jobID].out 
##SBATCH --error=%x_%jerror.out    # Set this dir where you want slurm outs to go
#SBATCH --partition=a40x ## the partitions to run in (comma seperated) 

#SBATCH --gpus=8 # number of gpus per task 
#SBATCH --cpus-per-gpu=12 
#SBATCH --nodes=1

##SBATCH --ntasks=1  ## number of tasks (analyses) to run 
##SBATCH --gpus-per-task=1 # number of gpus per task 
##SBATCH --mail-type=ALL
##SBATCH --ntasks-per-node=8

# module load cuda/11.7
module load cuda/12.1

# source /fsx/home-laura/lm-evaluation-harness-venv/bin/activate 
source /admin/home-laura/venvs/venv-lm-evaluation-harness/bin/activate


# accelerate launch -m lm_eval --model hf --model_args pretrained=EleutherAI/pythia-70m --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-70m-base-0shot &> files/model_evals/pythia-70m-base-0shot-shelloutput0.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=EleutherAI/pythia-70m --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-70m-base-5shot &> files/model_evals/pythia-70m-base-5shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-70m-helpful-sft --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-70m-helpful-sft-0shot &> files/model_evals/pythia-70m-helpful-sft-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-70m-helpful-sft --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-70m-helpful-sft-5shot &> files/model_evals/pythia-70m-helpful-sft-5shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-70m-helpful-dpo --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-70m-helpful-dpo-0shot &> files/model_evals/pythia-70m-helpful-dpo-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-70m-helpful-dpo --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-70m-helpful-dpo-5shot &> files/model_evals/pythia-70m-helpful-dpo-5shot-shelloutput.txt 

# accelerate launch -m lm_eval --model hf --model_args pretrained=EleutherAI/pythia-160m --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-160m-base-0shot &> files/model_evals/pythia-160m-base-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=EleutherAI/pythia-160m --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-160m-base-5shot &> files/model_evals/pythia-160m-base-5shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-160m-helpful-sft --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-160m-helpful-sft-0shot &> files/model_evals/pythia-160m-helpful-sft-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-160m-helpful-sft --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-160m-helpful-sft-5shot &> files/model_evals/pythia-160m-helpful-sft-5shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-160m-helpful-dpo --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-160m-helpful-dpo-0shot &> files/model_evals/pythia-160m-helpful-dpo-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-160m-helpful-dpo --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-160m-helpful-dpo-5shot &> files/model_evals/pythia-160m-helpful-dpo-5shot-shelloutput.txt 

# accelerate launch -m lm_eval --model hf --model_args pretrained=EleutherAI/pythia-410m --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-410m-base-0shot &> files/model_evals/pythia-410m-base-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=EleutherAI/pythia-410m --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-410m-base-5shot &> files/model_evals/pythia-410m-base-5shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-410m-helpful-sft --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-410m-helpful-sft-0shot &> files/model_evals/pythia-410m-helpful-sft-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-410m-helpful-sft --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-410m-helpful-sft-5shot &> files/model_evals/pythia-410m-helpful-sft-5shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-410m-helpful-dpo --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-410m-helpful-dpo-0shot &> files/model_evals/pythia-410m-helpful-dpo-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-410m-helpful-dpo --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-410m-helpful-dpo-5shot &> files/model_evals/pythia-410m-helpful-dpo-5shot-shelloutput.txt 

# accelerate launch -m lm_eval --model hf --model_args pretrained=EleutherAI/pythia-1b --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-1b-base-0shot &> files/model_evals/pythia-1b-base-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=EleutherAI/pythia-1b --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-1b-base-5shot &> files/model_evals/pythia-1b-base-5shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-1b-helpful-sft --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-1b-helpful-sft-0shot &> files/model_evals/pythia-1b-helpful-sft-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-1b-helpful-sft --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-1b-helpful-sft-5shot &> files/model_evals/pythia-1b-helpful-sft-5shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-1b-helpful-dpo --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-1b-helpful-dpo-0shot &> files/model_evals/pythia-1b-helpful-dpo-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-1b-helpful-dpo --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-1b-helpful-dpo-5shot &> files/model_evals/pythia-1b-helpful-dpo-5shot-shelloutput.txt 

# accelerate launch -m lm_eval --model hf --model_args pretrained=EleutherAI/pythia-1.4b --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-1.4b-base-0shot &> files/model_evals/pythia-1.4b-base-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=EleutherAI/pythia-1.4b --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-1.4b-base-5shot &> files/model_evals/pythia-1.4b-base-5shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-1.4b-helpful-sft --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-1.4b-helpful-sft-0shot &> files/model_evals/pythia-1.4b-helpful-sft-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-1.4b-helpful-sft --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-1.4b-helpful-sft-5shot &> files/model_evals/pythia-1.4b-helpful-sft-5shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-1.4b-helpful-dpo --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-1.4b-helpful-dpo-0shot &> files/model_evals/pythia-1.4b-helpful-dpo-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-1.4b-helpful-dpo --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-1.4b-helpful-dpo-5shot &> files/model_evals/pythia-1.4b-helpful-dpo-5shot-shelloutput.txt 

# accelerate launch -m lm_eval --model hf --model_args pretrained=EleutherAI/pythia-2.8b --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-2.8b-base-0shot &> files/model_evals/pythia-2.8b-base-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=EleutherAI/pythia-2.8b --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-2.8b-base-5shot &> files/model_evals/pythia-2.8b-base-5shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-2.8b-helpful-sft --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-2.8b-helpful-sft-0shot &> files/model_evals/pythia-2.8b-helpful-sft-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-2.8b-helpful-sft --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-2.8b-helpful-sft-5shot &> files/model_evals/pythia-2.8b-helpful-sft-5shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-2.8b-helpful-dpo --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path files/model_evals/pythia-2.8b-helpful-dpo-0shot &> files/model_evals/pythia-2.8b-helpful-dpo-0shot-shelloutput.txt 
# accelerate launch -m lm_eval --model hf --model_args pretrained=lomahony/pythia-2.8b-helpful-dpo --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path files/model_evals/pythia-2.8b-helpful-dpo-5shot &> files/model_evals/pythia-2.8b-helpful-dpo-5shot-shelloutput.txt 


# old
# python main.py --model hf --model_args pretrained=lomahony/pythia-70m-helpful-sft,parallelize=True --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path trlx-pythia/files/model_evals/pythia-70m-helpful-0shot > trlx-pythia/files/model_evals/pythia-70m-helpful-0shot-shelloutput.txt 
# python main.py --model hf --model_args pretrained=lomahony/pythia-70m-helpful-sft,parallelize=True --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path trlx-pythia/files/model_evals/pythia-70m-helpful-5shot > trlx-pythia/files/model_evals/pythia-70m-helpful-5shot-shelloutput.txt 
# accelerate launch main.py --model hf --model_args pretrained=lomahony/pythia-70m-helpful-sft --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 0 --batch_size 16 --output_path trlx-pythia/files/model_evals/pythia-70m-helpful-sft-0shot # > trlx-pythia/files/model_evals/pythia-70m-helpful-sft-0shot-shelloutput.txt 
# accelerate launch main.py --model hf --model_args pretrained=lomahony/pythia-70m-helpful-sft --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,wikitext,winogrande,piqa,boolq,openbookqa,sciq --num_fewshot 5 --batch_size 16 --output_path trlx-pythia/files/model_evals/pythia-70m-helpful-sft-5shot # > trlx-pythia/files/model_evals/pythia-70m-helpful-sft-5shot-shelloutput.txt 