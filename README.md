## SFT & RLHF finetuning Pythia with TRLx library

Code and instructions to perform Supervised and RLHF (PPO) finetuning of EleutherAI/pythia-XXX models with helpful part of Anthropic hh [dataset](https://huggingface.co/datasets/Dahoas/static-hh) with [TRLx](https://github.com/CarperAI/trlx/tree/main) library. 

Steps: 
1. Create and activate venv 

```bash
python3.10 -m venv trlx-pythia
source trlx-pythia/bin/activate
``` 

2. Install TRLx as per instructions in TRLx [README](https://github.com/CarperAI/trlx/blob/main/README.md)

```bash
mkdir trlx-pythia-project
git clone https://github.com/CarperAI/trlx.git
cd trlx
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
pip install -e .
cd ..
```

3. Clone repo
```bash
git clone https://github.com/lomahony/trlx-pythia.git
``` 

Install additional requirements
```bash
cd trlx-pythia
pip install -r requirements.txt
``` 
set up wandb tracker
```bash
wandb login --relogin
``` 

3. Adjust hyperparameters, accelerate configs... and run code as in bash_scripts/launch.sh

4. Upload to HuggingFace as in:
```bash
sbatch bash_scripts/upload.sh
``` 

5. Evals - lm-evaluation harness:
Clone repo - switch to big refactor branch. Create new env and install. Run bash_scripts/evaluate.sh

## Notes:
[sft wandb runs](https://wandb.ai/lauraomahony999/pythia-sft){:target="_blank"}, [dpo wandb runs](https://wandb.ai/lauraomahony999/pythia-dpo){:target="_blank"} 

Models available on HuggingFace: https://huggingface.co/lomahony 
