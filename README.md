Supervised finetuning of EleutherAI/pythia-XXX models with helpful part of Anthropic hh [dataset](https://huggingface.co/datasets/Dahoas/static-hh) with [TRLx](https://github.com/CarperAI/trlx/tree/main) library. 

Steps: 
1. Create and activate venv 

```bash
python3 -m venv trlx-venv
source trlx-venv/bin/activate
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

3. Adjust hyperparameters, accelerate configs... and run code
```bash
accelerate launch --config_file accelerate_config.yaml scripts/sft_hh.py
/
sbatch shell_scripts/launch.sh
``` 
or
```bash
sbatch shell_scripts/launch.sh
``` 

4. Upload to HuggingFace:
```bash
sbatch shell_scripts/upload.sh
``` 

5. Evals - lm-evaluation harness:
Clone repo - switch to big refactor branch. Create new env and install. Run: 
```bash
sbatch shell_scripts/evaluate.sh
``` 

## Notes:
[wandb runs](https://wandb.ai/lauraomahony999/sft-pythia){:target="_blank"} 

HuggingFace: lomahony/ TODO