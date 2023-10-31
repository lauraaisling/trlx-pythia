# trlx-pythia
 Supervised finetuning of EleutherAI/pythia-XXX models with helpful part of Anthropic hh [dataset](https://huggingface.co/datasets/Dahoas/full-hh-rlhf) with [TRLx](https://github.com/CarperAI/trlx/tree/main) library. 

Steps: 
1. Create and activate venv 

```bash
python -m venv trlx-env
source ~/trlx-env/bin/activate
``` 

2. Install TRLx as per instructions in TRLx [README](https://github.com/CarperAI/trlx/blob/main/README.md)

```bash
git clone https://github.com/CarperAI/trlx.git
cd trlx
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

3. Clone repo
```bash
git clone https://github.com/lomahony/trlx-pythia.git
``` 

Install additional requirements
```bash
pip install -r requirements.txt

``` 

3. Adjust hyperparameters and run code
```bash
accelerate launch --config_file accelerate_config.yaml scripts/sft_hh.py
``` 
or
```bash
bash run_record.sh
``` 