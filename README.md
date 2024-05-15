## SFT & RLHF finetuning Pythia with TRLx library

Code and instructions to perform Supervised and RLHF (PPO) finetuning of EleutherAI/pythia-XXX models with helpful part of Anthropic hh [dataset](https://huggingface.co/datasets/Dahoas/static-hh) with [TRLx](https://github.com/CarperAI/trlx/tree/main) library. 

More details about the HuggingFace pythia-helpful-1epoch [suite](https://huggingface.co/collections/lomahony/pythia-helpful-1epoch-65f0eb6b906e39ea4b3b1956), such as hyperparameters is available in the paper [Attributing Mode Collapse in the Fine-Tuning of Large Language Models](https://openreview.net/forum?id=3pDMYjpOxk)

DPO model training code is in [https://github.com/lauraaisling/direct-preference-optimization](https://github.com/lauraaisling/direct-preference-optimization)

Steps: 
1. Set up

Create and activate venv, whatever name - (trlx310). Tested python 3.10

```bash
python3.10 -m venv trlx-pythia
source trlx-pythia/bin/activate
``` 

Install TRLx as per instructions in TRLx [README](https://github.com/CarperAI/trlx/blob/main/README.md)

```bash
mkdir trlx-pythia-project
git clone https://github.com/CarperAI/trlx.git
cd trlx
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
pip install -e .
cd ..
```

Clone repo
```bash
git clone https://github.com/lomahony/trlx-pythia.git
``` 

Install additional requirements (more details in some_commands.txt)
```bash
cd trlx-pythia
pip install -r requirements.txt
``` 
set up wandb tracker
```bash
wandb login --relogin
``` 

2. Adjust hyperparameters, accelerate configs... and run code as in bash_scripts/launch.sh

3. Upload to HuggingFace as in:
```bash
sbatch bash_scripts/upload.sh
``` 

4. Evals - lm-evaluation harness:
Clone repo - switch to big refactor branch. Create new env and install. 
```bash
sbatch bash_scripts/evaluate.sh
``` 

## Notes:
[sft wandb runs](https://wandb.ai/lauraomahony999/pythia-sft){:target="_blank"}, [dpo wandb runs](https://wandb.ai/lauraomahony999/pythia-dpo){:target="_blank"} 

Models available on HuggingFace: https://huggingface.co/lomahony 

If you find the code or models useful, please feel consider citing us: 
<pre>
@inproceedings{o2024attributing,
  title={Attributing Mode Collapse in the Fine-Tuning of Large Language Models},
  author={O’Mahony, Laura and Grinsztajn, Leo and Schoelkopf, Hailey and Biderman, Stella},
  booktitle={ICLR 2024, Mathematical and Empirical Understanding of Foundation Models (ME-FoMo) workshop},
  year={2024}
}
</pre>

## Contact

For questions, please contact Laura O'Mahony - lauraaisling.ml@gmail.com
