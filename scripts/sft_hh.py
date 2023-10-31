#### commands to run ####
# accelerate launch --config_file accelerate_config.yaml scripts/sft_hh.py
#########################
# TODO: change checkpoint_dir, model, tokenizer
import json
import sys

from datasets import load_dataset
from ppo_hh import create_reward_fn

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    SFTConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=512, # DPO: 512 1024
        epochs=1, # 100
        total_steps=14000, #
        batch_size=1, # 4/1
        checkpoint_interval=1000, #
        eval_interval=1000,
        pipeline="PromptPipeline",
        trainer="AccelerateSFTTrainer",
        checkpoint_dir="checkpoints/sft_hh/pythia-70m",
        seed=0,
    ),
    model=ModelConfig(model_path="EleutherAI/pythia-70m", num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/pythia-70m", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)), # 0.1 DPO: RMSprop
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=100000000, eta_min=1e-6)),
    method=SFTConfig(
        name="sftconfig",
        gen_kwargs=dict(max_new_tokens=128, top_k=20, top_p=1.0, do_sample=True),
    ),
)


def preprocess(sample):
    sample["chosen_sample"] = sample["prompt"] + sample["chosen"]
    return sample


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    dataset = load_dataset("Dahoas/full-hh-rlhf").map(preprocess)
    reward_fn = create_reward_fn()

    trlx.train(
        config=config,
        samples=dataset["train"]["chosen_sample"],
        eval_prompts=dataset["test"]["prompt"][:280], # 40
        # metric_fn=lambda **kwargs: {"reward": reward_fn(**kwargs)}, # what??? 
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)