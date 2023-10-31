#!/bin/bash
# bash run_record.sh
source ~/trlx-env/bin/activate

accelerate launch --config_file accelerate_config.yaml scripts/sft_hh.py