#!/bin/bash

export GLUE_DIR=~/cs224n-project/data/glue

N_TRIALS=3
END=11

for TASK_NAME in 'SST-2' 'CoLA' 'QNLI'
do
  for SEED in $(seq 1 N_TRIALS)
  do
    for START in $(seq 12 -1 0)
    do
      python3 ../examples/run_glue.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir ~/cs224n-project/output/zapping/$TASK_NAME/$SEED/start-layer-$START/ \
        --overwrite_output_dir \
        --save_steps 2000 \
        --start_layer $START \
        --end_layer $END \
        --seed $SEED
    done
  done
done
