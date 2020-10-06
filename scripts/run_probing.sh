#!/bin/bash

export GLUE_DIR=/mnt/fs2/atamkin/glue_data
OUTPUT_DIR=/mnt/fs2/atamkin/zap_n_tune_outputs

N_TRIALS=3

CUDA_DEVICE=6

for TASK_NAME in 'SST-2' 'CoLA' 'QNLI'
do
  for SEED in $(seq 1 $N_TRIALS)
  do
    for END in $(seq 1 12) # Keep until END layer
    do
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 ../examples/run_glue_probing.py \
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
          --output_dir $OUTPUT_DIR/zapping/$TASK_NAME/probing/$SEED/end-layer-$END/ \
          --overwrite_output_dir \
          --save_steps 2000 \
          --end_layer $END \
          --seed $SEED
    done
  done
done
