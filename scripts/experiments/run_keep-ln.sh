#!/bin/bash

export GLUE_DIR=/mnt/fs2/atamkin/glue_data
OUTPUT_DIR=/mnt/fs2/atamkin/zap_n_tune_outputs

N_TRIALS=3
END=11

# Select GPU
CUDA_DEVICE=6

# 5000 keep layern norm
for TASK_NAME in 'SST-2' 'QNLI' 'CoLA'
do
  for SIZE in 5000
  do
    for SEED in $(seq 1 $N_TRIALS)
    do
      for START in $(seq 12 -1 0)
      do
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 ../../examples/run_glue.py \
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
          --learning_rate $LR_ORIGINAL \
          --num_train_epochs 3.0 \
          --output_dir $OUTPUT_DIR/zapping/$TASK_NAME/$SIZE-keep-ln/$SEED/start-layer-$START/ \
          --overwrite_output_dir \
          --save_steps 2000 \
          --start_layer $START \
          --end_layer $END \
          --seed $SEED \
          --subset_size $SIZE \
          --keep_layer_norm
      done
    done
  done
done