#!/bin/bash

export GLUE_DIR=/mnt/fs2/atamkin/glue_data
OUTPUT_DIR=/mnt/fs2/atamkin/zap_n_tune_outputs

N_TRIALS=3
SCRAMBLING_TRIALS=5
END=11
LR_ORIGINAL=2e-5
GAMMA_ORIGINAL=0.02
LR_NEW=1e-4
GAMMA_NEW=1

CUDA_DEVICE=4

# # LayerNorm
# for TASK_NAME in 'SST-2' 'QNLI' 'CoLA'
# do
#   for SIZE in 5000
#   do
#     for SEED in $(seq 1 $N_TRIALS)
#     do
#       for START in $(seq 12 -1 0)
#       do
#         CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 ../examples/run_glue.py \
#           --model_type bert \
#           --model_name_or_path bert-base-uncased \
#           --task_name $TASK_NAME \
#           --do_train \
#           --do_eval \
#           --do_lower_case \
#           --data_dir $GLUE_DIR/$TASK_NAME \
#           --max_seq_length 128 \
#           --per_gpu_eval_batch_size=8   \
#           --per_gpu_train_batch_size=8   \
#           --learning_rate $LR_ORIGINAL \
#           --num_train_epochs 3.0 \
#           --output_dir $OUTPUT_DIR/zapping/$TASK_NAME/$SIZE-layer_norm/$SEED/start-layer-$START/ \
#           --overwrite_output_dir \
#           --save_steps 2000 \
#           --start_layer $START \
#           --end_layer $END \
#           --seed $SEED \
#           --subset_size $SIZE \
#           --do_layer_norm
#       done
#     done
#   done
# done

# LEARNING RATE (5x)
for TASK_NAME in 'SST-2' 'QNLI' 'CoLA'
do
  for SIZE in 5000
  do
    for SEED in $(seq 1 $N_TRIALS)
    do
      for START in $(seq 12 -1 0)
      do
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 ../examples/run_glue_lr.py \
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
          --learning_rate_new $LR_NEW \
          --num_train_epochs 3.0 \
          --output_dir $OUTPUT_DIR/zapping/$TASK_NAME/$SIZE-lr-5x/$SEED/start-layer-$START/ \
          --overwrite_output_dir \
          --save_steps 2000 \
          --start_layer $START \
          --end_layer $END \
          --seed $SEED \
          --subset_size $SIZE
      done
    done
  done
done

# 5000 keep layern norm
for TASK_NAME in 'SST-2' 'QNLI' 'CoLA'
do
  for SIZE in 5000
  do
    for SEED in $(seq 1 $N_TRIALS)
    do
      for START in $(seq 12 -1 0)
      do
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 ../examples/run_glue.py \
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

# scrambling
for TASK_NAME in 'SST-2' 'CoLA' 'QNLI'
do
  for PERM_SEED in $(seq 1 $SCRAMBLING_TRIALS)
  do
      CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 ../examples/run_glue.py \
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
        --output_dir $OUTPUT_DIR/scrambling/$TASK_NAME/$PERM_SEED/ \
        --overwrite_output_dir \
        --permutation_seed $PERM_SEED \
        --save_steps 2000 \
        --seed 1
  done
done
