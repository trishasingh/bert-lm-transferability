#!/bin/bash

# CHANGE THIS TO PATH THAT CONTAINS DATASETS
export GLUE_DIR=/data/home/bestie/cs224n-project/data/glue

PERC=100

for TASK_NAME in 'SST-2' 'CoLA' 'QNLI'
do
    for END in $(seq 1 12)
    do
        python3 ../examples/run_glue_probing.py \
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
          --output_dir /data/home/bestie/cs224n-project/output/probing/$TASK_NAME-$PERC/start-layer-$END/ \
          --overwrite_output_dir \
          --save_steps 2000 \
	  --end_layer $END \
	  --subset_perc $PERC
    done
done
