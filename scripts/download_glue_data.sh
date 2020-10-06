#!/bin/bash

export GLUE_DIR=~/cs224n-project/data/glue

for TASK_NAME in 'CoLA' 'SST' 'QNLI'
do
  python3 ~/cs224n-project/download_glue_data.py --data_dir $GLUE_DIR --tasks $TASK_NAME
done