#!/bin/bash

# . Scrambling
python3 scrambling_stats.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --method_used 'scrambling' \
  --output_dir '../figures/test/scrambling/' \
  --do_scrambling
