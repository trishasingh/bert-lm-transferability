#!/bin/bash

# Progressive reinit WITH probing
python3 ./plotting/plot_arxiv.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '50000','5000','500','probing-10epoch' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/result/' \
  --plot_vertical

# Scatter plot
python3 ./plotting/plot_arxiv.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '500' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/result/' \
  --plot_scatter

## 5 on Slack list - Probing
#python3 ./plotting/plot_result_final.py \
#  --task_name 'SST-2','QNLI','CoLA' \
#  --subset_size 'probing' \
#  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
#  --output_dir '../figures/result/' \
#  --n_trials 10 \
#  --plot_probing

## 2 on Slack list - Localzied reinit
python3 ./plotting/plot_arxiv.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '5000' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/result/' \
  --experiment_name 'block-start-*-reinit-ln','inv-local-start-*'\
  --plot_block
#,'inv-local-start-*' \

# Progressive reinit with ln parameters kept
python3 ./plotting/plot_arxiv.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '5000','5000-selective-layer_norm' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/result/' \
  --keep_ln

# Progressive reinit with 5x learning rate
python3 ./plotting/plot_arxiv.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '5000-lr-5x','5000-selective-layer_norm' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/result/' \
  --lr_5x

# Individual reinitialization
python3 ./plotting/plot_arxiv.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '5000' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/result/' \
  --experiment_name 'only-layer-*-reinit-ln' \
  --plot_individual

## Alternate reinitialization
#python3 ./plotting/plot_result.py \
#  --task_name 'SST-2','QNLI','CoLA' \
#  --subset_size '5000' \
#  --data_dir '/mnt/fs2/davide3/zap_n_tune_outputs/' \
#  --output_dir '../figures/test/1alt/' \
#  --experiment_name 'alternate-start-*' \
#  --plot_alternate
