#!/bin/bash

# 1. Full analysis
python3 ./plotting/plot_result.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/test/exp1/'

# 2. Different sizes
python3 ./plotting/plot_result.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '50000','5000','500' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/test/exp2/'

# 4. 5k default vs selective layer norm vs full layer norm
python3 ./plotting/plot_result.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '5000','5000-layer_norm','5000-selective-layer_norm' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/test/exp4/'

# 5. 5k individual layer re-init default vs ln
python3 ./plotting/plot_result.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '5000' \
  --experiment_name 'only-layer' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/test/exp5/' \
  --ln True

# 6. 5k default vs lr 5x
python3 ./plotting/plot_result.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '5000','5000-lr-5x' \
  --experiment_name 'start-layer' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/test/exp6/'

# 9. block re inits
python3 ./plotting/plot_result.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '5000' \
  --experiment_name 'block-start' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/test/exp9/' \
  --ln True
