#!/bin/bash

# FIGURE 1
# Progressive reinitialization and probing 
python3 ./plotting/plot_arxiv.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '50000','5000','500','probing-10epoch' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/result/' \
  --plot_vertical

# FIGURE 2
# Localized reinitialization (with blocks reinitialized and preserved)
python3 ./plotting/plot_arxiv.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '5000' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/result/' \
  --experiment_name 'block-start-*-reinit-ln','inv-local-start-*'\
  --plot_block

# FIGURE 3
# Changing order of pretrained layers

# FIGURE 4
# Scatter plot of partial reinitialization for 500 examples
python3 ./plotting/plot_arxiv.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '500' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/result/' \
  --plot_scatter

# FIGURE 5
# Localized reinitialization with individual layers
python3 ./plotting/plot_arxiv.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '5000' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/result/' \
  --experiment_name 'only-layer-*-reinit-ln' \
  --plot_individual

# FIGURE 6
# Progressive reinitialization with 5x larger learning rate
python3 ./plotting/plot_arxiv.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '5000-lr-5x','5000-selective-layer_norm' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/result/' \
  --lr_5x

# FIGURE 7
# Progressive reinitialization with layer norm parameters kept
python3 ./plotting/plot_arxiv.py \
  --task_name 'SST-2','QNLI','CoLA' \
  --subset_size '5000','5000-selective-layer_norm' \
  --data_dir '/mnt/fs2/atamkin/zap_n_tune_outputs/' \
  --output_dir '../figures/result/' \
  --keep_ln

## Alternate reinitialization
#python3 ./plotting/plot_result.py \
#  --task_name 'SST-2','QNLI','CoLA' \
#  --subset_size '5000' \
#  --data_dir '/mnt/fs2/davide3/zap_n_tune_outputs/' \
#  --output_dir '../figures/test/1alt/' \
#  --experiment_name 'alternate-start-*' \
#  --plot_alternate
