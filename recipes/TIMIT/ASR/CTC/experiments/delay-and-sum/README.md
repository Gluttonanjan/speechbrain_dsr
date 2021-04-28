# Delay and sum beamforming

This folder contains the scripts to train a CTC system using delay and sum beamforming on TIMIT.

# How to run

``python delay_and_sum_prepare.py`` (Creates a directory named ``TIMIT_DAS``)

With environmental corruption: ``python train_env.py hparams/train.yaml --data_folder /path/to/TIMIT_DAS``

Without environmental corruption: ``python train.py hparams/train.yaml --data_folder /path/to/TIMIT_DAS``

# Results

| train file   | hyperparams file | Val. PER | Test PER | Model link    | GPUs        |
| ------------ | ---------------- | -------- | -------- | ------------- | ----------- |
| train_env.py | train.yaml       | 25.18    | 27.0     | Not available | 1xV100 16GB |
| train.py     | train.yaml       | 26.02    | 28.01    | Not available | 1xV100 16GB |

# Training time

With environmental corruption: About 3 min and 10 sec for each epoch with a Tesla V100.

Without environmental corruption: About 1 min and 18 sec for each epoch with a Tesla V100.