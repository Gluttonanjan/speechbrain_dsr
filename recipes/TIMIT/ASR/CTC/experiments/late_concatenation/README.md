# Late concatenation

This folder contains the scripts to train a CTC system using late concatenation on TIMIT.

# How to run

``python late_concatenation_prepare.py``

With environmental corruption: ``python train_env.py hparams/train_env.yaml --data_folder /path/to/TIMIT_4_channels/TIMIT_LA2``

Without environmental corruption: ``python train.py hparams/train.yaml --data_folder /path/to/TIMIT_4_channels/TIMIT_LA2``

# Results

| train file   | hyperparams file | Val. PER | Test PER | Model link    | GPUs        |
| ------------ | ---------------- | -------- | -------- | ------------- | ----------- |
| train_env.py | train_env.yaml   | 23.96    | 26.33    | Not available | 1xV100 16GB |
| train.py     | train.yaml       | 23.82    | 26.01    | Not available | 1xV100 16GB |

# Training time

With environmental corruption: About 11 min and 42 sec for each epoch with a Tesla V100.

Without environmental corruption: About 5 min and 32 sec for each epoch with a Tesla V100.