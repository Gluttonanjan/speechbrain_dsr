# Early concatenation

This folder contains the scripts to train a CTC system using early concatenation on TIMIT.

# How to run

``python early_concatenation_prepare.py``

With environmental corruption: ``python train_env.py hparams/train_env.yaml --data_folder /path/to/TIMIT_4_channels/TIMIT_LA2``

Without environmental corruption: ``python train.py hparams/train.yaml --data_folder /path/to/TIMIT_4_channels/TIMIT_LA2``

# Results

| train file   | hyperparams file | Val. PER | Test PER | Model link    | GPUs        |
| ------------ | ---------------- | -------- | -------- | ------------- | ----------- |
| train_env.py | train_env.yaml   | 22.15    | 24.89    | Not available | 1xV100 16GB |
| train.py     | train.yaml       | 23.36    | 25.76    | Not available | 1xV100 16GB |

# Training time

With environmental corruption: About 5 min and 54 sec for each epoch with a Tesla V100.

Without environmental corruption: About 2 min and 25 sec for each epoch with a Tesla V100.