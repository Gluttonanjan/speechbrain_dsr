# Single channel

This folder contains the scripts to train a CTC system on one of the channels in TIMIT.

# How to run

``python single_channel_prepare.py``

With environmental corruption: ``python train_env.py hparams/train.yaml --data_folder /path/to/TIMIT_4_channels/TIMIT_LA2``

Without environmental corruption: ``python train.py hparams/train.yaml --data_folder /path/to/TIMIT_4_channels/TIMIT_LA2``

# Results

| train file   | hyperparams file | Val. PER | Test PER | Model link    | GPUs        |
| ------------ | ---------------- | -------- | -------- | ------------- | ----------- |
| train_env.py | train.yaml       | 26.38    | 29.15    | Not available | 1xV100 16GB |
| train.py     | train.yaml       | 27.26    | 29.54    | Not available | 1xV100 16GB |

# Training time

With environmental corruption: About 3 min for each epoch with a Tesla V100.

Without environmental corruption: About 1 min and 18 sec for each epoch with a Tesla V100.