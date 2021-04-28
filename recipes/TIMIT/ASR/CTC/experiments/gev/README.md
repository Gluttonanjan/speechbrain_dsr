# Generalized eigenvalue beamforming

This folder contains the scripts to train a CTC system using generalized eigenvalue beamforming on TIMIT.

# How to run

``python gev_prepare.py`` (Creates a directory named ``TIMIT_GeV``)

With environmental corruption: ``python train_env.py hparams/train.yaml --data_folder /path/to/TIMIT_GeV``

Without environmental corruption: ``python train.py hparams/train.yaml --data_folder /path/to/TIMIT_GeV``

# Results

| train file   | hyperparams file | Val. PER | Test PER | Model link    | GPUs        |
| ------------ | ---------------- | -------- | -------- | ------------- | ----------- |
| train_env.py | train.yaml       | 24.78    | 28.29    | Not available | 1xV100 16GB |
| train.py     | train.yaml       | 25.72    | 28.25    | Not available | 1xV100 16GB |

# Training time

With environmental corruption: About 2 min and 18 sec for each epoch with a Tesla V100.

Without environmental corruption: About 1 min and 42 sec for each epoch with a Tesla V100.