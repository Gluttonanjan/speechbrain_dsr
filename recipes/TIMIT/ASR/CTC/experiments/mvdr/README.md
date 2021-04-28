# Minimum variance distortionless response (MVDR) beamforming

This folder contains the scripts to train a CTC system using minimum variance distortionless response (MVDR) beamforming on TIMIT.

# How to run

``python mvdr_prepare.py`` (Creates a directory named ``TIMIT_MVDR``)

With environmental corruption: ``python train_env.py hparams/train.yaml --data_folder /path/to/TIMIT_MVDR``

Without environmental corruption: ``python train.py hparams/train.yaml --data_folder /path/to/TIMIT_MVDR``

# Results

| train file   | hyperparams file | Val. PER | Test PER | Model link    | GPUs        |
| ------------ | ---------------- | -------- | -------- | ------------- | ----------- |
| train_env.py | train.yaml       | 26.34    | 28.32    | Not available | 1xV100 16GB |
| train.py     | train.yaml       | 26.64    | 28.93    | Not available | 1xV100 16GB |

# Training time

With environmental corruption: About 3 min and 12 sec for each epoch with a Tesla V100.

Without environmental corruption: About 1 min and 18 sec for each epoch with a Tesla V100.