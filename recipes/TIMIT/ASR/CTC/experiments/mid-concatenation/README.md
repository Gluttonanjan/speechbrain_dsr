# Mid-concatenation

This folder contains the scripts to train a CTC system using mid-concatenation on TIMIT.

# How to run

**Note:** Copy and replace ``CRDNN.py`` in ``speechbrain/lobes/models`` in pip's installation directory.

``python mid-concatenation_prepare.py``

With environmental corruption: ``python train_env.py hparams/train_env.yaml --data_folder /path/to/TIMIT_4_channels/TIMIT_LA2``

Without environmental corruption: ``python train.py hparams/train.yaml --data_folder /path/to/TIMIT_4_channels/TIMIT_LA2``

# Results

| train file   | hyperparams file | Val. PER | Test PER | Model link    | GPUs        |
| ------------ | ---------------- | -------- | -------- | ------------- | ----------- |
| train_env.py | train_env.yaml   | 22.49    | 25.05    | Not available | 1xV100 16GB |
| train.py     | train.yaml       | 23.22    | 25.69    | Not available | 1xV100 16GB |

# Training time

With environmental corruption: About 5 min and 58 sec for each epoch with a Tesla V100.

Without environmental corruption: About 2 min for each epoch with a Tesla V100.