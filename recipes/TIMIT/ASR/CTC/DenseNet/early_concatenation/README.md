# Early concatenation

This folder contains the scripts to train a CTC system using early concatenation on TIMIT.

# How to run

**Note:** Copy and replace the following files in mentioned locations in pip's installation directory:

1. ``CRDNN.py`` in ``speechbrain/lobes/models``
2. ``CNN.py`` in ``speechbrain/nnet``
3. ``containers.py`` in ``speechbrain/nnet``

``python early_concatenation_prepare.py``

``python train_env.py hparams/train_env.yaml --data_folder /path/to/TIMIT_4_channels/TIMIT_LA2``

# Results

| train file   | hyperparams file | Val. PER | Test PER | Model link    | GPUs        |
| ------------ | ---------------- | -------- | -------- | ------------- | ----------- |
| train_env.py | train_env.yaml   | 22.66    | 24.90    | Not available | 1xV100 16GB |

# Training time

About 9 min and 45 sec for each epoch with a Tesla V100.