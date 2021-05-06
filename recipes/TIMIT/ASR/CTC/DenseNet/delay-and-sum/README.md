# Delay and sum beamforming

This folder contains the scripts to train a CTC system using delay and sum beamforming on TIMIT.

# How to run

**Note:** Copy and replace the following files in mentioned locations in pip's installation directory:

1. ``CRDNN.py`` in ``speechbrain/lobes/models``
2. ``CNN.py`` in ``speechbrain/nnet``
3. ``containers.py`` in ``speechbrain/nnet``

``python delay_and_sum_prepare.py`` (Creates a directory named ``TIMIT_DAS``)

``python train_env.py hparams/train_env.yaml --data_folder /path/to/TIMIT_DAS``

# Results

| train file   | hyperparams file | Val. PER | Test PER | Model link    | GPUs        |
| ------------ | ---------------- | -------- | -------- | ------------- | ----------- |
| train_env.py | train_env.yaml   | 24.85    | 27.50    | Not available | 1xV100 16GB |

# Training time

About 4 min and 10 sec for each epoch with a Tesla V100.