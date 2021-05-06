# BeamformIt

This folder contains the scripts to train a CTC system using a tool named [BeamformIt](https://github.com/xanguera/BeamformIt) on TIMIT.

# How to run
``python create_timit_combined.py`` (Creates ``timit_combined.csv``)

``python beamformit_prepare.py`` (Creates a directory named ``TIMIT_BeamformIt``)

With environmental corruption: ``python train_env.py hparams/train.yaml --data_folder /path/to/TIMIT_BeamformIt``

Without environmental corruption: ``python train.py hparams/train.yaml --data_folder /path/to/TIMIT_BeamformIt``

# Results

| train file   | hyperparams file | Val. PER | Test PER | Model link    | GPUs        |
| ------------ | ---------------- | -------- | -------- | ------------- | ----------- |
| train_env.py | train.yaml       | 24.78    | 26.48    | Not available | 1xV100 16GB |
| train.py     | train.yaml       | 25.70    | 27.05    | Not available | 1xV100 16GB |

# Training time

With environmental corruption: About 3 min and 4 sec for each epoch with a Tesla V100.

Without environmental corruption: About 1 min and 18 sec for each epoch with a Tesla V100.