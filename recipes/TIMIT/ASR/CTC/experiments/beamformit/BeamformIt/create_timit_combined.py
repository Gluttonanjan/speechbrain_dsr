import os
import pandas as pd

import torch
from speechbrain.dataio.dataio import read_audio, write_audio

fs = 16000


train = []
test = []

for root, dirs, files in os.walk(
    os.path.abspath("TIMIT_4_channels/train")
):  # /path/to/TIMIT_4_channels/train
    for file in files:
        if "LA2" in file or "LA3" in file or "LA4" in file or "LA5" in file:
            train.append(os.path.join(root, file))

for root, dirs, files in os.walk(
    os.path.abspath("TIMIT_4_channels/test")
):  # /path/to/TIMIT_4_channels/test
    for file in files:
        if "LA2" in file or "LA3" in file or "LA4" in file or "LA5" in file:
            test.append(os.path.join(root, file))

df = pd.DataFrame(train + test)
df.rename(columns={0: "location"}, inplace=True)
df["label"] = df["location"].apply(
    lambda x: "train" if "train" in x else "test"
)

df = df.sort_values(by=["location", "label"]).reset_index(drop=True)
train_df = df[df["label"] == "train"].reset_index(drop=True)
test_df = df[df["label"] == "test"].reset_index(drop=True)

dir_train = []
for root, dirs, files in os.walk(
    "TIMIT_4_channels/train"
):  # /path/to/TIMIT_4_channels/train
    for dir in dirs:
        dir_train.append(os.path.join(root, dir))

dir_train = set(dir_train)
dir_train = [i[len("TIMIT_4_channels/train/") :] for i in dir_train]
dir_train = [i for i in dir_train if len(i) > 3]

for i in dir_train:
    os.makedirs("TIMIT_combined/train/" + i)

dir_test = []
for root, dirs, files in os.walk(
    "TIMIT_4_channels/test"
):  # /path/to/TIMIT_4_channels/test
    for dir in dirs:
        dir_test.append(os.path.join(root, dir))

dir_test = set(dir_test)
dir_test = [i[len("TIMIT_4_channels/test/") :] for i in dir_test]
dir_test = [i for i in dir_test if len(i) > 3]

for i in dir_test:
    os.makedirs("TIMIT_combined/test/" + i)


for i in range(len(train_df)):
    if i % 4 == 0:
        fname = (
            train_df["location"][i][len("TIMIT_4_channels/train/") :]
            .split(".")[0]
            .split("_")[0]
            + ".wav"
        )
        mic1 = read_audio(train_df["location"][i])
        mic2 = read_audio(train_df["location"][i + 1])
        mic3 = read_audio(train_df["location"][i + 2])
        mic4 = read_audio(train_df["location"][i + 3])
        sa = torch.stack((mic1, mic2, mic3, mic4)).transpose(0, 1)
        write_audio("TIMIT_combined/train/" + fname, sa, samplerate=fs)

for i in range(len(test_df)):
    if i % 4 == 0:
        fname = (
            test_df["location"][i][len("TIMIT_4_channels/test/") :]
            .split(".")[0]
            .split("_")[0]
            + ".wav"
        )
        mic1 = read_audio(test_df["location"][i])
        mic2 = read_audio(test_df["location"][i + 1])
        mic3 = read_audio(test_df["location"][i + 2])
        mic4 = read_audio(test_df["location"][i + 3])
        sa = torch.stack((mic1, mic2, mic3, mic4)).transpose(0, 1)
        write_audio("TIMIT_combined/test/" + fname, sa, samplerate=fs)

train_timit_combined = []
test_timit_combined = []
for root, dirs, files in os.walk(os.path.abspath("TIMIT_combined/train")):
    for file in files:
        train_timit_combined.append(os.path.join(root, file))

test_timit_combined = []
for root, dirs, files in os.walk(os.path.abspath("TIMIT_combined/test")):
    for file in files:
        test_timit_combined.append(os.path.join(root, file))

df = pd.DataFrame(train_timit_combined + test_timit_combined)
df.rename(columns={0: "location"}, inplace=True)
df["label"] = df["location"].apply(
    lambda x: "train" if "train" in x else "test"
)

df = df.sort_values(by=["location", "label"]).reset_index(drop=True)
df.to_csv("timit_combined.csv", index=False)
