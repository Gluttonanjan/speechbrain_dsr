import os
import pandas as pd
import matplotlib.pyplot as plt

import shutil

from timit_prepare import prepare_timit

from speechbrain.dataio.dataio import read_audio, write_audio
from speechbrain.processing.features import STFT, ISTFT
from speechbrain.processing.multi_mic import Covariance
from speechbrain.processing.multi_mic import GccPhat
from speechbrain.processing.multi_mic import Mvdr

import torch

fs = 16000  # sampling rate


def minimum_variance_distortionless_response(audio_file, show_plots=False):
    xs_speech = read_audio(audio_file)
    xs_speech = xs_speech.unsqueeze(0)

    stft = STFT(sample_rate=fs)
    cov = Covariance()
    gccphat = GccPhat()
    mvdr = Mvdr()
    istft = ISTFT(sample_rate=fs)

    Xs = stft(xs_speech)
    XXs = cov(Xs)
    tdoas = gccphat(XXs)
    Ys_mvdr = mvdr(Xs, XXs, tdoas)
    ys_mvdr = istft(Ys_mvdr)

    if show_plots:
        plt.figure(1)
        plt.title("Noisy signal at microphone 1")
        plt.imshow(
            torch.transpose(
                torch.log(Xs[0, :, :, 0, 0] ** 2 + Xs[0, :, :, 1, 0] ** 2), 1, 0
            ),
            origin="lower",
        )
        plt.figure(2)
        plt.title("Noisy signal at microphone 1")
        plt.plot(xs_speech.squeeze()[:, 0])
        plt.figure(3)
        plt.title("Beamformed signal")
        plt.imshow(
            torch.transpose(
                torch.log(
                    Ys_mvdr[0, :, :, 0, 0] ** 2 + Ys_mvdr[0, :, :, 1, 0] ** 2
                ),
                1,
                0,
            ),
            origin="lower",
        )
        plt.figure(4)
        plt.title("Beamformed signal")
        plt.plot(ys_mvdr.squeeze())
        plt.show()

    return ys_mvdr.squeeze()


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

timit_df = pd.DataFrame(train_timit_combined + test_timit_combined)
timit_df.rename(columns={0: "location"}, inplace=True)
timit_df["label"] = timit_df["location"].apply(
    lambda x: "train" if "train" in x else "test"
)
timit_df = timit_df.sort_values(by=["location", "label"]).reset_index(drop=True)

dir_files = []
for root, dirs, files in os.walk("TIMIT_combined"):
    for dir in dirs:
        dir_files.append(os.path.join(root, dir))

dir_files = ["/".join(i.split("/")[3:]) for i in dir_files]

for i in dir_files:
    os.makedirs("TIMIT_MVDR/" + i)

for i in range(len(timit_df)):
    source = timit_df["location"][i][9:]
    destination = "TIMIT_MVDR/" + "/".join(source.split("/")[1:])
    speech_ds = minimum_variance_distortionless_response(
        source, show_plots=False
    )
    write_audio(destination, speech_ds, samplerate=fs)

data_folder = "TIMIT"  # /path/to/TIMIT
splits = ["train", "dev", "test"]
save_folder = "TIMIT_prepared"
prepare_timit(data_folder, splits, save_folder)

train_df = pd.read_csv("TIMIT_prepared/train.csv")
train_df["wav"] = train_df["wav"].apply(
    lambda x: x.replace("TIMIT", "TIMIT_MVDR")
)
train_df.to_csv("TIMIT_prepared/train.csv", index=False)

dev_df = pd.read_csv("TIMIT_prepared/dev.csv")
dev_df["wav"] = dev_df["wav"].apply(lambda x: x.replace("TIMIT", "TIMIT_MVDR"))
dev_df.to_csv("TIMIT_prepared/dev.csv", index=False)

test_df = pd.read_csv("TIMIT_prepared/test.csv")
test_df["wav"] = test_df["wav"].apply(
    lambda x: x.replace("TIMIT", "TIMIT_MVDR")
)
test_df.to_csv("TIMIT_prepared/test.csv", index=False)

shutil.copy("TIMIT_prepared/train.csv", "TIMIT_MVDR")
shutil.copy("TIMIT_prepared/dev.csv", "TIMIT_MVDR")
shutil.copy("TIMIT_prepared/test.csv", "TIMIT_MVDR")

for root, file, dirs in os.walk("TIMIT"):  # /path/to/TIMIT
    for dir in dirs:
        source = os.path.join(root, dir)
        if ".wav" not in source:
            destination = source.replace("TIMIT", "TIMIT_MVDR")
            shutil.copyfile(
                source, destination
            )  # copy from /path/to/TIMIT to /path/to/TIMIT_MVDR
