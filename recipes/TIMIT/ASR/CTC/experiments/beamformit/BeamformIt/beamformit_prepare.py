import os
import pandas as pd

import shutil

from timit_prepare import prepare_timit

dir_train = []
for root, dirs, files in os.walk(
    "BeamformIt/TIMIT_4_channels/train"
):  # /path/to/TIMIT_4_channels/train
    for dir in dirs:
        dir_train.append(os.path.join(root, dir))

dir_train = set(dir_train)
dir_train = [i[len("BeamformIt/TIMIT_4_channels/train/") :] for i in dir_train]
dir_train = [i for i in dir_train if len(i) > 3]

for i in dir_train:
    os.makedirs("BeamformIt/TIMIT_BeamformIt/train/" + i)

dir_test = []
for root, dirs, files in os.walk(
    "BeamformIt/TIMIT_4_channels/test"
):  # /path/to/TIMIT_4_channels/test
    for dir in dirs:
        dir_test.append(os.path.join(root, dir))

dir_test = set(dir_test)
dir_test = [i[len("BeamformIt/TIMIT_4_channels/test/") :] for i in dir_test]
dir_test = [i for i in dir_test if len(i) > 3]

for i in dir_test:
    os.makedirs("BeamformIt/TIMIT_BeamformIt/test/" + i)


df = pd.read_csv("timit_combined.csv")
df["location"] = df["location"].apply(
    lambda x: x.replace(
        "TIMIT_combined", "BeamformIt/TIMIT_4_channels/TIMIT_LA2"
    )
)
df["location"] = df["location"].apply(lambda x: x.replace(".wav", "_LA2.wav"))

for i in range(len(df)):
    os.makedirs("input_channels")
    mic1 = df["location"][i]
    mic2 = df["location"][i].replace("_LA2", "_LA3")
    mic3 = df["location"][i].replace("_LA2", "_LA4")
    mic4 = df["location"][i].replace("_LA2", "_LA5")
    filename = mic1.split("/")[-1].split("_")[0]
    destination = "/".join(
        mic1.replace("/TIMIT_LA2", "")
        .replace("TIMIT_4_channels", "TIMIT_BeamformIt")
        .split("/")[:-1]
    )
    os.system("cp {} -r BeamformIt/input_channels".format(mic1))
    os.system("cp {} -r BeamformIt/input_channels".format(mic2))
    os.system("cp {} -r BeamformIt/input_channels".format(mic3))
    os.system("cp {} -r BeamformIt/input_channels".format(mic4))
    os.system(
        "bash do_beamforming.sh BeamformIt/input_channels {}".format(filename)
    )  # /path/to/do_beamforming.sh
    location = "BeamformIt/output/{}/{}.wav".format(filename, filename)
    os.system("cp {} -r {}".format(location, destination))
    shutil.rmtree("BeamformIt/input_channels")


data_folder = "BeamformIt/TIMIT"  # /path/to/TIMIT
splits = ["train", "dev", "test"]
save_folder = "TIMIT_prepared"
prepare_timit(data_folder, splits, save_folder)

train_df = pd.read_csv("TIMIT_prepared/train.csv")
train_df["wav"] = train_df["wav"].apply(
    lambda x: x.replace("BeamformIt/TIMIT", "BeamformIt/TIMIT_BeamformIt")
)
train_df.to_csv("BeamformIt/TIMIT_prepared/train.csv", index=False)

dev_df = pd.read_csv("BeamformIt/TIMIT_prepared/dev.csv")
dev_df["wav"] = dev_df["wav"].apply(
    lambda x: x.replace("BeamformIt/TIMIT", "BeamformIt/TIMIT_BeamformIt")
)
dev_df.to_csv("BeamformIt/TIMIT_prepared/dev.csv", index=False)

test_df = pd.read_csv("TIMIT_prepared/test.csv")
test_df["wav"] = test_df["wav"].apply(
    lambda x: x.replace("BeamformIt/TIMIT", "BeamformIt/TIMIT_BeamformIt")
)
test_df.to_csv("BeamformIt/TIMIT_prepared/test.csv", index=False)

shutil.copy(
    "BeamformIt/TIMIT_prepared/train.csv", "BeamformIt/TIMIT_BeamformIt"
)
shutil.copy("BeamformIt/TIMIT_prepared/dev.csv", "BeamformIt/TIMIT_BeamformIt")
shutil.copy("BeamformIt/TIMIT_prepared/test.csv", "BeamformIt/TIMIT_BeamformIt")

for root, file, dirs in os.walk("BeamformIt/TIMIT"):
    for dir in dirs:
        source = os.path.join(root, dir)
        if ".wav" not in source:
            destination = source.replace("TIMIT", "TIMIT_BeamformIt")
            shutil.copyfile(source, destination)
