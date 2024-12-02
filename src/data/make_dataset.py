import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
sing_file_acl = pd.read_csv(
    "../../data/raw/MetaMotion/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)

sing_file_gyr = pd.read_csv(
    "../../data/raw/MetaMotion/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files = glob("../../data/raw/MetaMotion/MetaMotion/*.csv")


# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
data_path = "../../data/raw/MetaMotion/MetaMotion/*.csv"
f = files[0]
participant = f.split("-")[0][-1]
label = f.split("-")[1]
category = f.split("-")[2].rstrip("12345")

# dict_meta = {
#     "Participant": f.split("-")[0][-1],
#     "Label": f.split("-")[1],
#     "Category": f.split("-")[2].rstrip("12345"),
# }

df = pd.read_csv(f)

df["Participant"] = participant
df["Label"] = label
df["Category"] = category


# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
df_acl = pd.DataFrame()
df_gyr = pd.DataFrame()

acl_set = 1
gyr_set = 1

for f in files:
    participant = f.split("-")[0][-1]
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("12345").rstrip("MetaWear_2019")

    df = pd.read_csv(f)

    df["Participant"] = participant
    df["Label"] = label
    df["Category"] = category

    if "Accelerometer" in f:
        df["Set"] = acl_set
        acl_set += 1
        df_acl = pd.concat([df_acl, df])
    elif "Gyroscope" in f:
        df["Set"] = gyr_set
        gyr_set += 1
        df_gyr = pd.concat([df_gyr, df])

pd.to_datetime(df["epoch (ms)"], unit="ms")


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

df_acl.index = pd.to_datetime(df_acl["epoch (ms)"], unit="ms")
df_gyr.index = pd.to_datetime(df_gyr["epoch (ms)"], unit="ms")


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
files = glob("../../data/raw/MetaMotion/MetaMotion/*.csv")


def read_data_from_files(Files):
    df_acl = pd.DataFrame()
    df_gyr = pd.DataFrame()

    acl_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split("-")[0][-1]
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("12345").rstrip("MetaWear_2019")

        df = pd.read_csv(f)

        df["Participant"] = participant
        df["Label"] = label
        df["Category"] = category

        if "Accelerometer" in f:
            df["Set"] = acl_set
            acl_set += 1
            df_acl = pd.concat([df_acl, df])
        elif "Gyroscope" in f:
            df["Set"] = gyr_set
            gyr_set += 1
            df_gyr = pd.concat([df_gyr, df])

    df_acl.index = pd.to_datetime(df_acl["epoch (ms)"], unit="ms")
    df_gyr.index = pd.to_datetime(df_gyr["epoch (ms)"], unit="ms")

    del df_acl["epoch (ms)"]
    del df_acl["time (01:00)"]
    del df_acl["elapsed (s)"]

    del df_gyr["epoch (ms)"]
    del df_gyr["time (01:00)"]
    del df_gyr["elapsed (s)"]

    return [df_acl, df_gyr]


[df_acl, df_gyr] = read_data_from_files(Files=files)
# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
data_merged = pd.concat([df_acl.iloc[:, :3], df_gyr], axis=1)

data_merged.columns = [
    "acl_x",
    "acl_y",
    "acl_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "Participant",
    "Label",
    "Category",
    "Set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    "acl_x": "mean",
    "acl_y": "mean",
    "acl_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "Participant": "last",
    "Label": "last",
    "Category": "last",
    "Set": "last",
}

data_merged.columns

# data_merged.info()

# data_merged[:1000].resample(rule="200ms").apply(sampling)

days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

data_resampled["Set"] = data_resampled["Set"].astype(int)

# data_resampled.info()
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
data_resampled.to_pickle(path="../../data/interim/01_data_processed.pkl")
