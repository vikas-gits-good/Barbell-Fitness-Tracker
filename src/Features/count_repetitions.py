import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df = df[df["Label"] != "rest"]
# df["Label"].unique()

r_acl = df["acl_x"] ** 2 + df["acl_y"] ** 2 + df["acl_z"] ** 2
r_gyr = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2

df["r_acl"] = np.sqrt(r_acl)
df["r_gyr"] = np.sqrt(r_gyr)


# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
df_bench = df_ohp = df_row = df_dead = df_squat = pd.DataFrame()
dict_exer = {
    "bench": df_bench,
    "ohp": df_ohp,
    "row": df_row,
    "dead": df_dead,
    "squat": df_squat,
}
for key in dict_exer.keys():
    new_df = pd.concat([dict_exer[key], df[df["Label"] == key]])
    dict_exer[key] = new_df
    globals()[f"df_{key}"] = new_df


# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------
check = 0
df_plot = dict_exer[list(dict_exer.keys())[check]]
sets = 0
df_plot[df_plot["Set"] == df_plot["Set"].unique()[sets]]["acl_x"].plot()
df_plot[df_plot["Set"] == df_plot["Set"].unique()[sets]]["acl_y"].plot()
df_plot[df_plot["Set"] == df_plot["Set"].unique()[sets]]["acl_z"].plot()
df_plot[df_plot["Set"] == df_plot["Set"].unique()[sets]]["r_acl"].plot()

df_plot[df_plot["Set"] == df_plot["Set"].unique()[sets]]["gyr_x"].plot()
df_plot[df_plot["Set"] == df_plot["Set"].unique()[sets]]["gyr_y"].plot()
df_plot[df_plot["Set"] == df_plot["Set"].unique()[sets]]["gyr_z"].plot()
df_plot[df_plot["Set"] == df_plot["Set"].unique()[sets]]["r_gyr"].plot()


# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------
fs = 1000 / 200
lpf = LowPassFilter()


# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------
bench_set = df_bench[df_bench["Set"] == df_bench["Set"].unique()[0]]
squat_set = df_squat[df_squat["Set"] == df_squat["Set"].unique()[0]]
row_set = df_row[df_row["Set"] == df_row["Set"].unique()[0]]
ohp_set = df_ohp[df_ohp["Set"] == df_ohp["Set"].unique()[0]]
dead_set = df_dead[df_dead["Set"] == df_dead["Set"].unique()[0]]

bench_set["r_acl"].plot()

lpf.low_pass_filter(
    data_table=bench_set,
    col="r_acl",
    sampling_frequency=fs,
    cutoff_frequency=0.4,
    order=10,
)["r_acl_lowpass"].plot()

# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------


def count_reps(
    dataset: pd.DataFrame,
    smpl_freq: float = 5,
    ctof_freq: float = 0.4,
    order: int = 10,
    column: str = "r_acl",
):
    data = lpf.low_pass_filter(
        data_table=dataset,
        col=column,
        sampling_frequency=smpl_freq,
        cutoff_frequency=ctof_freq,
        order=order,
    )
    peak_indices = argrelextrema(data[f"{column}_lowpass"].values, np.greater)
    peaks = data.iloc[peak_indices]

    fig, ax = plt.subplots()
    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"], "o", color="red")
    ax.set_ylabel(f"{column}_lowpass")
    exercise = dataset["Label"].iloc[0].title()
    category = dataset["Category"].iloc[0].title()
    plt.title(f"{category} {exercise}: {len(peaks)} peaks")
    return len(peaks)


reps_bench = count_reps(dataset=bench_set, ctof_freq=0.4, column="r_acl")
reps_row = count_reps(dataset=row_set, ctof_freq=0.6, column="gyr_x")
reps_ohp = count_reps(dataset=ohp_set, ctof_freq=0.35, column="r_acl")
reps_dead = count_reps(dataset=dead_set, ctof_freq=0.4, column="r_acl")
reps_squat = count_reps(dataset=squat_set, ctof_freq=0.35, column="r_acl")

# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------
df["Reps"] = df["Category"].apply(lambda x: 5 if x == "heavy" else 10)

df_rep = df.groupby(["Label", "Category", "Set"])["Reps"].max().reset_index()
df_rep["reps_pred"] = 0

for s in df["Set"].unique():
    subset = df[df["Set"] == s]

    column = "r_acl"
    ctf = 0.4

    if subset["Label"].iloc[0] == "Squat":
        ctf = 0.35
    if subset["Label"].iloc[0] == "row":
        ctf = 0.65
        column = "gyr_x"
    if subset["Label"].iloc[0] == "ohp":
        ctf = 0.35

    reps = count_reps(dataset=subset, ctof_freq=ctf, column=column)
    df_rep.loc[df_rep["Set"] == s, "reps_pred"] = reps


# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------
error = mean_absolute_error(df_rep["Reps"], df_rep["reps_pred"]).round(2)

df_rep.groupby(["Label", "Category"])["Reps", "reps_pred"].mean().plot.bar()
