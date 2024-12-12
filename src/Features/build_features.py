import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
df.columns
pred_cols = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in pred_cols:
    df[col] = df[col].interpolate()

# df.info()
# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
df[df["Set"] == 25]["acl_y"].plot()
df[df["Set"] == 50]["acl_y"].plot()

for s in df["Set"].unique():
    strt = df[df["Set"] == s].index[0]
    stop = df[df["Set"] == s].index[-1]
    duration = stop - strt
    df.loc[(df["Set"] == s), "Duration"] = duration.seconds

dur_df = df.groupby(["Category"])["Duration"].mean()

dur_df.iloc[0] / 5
dur_df.iloc[1] / 10


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
df_lpf = df.copy()
lpf = LowPassFilter()

fs = 1000 / 200
cutoff = 1.3

df_lpf = lpf.low_pass_filter(
    data_table=df_lpf, col="acl_y", sampling_frequency=fs, cutoff_frequency=cutoff
)

###
subset = df_lpf[df_lpf["Set"] == 45]
print(subset["Label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acl_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acl_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True)
###

for col in pred_cols:
    df_lpf = lpf.low_pass_filter(
        data_table=df_lpf, col=col, sampling_frequency=fs, cutoff_frequency=cutoff
    )
    df_lpf[col] = df_lpf[f"{col}_lowpass"]
    del df_lpf[f"{col}_lowpass"]


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = df_lpf.copy()
PCA = PrincipalComponentAnalysis()
pc_val = PCA.determine_pc_explained_variance(df_pca, pred_cols)

# plt.figure(figsize=(10, 10))
# plt.plot(range(1, len(pred_cols) + 1), pc_val)
# plt.xlabel("Principal Component Number")
# plt.show()

df_pca = PCA.apply_pca(data_table=df_pca, cols=pred_cols, number_comp=3)

###
subset = df_pca[df_pca["Set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()
###


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_sqr = df_pca.copy()
r_acl = df_sqr["acl_x"] ** 2 + df_sqr["acl_y"] ** 2 + df_sqr["acl_z"] ** 2
r_gyr = df_sqr["gyr_x"] ** 2 + df_sqr["gyr_y"] ** 2 + df_sqr["gyr_z"] ** 2

df_sqr["r_acl"] = np.sqrt(r_acl)
df_sqr["r_gyr"] = np.sqrt(r_gyr)

# df_sqr['r_acl'] = np.linalg.norm(df_sqr[['acl_x', 'acl_y', 'acl_z']], axis=1)
# df_sqr['r_gyr'] = np.linalg.norm(df_sqr[['gyr_x', 'gyr_y', 'gyr_z']], axis=1)

subset = df_sqr[df_sqr["Set"] == 18]
subset[["r_acl", "r_gyr"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------
df_tmp = df_sqr.copy()
NumAbs = NumericalAbstraction()

pred_cols = pred_cols + ["r_acl", "r_gyr"]

ws = int(1000 / 200)

# for col in pred_cols:
#     df_tmp = NumAbs.abstract_numerical(
#         data_table=df_tmp, cols=[col], window_size=ws, aggregation_function="mean"
#     )
#     df_tmp = NumAbs.abstract_numerical(
#         data_table=df_tmp, cols=[col], window_size=ws, aggregation_function="std"
#     )

df_list_temp = []
for s in df_tmp["Set"].unique():
    subset = df_tmp[df_tmp["Set"] == s].copy()
    for col in pred_cols:
        subset = NumAbs.abstract_numerical(
            data_table=subset, cols=[col], window_size=ws, aggregation_function="mean"
        )
        subset = NumAbs.abstract_numerical(
            data_table=subset, cols=[col], window_size=ws, aggregation_function="std"
        )
    df_list_temp.append(subset)

df_tmp = pd.concat(df_list_temp)
# df_tmp.info()

# subset[
#     ["acl_y", "acl_y_temp_mean_ws_5", "acl_y_temp_std_ws_5"]
# ].plot()  # subplots=True)

# subset[
#     ["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]
# ].plot()  # subplots=True)

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
df_freq = df_tmp.copy().reset_index()
FreqAbs = FourierTransformation()
df_freq.columns
fs = int(1000 / 200)
ws = int(2800 / 200)

# df_freq = FreqAbs.abstract_frequency(
#     data_table=df_freq, cols=["acl_y"], window_size=ws, sampling_rate=fs
# )

subset = df_freq[df_freq["Set"] == 15]
subset[["acl_y"]].plot()
subset[
    [
        "acl_y_max_freq",
        "acl_y_freq_weighted",
        "acl_y_pse",
        "acl_y_freq_1.429_Hz_ws_14",
        "acl_y_freq_2.5_Hz_ws_14",
    ]
].plot()

df_list_freq = []
for s in df_freq["Set"].unique():
    print(f"Applying FT to {s}")
    subset = df_freq[df_freq["Set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(
        data_table=subset, cols=pred_cols, window_size=ws, sampling_rate=fs
    )
    df_list_freq.append(subset)
df_freq = pd.concat(df_list_freq).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
df_freq = df_freq.dropna()

df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_clus = df_freq.copy()
clus_cols = ["acl_x", "acl_y", "acl_z"]
k_val = range(2, 10)
inertia = []

for k in k_val:
    subset = df_clus[clus_cols]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    clus_lbl = kmeans.fit_predict(subset)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_val, inertia)
plt.xlabel("k")
plt.ylabel("Inertia")
plt.show()

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_clus[clus_cols]
df_clus["Cluster"] = kmeans.fit_predict(subset)

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for c in df_clus["Cluster"].unique():
    subset = df_clus[df_clus["Cluster"] == c]
    ax.scatter(subset["acl_x"], subset["acl_y"], subset["acl_z"], label=c)

ax.set_xlabel("axl_x")
ax.set_ylabel("axl_y")
ax.set_zlabel("axl_z")
plt.legend()
plt.show()

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for c in df_clus["Label"].unique():
    subset = df_clus[df_clus["Label"] == c]
    ax.scatter(subset["acl_x"], subset["acl_y"], subset["acl_z"], label=c)

ax.set_xlabel("axl_x")
ax.set_ylabel("axl_y")
ax.set_zlabel("axl_z")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_clus.to_pickle("../../data/interim/03_data_features.pkl")
