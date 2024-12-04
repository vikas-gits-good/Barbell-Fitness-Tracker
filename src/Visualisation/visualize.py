import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
set_df = df[df["Set"] == 1]
plt.plot(set_df["acl_y"])

plt.plot(set_df["acl_y"].reset_index(drop=True))


# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
for lbl in df["Label"].unique():
    subset = df[df["Label"] == lbl]
    display(subset.head(2))
    fig, ax = plt.subplots()
    plt.plot(subset["acl_y"].reset_index(drop=True), label=lbl)
    plt.legend()
    plt.show()

for lbl in df["Label"].unique():
    subset = df[df["Label"] == lbl]
    # display(subset.head(2))
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acl_y"].reset_index(drop=True), label=lbl)
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
ctgr_df = df.query("Label == 'squat'").query("Participant == 'A'").reset_index()
fig, ax = plt.subplots()
ctgr_df.groupby(["Category"])["acl_y"].plot()
ax.set_ylabel("acl_y")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
prpn_df = df.query("Label == 'bench'").sort_values("Participant").reset_index()
fig, ax = plt.subplots()
prpn_df.groupby(["Participant"])["acl_y"].plot()
ax.set_ylabel("acl_y")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
label = "squat"
participant = "A"
all_axis_df = (
    df.query(f"Label == '{label}'")
    .query(f"Participant == '{participant}'")
    .reset_index()
)

fig, ax = plt.subplots()
all_axis_df[["acl_x", "acl_y", "acl_z"]].plot(ax=ax)
ax.set_ylabel("acl_y")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
labels = df["Label"].unique()
participants = df["Participant"].unique()

for lbl in labels:
    for prpn in participants:
        all_axis_df = (
            df.query(f"Label == '{lbl}'")
            .query(f"Participant == '{prpn}'")
            .reset_index()
        )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["acl_x", "acl_y", "acl_z"]].plot(ax=ax)
            ax.set_ylabel("acl_y")
            ax.set_xlabel("samples")
            plt.title(f"{lbl}: {prpn}".title())
            plt.legend()

for lbl in labels:
    for prpn in participants:
        all_axis_df = (
            df.query(f"Label == '{lbl}'")
            .query(f"Participant == '{prpn}'")
            .reset_index()
        )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_ylabel("gyr_y")
            ax.set_xlabel("samples")
            plt.title(f"{lbl}: {prpn}".title())
            plt.legend()

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
label = "row"
participant = "A"
comb_plot_df = (
    df.query(f"Label == '{label}'")
    .query(f"Participant == '{participant}'")
    .reset_index(drop=True)
)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
comb_plot_df[["acl_x", "acl_y", "acl_z"]].plot(ax=ax[0])
comb_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

ax[0].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].set_xlabel("samples")


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["Label"].unique()
participants = df["Participant"].unique()

for lbl in labels:
    for prpn in participants:
        comb_plot_df = (
            df.query(f"Label == '{lbl}'")
            .query(f"Participant == '{prpn}'")
            .reset_index()
        )
        if len(comb_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            comb_plot_df[["acl_x", "acl_y", "acl_z"]].plot(ax=ax[0])
            comb_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

            ax[0].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[0].set_ylabel("acl")
            ax[1].set_xlabel("samples")
            ax[1].set_ylabel("gyr")
            plt.title(f"{lbl}: {prpn}".title())  # ,loc="upper left")
            plt.savefig(f"../../reports/figures/{lbl.title()} {prpn}.png")
            plt.show()
