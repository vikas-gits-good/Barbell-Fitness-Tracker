import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


df = pd.read_pickle("../../data/interim/03_data_features.pkl")

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
df_train = df.drop(["Participant", "Category", "Set", "Duration"], axis=1)

X = df_train.drop("Label", axis=1)
Y = df_train["Label"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42, stratify=Y
)

fig, ax = plt.subplots(figsize=(10, 5))
df_train["Label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)
Y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
Y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------
basic_feat = ["acl_x", "acl_y", "acl_z", "gyr_x", "gyr_y", "gyr_z"]
squared_feat = ["r_acl", "r_gyr"]
pca_feat = ["pca_1", "pca_2", "pca_3"]
time_feat = [col for col in df_train.columns if "_temp_" in col]
freq_feat = [col for col in df_train.columns if ("_freq" in col) or ("_pse" in col)]
clus_feat = ["Cluster"]

dict_feat = {
    "Basic": basic_feat,
    "Squared": squared_feat,
    "PCA": pca_feat,
    "Time": time_feat,
    "Frequency": freq_feat,
    "Cluster": clus_feat,
}
for key, val in dict_feat.items():
    print(f"{key}: {len(val)}")

FS_1 = list(set(basic_feat))
FS_2 = list(set(basic_feat + squared_feat + pca_feat))
FS_3 = list(set(FS_2 + time_feat))
FS_4 = list(set(FS_3 + freq_feat + clus_feat))


# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
learner = ClassificationAlgorithms()
max_feat = 10

sel_feat, ord_feat, ord_scores = learner.forward_selection(
    max_features=max_feat, X_train=X_train, y_train=Y_train
)

# sel_feat = [
#     "pca_1",
#     "acl_z_freq_0.0_Hz_ws_14",
#     "acl_x_freq_0.0_Hz_ws_14",
#     "r_gyr_freq_0.0_Hz_ws_14",
#     "acl_z_temp_mean_ws_5",
#     "gyr_y_temp_mean_ws_5",
#     "gyr_x_freq_0.0_Hz_ws_14",
#     "gyr_y_freq_1.071_Hz_ws_14",
#     "gyr_z_max_freq",
#     "gyr_y_max_freq",
# ]

sel_feat = [
    "pca_1",
    "r_gyr_freq_0.0_Hz_ws_14",
    "acl_x_freq_0.0_Hz_ws_14",
    "acl_z_freq_0.0_Hz_ws_14",
    "acl_z_temp_mean_ws_5",
    "r_gyr_freq_2.143_Hz_ws_14",
    "gyr_z_freq_1.071_Hz_ws_14",
    "r_gyr_freq_0.714_Hz_ws_14",
    "r_gyr_max_freq",
    "Cluster",
]

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_feat + 1, 1), ord_scores)
plt.xlabel("Nu. of Feats")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, max_feat + 1, 1))
plt.show()

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------
possible_feature_sets = [FS_1, FS_2, FS_3, FS_4, sel_feat]
feature_names = ["FS-1", "FS-2", "FS-3", "FS-4", "Selected Features"]

iterations = 1
score_df = pd.DataFrame()

for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            Y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(Y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, Y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(Y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, Y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(Y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, Y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(Y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, Y_train, selected_test_X)

    performance_test_nb = accuracy_score(Y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])

# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------
score_df.sort_values(by="accuracy", ascending=False)

plt.figure(figsize=(10, 10))
sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1)
plt.legend(loc="lower right")
plt.show()

# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(X_train[FS_4], Y_train, X_test[FS_4], gridsearch=True)

accuracy = accuracy_score(Y_test, class_test_y)

classes = class_test_prob_y.columns

cm = confusion_matrix(Y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------
df_partic = df.drop(["Set", "Category", "Duration"], axis=1)

X_train = df_partic[df_partic["Participant"] != "A"].drop("Label", axis=1)
Y_train = df_partic[df_partic["Participant"] != "A"]["Label"]

X_test = df_partic[df_partic["Participant"] == "A"].drop("Label", axis=1)
Y_test = df_partic[df_partic["Participant"] == "A"]["Label"]

X_train = X_train.drop(["Participant"], axis=1)
X_test = X_test.drop(["Participant"], axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
df_train["Label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)
Y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
Y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(X_train[FS_4], Y_train, X_test[FS_4], gridsearch=True)

accuracy = accuracy_score(Y_test, class_test_y)

classes = class_test_prob_y.columns

cm = confusion_matrix(Y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.feedforward_neural_network(
    X_train[FS_4], Y_train, X_test[FS_4], gridsearch=True
)

accuracy = accuracy_score(Y_test, class_test_y)

classes = class_test_prob_y.columns

cm = confusion_matrix(Y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()
print(accuracy)
