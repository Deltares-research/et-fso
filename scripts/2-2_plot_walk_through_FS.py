import pathlib
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import FSO_VAE_generator


data_dir = pathlib.Path(
    "C:/Users/hemert/OneDrive - Stichting Deltares/Desktop/Projects/ET_FSO/Snellius"
)

generator_data = pd.read_feather(
    data_dir.joinpath(
        "Data/generator_data_simple_10numerics_wrecalc_allBasins_no_extremes.feather"
    )
)
index2word = pd.read_feather(data_dir.joinpath("Data/index2word_10numerics.feather"))
tf_dist = pd.read_feather(
    data_dir.joinpath(
        "Data/functions_simple_10_numerics_Distribution_indiv_scale_wrecalc_allBasins_no_extremes.feather"
    )
)


encoder, generator_tf, generator_dist = FSO_VAE_generator.encoder_decoder()


def rescale(x, from_range, to_range):
    if np.std(x) != 0:
        return (x - from_range[0]) / (from_range[1] - from_range[0]) * (
            to_range[1] - to_range[0]
        ) + to_range[0]
    else:
        return x


def ind2word(x, index2word):
    x = x[x != 0].astype(np.int32).astype(np.str_)
    # tf = index2word[x]
    return "".join(index2word[i][0] for i in x)


def index_reconstructor(pred_matrix):
    index = np.zeros(pred_matrix.shape[1]).astype(np.int32)
    for i in range(pred_matrix.shape[0]):
        index[i] = np.argmax(pred_matrix[i, :])
    return index


def index_sampler(pred_matrix):
    index = np.zeros(pred_matrix.shape[0]).astype(np.int32)
    for i in range(pred_matrix.shape[0]):
        index[i] = np.random.choice(pred_matrix.shape[1], 1, p=pred_matrix[i, :])
    return index


def tf_evaluation(predicted_tf):
    predicted_tf_num = predicted_tf
    for i in ["slope", "evi", "sand", "clay", "elevation", "hand", "noise", "bdim"]:
        predicted_tf_num = re.sub(i, "1", predicted_tf_num)
    if (
        re.search("11", predicted_tf_num)
        or re.search("10.", predicted_tf_num)
        or re.search("12", predicted_tf_num)
        or re.search("13", predicted_tf_num)
    ):
        predicted_tf_num = "ERROR"
    for i in range(1, 10):
        if re.search("".join([str(i), "1"]), predicted_tf_num):
            predicted_tf_num = re.sub("".join([str(i), "1"]), "ERROR", predicted_tf_num)
    try:
        # exec("def f_test():\n\t" + "return " + str(predicted_tf_num))
        tf_eval = predicted_tf_num
    except:
        tf_eval = None
    return tf_eval


def tf_prediction(index_pred):
    index_prediction = index_reconstructor(index_pred)
    # index_prediction = index_prediction - 1
    predicted_tf = ind2word(index_prediction, index2word=index2word)
    # print(predicted_tf)
    tf_eval = tf_evaluation(predicted_tf)
    fail_count = 0
    sample_df = []
    if tf_eval is None or tf_eval == "ERROR":
        while len(sample_df) <= 200 & fail_count < 2000:
            index_prediction = index_sampler(index_pred)
            # index_prediction = index_prediction - 1
            predicted_tf = ind2word(index_prediction, index2word=index2word)
            tf_eval = tf_evaluation(predicted_tf)
            if tf_eval is None or tf_eval == "ERROR":
                fail_count += 1
                continue
            sample_df.append(predicted_tf)
        # max_id = np.argmax(np.array(sample_df).astype(np.float32))
        predicted_tf = max(set(sample_df), key=sample_df.count)
    return predicted_tf


# Scale distribution values
def min_max_scale(x):
    return np.round((x - np.min(x)) / (np.max(x) - np.min(x)), 11)


to_drop = ["transfer_function", "min", "max"]
dist_scaled = tf_dist.copy().drop(to_drop, axis=1)
for name in dist_scaled.columns:
    dist_scaled[name] = min_max_scale(dist_scaled[name])

# Train/val split
train_ind = np.load(data_dir.joinpath("VAE Model/train_ind_no_extremes.npy"))

x_train_tf = np.expand_dims(generator_data.iloc[train_ind].values, axis=-1)
x_val_tf = np.expand_dims(generator_data.drop(train_ind).values, axis=-1)

y_train_tf = np.expand_dims(generator_data.iloc[train_ind].values, axis=-1)
y_val_tf = np.expand_dims(generator_data.drop(train_ind).values, axis=-1)


train_dist = dist_scaled.iloc[train_ind].values
val_dist = dist_scaled.drop(train_ind).values

# Walk through Function Space ------------------------------------------------------------
weights = np.arange(0, 1.1, 0.1)
np.random.seed(0)
function_idx = np.random.choice(x_val_tf.shape[0], 2, replace=False)
proto_dist = val_dist[function_idx, ...]
proto_tfs = x_val_tf[function_idx, ...]
proto_tfs_words = [
    ind2word(proto_tfs[i, ...], index2word) for i in range(proto_tfs.shape[0])
]
proto_tfs_encoded = encoder.predict([proto_tfs, proto_dist])

new_tfs = np.zeros((len(weights), proto_tfs_encoded.shape[1]))
for i in range(len(weights)):
    new_tfs[i, :] = (
        proto_tfs_encoded[0, :] * (1 - weights[i])
        + proto_tfs_encoded[1, :] * weights[i]
    )

new_tfs_pred = generator_tf.predict(new_tfs, batch_size=1)
new_tfs_function = [
    tf_prediction(new_tfs_pred[i, ...]) for i in range(new_tfs_pred.shape[0])
]

print(new_tfs.shape, new_tfs_pred.shape)
print(new_tfs_function)

sent_encoded = encoder.predict([proto_tfs, proto_dist])

# print(sent_encoded)
dist_prediction = pd.DataFrame(generator_dist.predict(new_tfs, batch_size=1))
for i in range(dist_prediction.shape[1]):
    dist_prediction[i] = rescale(
        dist_prediction[i], from_range=[0, 1], to_range=[-11, 11]
    )

names = ["step " + str(i + 1) for i in range(dist_prediction.shape[0] - 2)]
names.insert(0, "start")
names.append("end")
dist_prediction.index = names
dist_prediction.columns = np.arange(0.1, 1, 0.1)
dist_prediction_scaled = min_max_scale(dist_prediction.values)
dist_prediction = pd.DataFrame(
    dist_prediction_scaled, index=dist_prediction.index, columns=dist_prediction.columns
)
dist_prediction["name"] = dist_prediction.index

print(dist_prediction)
# dist_prediction_scaled = min_max_scale(dist_prediction.values)
# print(dist_prediction_scaled)
# fig, axes = plt.subplots()
# j = 0
# for i in dist_prediction.index:
#     axes.plot(np.arange(0.1, 1, 0.1), dist_prediction_scaled[j, :], label=i)
#     j += 1
# axes.legend()
# axes.set_ylabel("Cumulative probability")
# axes.set_xlabel("Scaled parameter values")

# plt.show()
# # dist_prediction[:, -[1, 11]] = min_max_scale(dist_prediction[:, -[1, 11]])

plot_data = pd.melt(
    dist_prediction, id_vars=["name"], var_name="variable", value_name="value"
)
plot_data["variable"] = pd.to_numeric(plot_data["variable"])

import seaborn as sns
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(8, 12))
sns.lineplot(
    x="value", y="variable", hue="name", data=plot_data, ax=ax, palette="RdYlGn_r"
)
plt.setp(ax.get_legend().get_texts(), fontsize="16")  # for legend text
plt.setp(ax.get_legend().get_title(), fontsize="0")  # for legend title
ax.set_xlabel("Scaled parameter values", size=16)
ax.set_ylabel("Cumulative probability", size=16)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.tick_params(axis="both", which="major", labelsize=14)
plt.savefig(data_dir / "walk_through_function_space.png", bbox_inches="tight")
plt.show()


# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(3, 12))

# Define the colormap with 11 distinct colors
cmap = plt.get_cmap("RdYlGn_r", 11)

# Create a colorbar
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="vertical")

# Define the labels
labels = dist_prediction.index.tolist()

# Set the colorbar labels
cb1.ax.yaxis.set_ticks_position("left")
cb1.set_ticks(np.linspace(0, 1, len(labels)))
cb1.set_ticklabels(labels)
cb1.ax.tick_params(labelsize=22)

# Define the labels for the right side
labels_right = new_tfs_function
# Add the labels for the right side
for i, label in enumerate(labels_right):
    cb1.ax.text(
        1.01,
        cb1.get_ticks()[i] + 0.01,
        label,
        va="top",
        transform=cb1.ax.transAxes,
        size=22,
    )

plt.savefig(data_dir / "walk_through_function_space_functions.png", bbox_inches="tight")
plt.show()
