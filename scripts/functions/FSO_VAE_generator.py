import pathlib
import re
import pandas as pd
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    Reshape,
    Lambda,
    Cropping2D,
)
from tensorflow.keras.layers import (
    Activation,
    Dropout,
    Dense,
    Flatten,
    Input,
    UpSampling2D,
    Concatenate,
    Embedding,
)
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

data_dir = pathlib.Path(__file__).parent.parent.parent.resolve()

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

# %% Model architecture for loading the model
# Model variables ------------------------------------------------------------------------
# variables for model architecture
max_sent_length = generator_data.shape[1]
embedding_length = 5
emb_input_dim = index2word.shape[1] + 1
distribution_dim = 9
input_cols = embedding_length
input_rows = max_sent_length
intermediate_dim = 156
latent_dim = 6
decoder_dense_dim = 20
epsilon_std = 1.0

# variables for training
epochs = 200
batch_size = 1000
kl_weight = 100
dist_weight = 1000

# def vae_loss(y_true, y_pred):
#     xent_loss = K.mean(tf.keras.backend.sparse_categorical_crossentropy(y_true, y_pred))
#     kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
#     # xent_loss = K.mean(xent_loss)
#     # kl_loss = K.mean(kl_loss)

#     return xent_loss + kl_loss * kl_weight


def vae_loss(y_true, y_pred):
    xent_loss = K.sum(tf.keras.backend.sparse_categorical_crossentropy(y_true, y_pred))
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    xent_loss = K.mean(xent_loss)
    kl_loss = K.mean(kl_loss)
    return K.mean(xent_loss + kl_loss * kl_weight)


def dist_MSE(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred) * dist_weight


losses = {
    "TF_output": vae_loss,
    "dist_output": dist_MSE,
}

# CNN encoder ----------------------------------------------------------------------------
model_input = Input(shape=(max_sent_length, None), name="tf_input")
embedding = Embedding(
    input_dim=emb_input_dim,
    output_dim=embedding_length,
    input_length=max_sent_length,
    trainable=True,
)(model_input)
embedding = tf.keras.layers.Reshape((input_rows, input_cols, 1))(embedding)
x = Conv2D(32, (3, 3), activation="tanh")(embedding)
x = MaxPooling2D(pool_size=(1, input_cols - 2))(x)
column1 = Flatten()(x)

x = Conv2D(16, (4, 4), activation="tanh")(embedding)
x = MaxPooling2D(pool_size=(1, input_cols - 3))(x)
column2 = Flatten()(x)

x = Conv2D(16, (5, 5), activation="tanh")(embedding)
x = MaxPooling2D(pool_size=(1, input_cols - 4))(x)
column3 = Flatten()(x)

# Latent space
# cnn_encoding = tf.keras.layers.concatenate([column1, column2, column3])
aggregated_space = tf.keras.layers.concatenate([column1, column2, column3])
aggregated_space = Dense(453, activation="tanh")(aggregated_space)
cnn_encoding = Dense(intermediate_dim, activation="relu")(aggregated_space)

# autoencoder = tf.keras.models.Model(model_input, cnn_encoding)

# print(autoencoder.summary())

# Distribution properties encoding -------------------------------------------------------
d = Input(distribution_dim, name="dist_input")
x = Dense(80, activation="selu")(d)
x = Dense(40, activation="selu")(x)
dist_encoder = Dense(20, activation="selu")(x)

cnn_encoding = tf.keras.layers.concatenate(
    [cnn_encoding, dist_encoder], name="encoder_output"
)

# VAE sampling encoded space -------------------------------------------------------------
z_mean = Dense(latent_dim, activation="linear")(cnn_encoding)
z_log_var = Dense(latent_dim, activation="linear")(cnn_encoding)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)[0], latent_dim), mean=0.0, stddev=epsilon_std
    )
    return z_mean + K.exp(z_log_var / 2) * epsilon


z = Lambda(sampling)([z_mean, z_log_var])

for i in range(input_rows):
    globals()[f"decoder0{i}"] = Dense(latent_dim, activation="selu")

for i in range(input_rows):
    decoder0 = globals()[f"decoder0{i}"]
    tmp_layer = decoder0(z)
    tmp_layer = Reshape((-1, latent_dim))(tmp_layer)
    globals()[f"decoder0Z{i}"] = tmp_layer

decoder2 = tf.keras.layers.Bidirectional(
    layer=tf.keras.layers.LSTM(200, return_sequences=True), merge_mode="sum"
)
decoder3 = Dense(emb_input_dim, activation="softmax", name="TF_output")
decoded_list = []
for i in range(input_rows):
    decoded_list.append(globals()[f"decoder0Z{i}"])
decoded = decoder2(tf.keras.layers.concatenate(decoded_list, axis=1))
decoded = decoder3(decoded)

dist_decoder1 = Dense(80, activation="selu")
dist_decoder2 = Dense(40, activation="selu")
dist_decoder3 = Dense(20, activation="selu")
dist_decoder4 = Dense(distribution_dim, name="dist_output")

dist_decoded = dist_decoder4(dist_decoder3(dist_decoder2(dist_decoder1(z))))

autoencoder = tf.keras.models.Model([model_input, d], [decoded, dist_decoded])
autoencoder.compile(optimizer="Adam", loss=losses)
autoencoder.load_weights(
    data_dir.joinpath("VAE Model/training/FSO_VAE-weights-ep200-bs1000-v1.h5")
)

encoder = tf.keras.models.Model([model_input, d], z)

decoder_input = Input(shape=latent_dim)
for i in range(input_rows):
    decoder0 = globals()[f"decoder0{i}"]
    tmp_layer = decoder0(decoder_input)
    tmp_layer = Reshape((-1, latent_dim))(tmp_layer)
    globals()[f"decoder0Z{i}"] = tmp_layer

decoded_list = []
for i in range(input_rows):
    decoded_list.append(globals()[f"decoder0Z{i}"])
decoded = decoder2(tf.keras.layers.concatenate(decoded_list, axis=1))
decoded_g = decoder3(decoded)

x = dist_decoder1(decoder_input)
x = dist_decoder2(x)
x = dist_decoder3(x)
dist_decoded_g = dist_decoder4(x)

generator_tf = tf.keras.models.Model(decoder_input, decoded_g)
generator_dist = tf.keras.models.Model(decoder_input, dist_decoded_g)


def encoder_decoder():
    return (encoder, generator_tf, generator_dist)


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
    for i in ["silt", "sand", "clay", "OC", "BD", "pH", "BL"]:
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


def tf_generator(point):
    index_prob_prediction = generator_tf.predict(point, batch_size=1)
    point_tf = np.empty(index_prob_prediction.shape[0]).astype(np.str_)
    # print(tf_prediction(index_prob_prediction[0, :]))
    # point_tf[0] = "1-1/-1-1)))"
    # print(point_tf)
    for i in range(index_prob_prediction.shape[0]):
        point_tf[i] = tf_prediction(index_prob_prediction[i, :])
    # print(point_tf)
    for i in range(5):
        for ij, v in np.ndenumerate(point_tf):
            point_tf[ij] = re.sub("--", "+", point_tf[ij])
            point_tf[ij] = re.sub("\++", "+", point_tf[ij])
        # for j in range(point_tf.shape[0]):
        #     point_tf[j, :] = re.sub("--", "-", point_tf[j, :])
        #     point_tf[j, :] = re.sub("++", "-", point_tf[j, :])
    return point_tf


def tf_predict(point):
    return generator_tf.predict(point, batch_size=1)
