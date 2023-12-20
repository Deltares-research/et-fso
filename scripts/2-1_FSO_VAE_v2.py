import os
import pathlib
import json
import pandas as pd
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Lambda, Cropping2D
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, Input, UpSampling2D, Concatenate, Embedding
from tensorflow.keras import backend as K

tf.keras.utils.disable_interactive_logging()

# Paths
data_dir = pathlib.Path("/gpfs/home4/aweerts/Data")
train_data_dir = pathlib.Path("/gpfs/home4/aweerts/Data/tfrecords_train")
val_data_dir = pathlib.Path("/gpfs/home4/aweerts/Data/tfrecords_val")
model_dir = pathlib.Path("/gpfs/home4/aweerts/VAE Model")

index2word = pd.read_feather(data_dir.joinpath(
    "index2word_10numerics.feather")).iloc[0].astype(str).to_list()

generator_data = pd.read_feather(data_dir.joinpath(
    "generator_data_simple_10numerics_wrecalc_allBasins_no_extremes.feather"))
index2word = pd.read_feather(data_dir.joinpath(
    "index2word_10numerics.feather")).iloc[0].astype(str).to_list()
tf_dist = pd.read_feather(data_dir.joinpath(
    "functions_simple_10_numerics_Distribution_indiv_scale_wrecalc_allBasins_no_extremes.feather"
))

def get_files(data_path):
    files = tf.io.gfile.glob(data_path + "/" + "*.tfrecords")
    return files

def get_dataset(files):
    """return a tfrecord dataset with all tfrecord files"""
    dataset =  tf.data.TFRecordDataset(files)
    dataset = dataset.map(tf_parse)
    return dataset

def tf_parse(eg):
    """parse an example (or batch of examples, not quite sure...)"""

    # here we re-specify our format
    # you can also infer the format from the data using tf.train.Example.FromString
    # but that did not work
    example = tf.io.parse_example(
        eg[tf.newaxis],
        {
            'x': tf.io.FixedLenFeature([], tf.string),
            'y': tf.io.FixedLenFeature([], tf.string),
            'dist': tf.io.FixedLenFeature([], tf.string),
        },
    )
    x_tf = tf.io.parse_tensor(example["x"][0], out_type="int32")
    y_tf = tf.io.parse_tensor(example["y"][0], out_type="int32")
    dist = tf.io.parse_tensor(example["dist"][0], out_type="float64")

    x_tf = tf.cast(tf.ensure_shape(x_tf, (32, None)), tf.float64)
    y_tf = tf.cast(tf.ensure_shape(y_tf, (32, None)), tf.float64)
    dist = tf.ensure_shape(dist, 9)
    return (x_tf, dist), (y_tf, dist)

# Scale distribution values
def min_max_scale(x):
    return np.round((x - np.min(x)) / (np.max(x) - np.min(x)), 11)

to_drop = ["transfer_function", "min", "max"]
dist_scaled = tf_dist.copy().drop(to_drop, axis=1)
for name in dist_scaled.columns:
    dist_scaled[name] = min_max_scale(dist_scaled[name])

# Train/val split
np.random.seed(0)
train_ind = np.random.choice(tf_dist.index, int(0.8*tf_dist.shape[0]), replace=False)
# np.save("train_ind_no_extremes.npy", train_ind)

x_train_tf = np.expand_dims(generator_data.iloc[train_ind].values, axis=-1)
x_val_tf = np.expand_dims(generator_data.drop(train_ind).values, axis=-1)

y_train_tf = np.expand_dims(generator_data.iloc[train_ind].values, axis=-1)
y_val_tf = np.expand_dims(generator_data.drop(train_ind).values, axis=-1)


train_dist = dist_scaled.iloc[train_ind].values
val_dist = dist_scaled.drop(train_ind).values



# Function from index to words to sentences

# Model variables ------------------------------------------------------------------------
# variables for model architecture
max_sent_length = 32
embedding_length = 5
emb_input_dim = len(index2word) + 1
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

# Load training and validation data
train_files = get_files(str(train_data_dir))
val_files  = get_files(str(val_data_dir))

train_dataset = get_dataset(train_files).shuffle(1).batch(batch_size, drop_remainder=True)
val_dataset = get_dataset(val_files).shuffle(1).batch(batch_size, drop_remainder=True)

total_samples = train_dist.shape[0]
steps_per_epoch = total_samples // batch_size

x_train_tf = x_train_tf[:steps_per_epoch * batch_size, ...]
y_train_tf = y_train_tf[:steps_per_epoch * batch_size, ...]
train_dist = train_dist[:steps_per_epoch * batch_size, ...]

total_samples = val_dist.shape[0]
steps_per_epoch = total_samples // batch_size

x_val_tf = x_val_tf[:steps_per_epoch * batch_size, ...]
y_val_tf = y_val_tf[:steps_per_epoch * batch_size, ...]
val_dist = val_dist[:steps_per_epoch * batch_size, ...]

# Create VAE (using a class, since it did not work otherwise due to the vae loss calling layer outputs in a session)
class VAE:
    def __init__(self):
        self.model = self.get_model()
        losses = {
            "TF_output": self.vae_loss,
            "dist_output": self.dist_MSE,
        }
        self.model.compile(optimizer="adam", loss=losses)

    def get_model(self):
        model_input = Input(shape = (max_sent_length, None), name = "tf_input")
        embedding = Embedding(input_dim = emb_input_dim,
                        output_dim = embedding_length,
                        input_length = max_sent_length,
                        trainable = True)(model_input)
        embedding = tf.keras.layers.Reshape((input_rows, input_cols, 1))(embedding)
        x = Conv2D(32, (3, 3), activation='tanh')(embedding)
        x = MaxPooling2D(pool_size = (1, input_cols - 2))(x)
        column1 = Flatten()(x)

        x = Conv2D(16, (4, 4), activation='tanh')(embedding)
        x = MaxPooling2D(pool_size = (1, input_cols - 3))(x)
        column2 = Flatten()(x)

        x = Conv2D(16, (5, 5), activation='tanh')(embedding)
        x = MaxPooling2D(pool_size = (1, input_cols - 4))(x)
        column3 = Flatten()(x)

        # Latent space
        aggregated_space = tf.keras.layers.concatenate([column1, column2, column3])
        aggregated_space = Dense(453, activation="tanh")(aggregated_space)
        cnn_encoding = Dense(intermediate_dim, activation='relu')(aggregated_space)

        # Distribution properties encoding -------------------------------------------------------
        d = Input(distribution_dim, name='dist_input')
        x = Dense(80, activation='selu')(d)
        x = Dense(40, activation='selu')(x)
        dist_encoder = Dense(20, activation='selu')(x)

        cnn_encoding = tf.keras.layers.concatenate([cnn_encoding, dist_encoder], name='encoder_output')

        # VAE sampling encoded space -------------------------------------------------------------
        self.z_mean = Dense(latent_dim, activation='linear', name='z_mean')(cnn_encoding)
        self.z_log_var = Dense(latent_dim, activation='linear', name='z_log_var')(cnn_encoding)

        z = Lambda(self.sampling)([self.z_mean, self.z_log_var])

        decoder0_dict = {}
        decoder0Z_dict = {}
        for i in range(input_rows):
            decoder0_dict[f"decoder0{i}"] = Dense(latent_dim, activation='selu')

        for i in range(input_rows):
            decoder0 = decoder0_dict[f"decoder0{i}"]
            tmp_layer = decoder0(z)
            tmp_layer = Reshape((-1, latent_dim))(tmp_layer)
            decoder0Z_dict[f"decoder0Z{i}"] = tmp_layer

        decoder2 = tf.keras.layers.Bidirectional(layer=tf.keras.layers.LSTM(200, return_sequences=True),
                                                    merge_mode="sum")
        decoder3 = Dense(emb_input_dim, activation="softmax", name='TF_output')
        decoded_list = []
        for i in range(input_rows):
            decoded_list.append(decoder0Z_dict[f"decoder0Z{i}"])
        decoded = decoder2(tf.keras.layers.concatenate(decoded_list, axis=1))
        decoded = decoder3(decoded)

        x = Dense(80, activation='selu')(z)
        x = Dense(40, activation='selu')(x)
        x = Dense(20, activation='selu')(x)
        dist_decoded = Dense(distribution_dim, name='dist_output')(x)

        autoencoder = tf.keras.models.Model([model_input, d], [decoded, dist_decoded])

        return autoencoder

    def sampling(self, args):
        self.z_mean, self.z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(self.z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
        return self.z_mean + K.exp(self.z_log_var / 2) * epsilon

    def vae_loss(self, y_true, y_pred):
        z_l_v = self.z_log_var
        z_m = self.z_mean
        xent_loss = K.mean(tf.keras.backend.sparse_categorical_crossentropy(y_true, y_pred), axis=-1)
        kl_loss = -0.5 * K.mean(1 + z_l_v - K.square(z_m) - K.exp(z_l_v), axis=-1)
        # xent_loss = K.mean(xent_loss)
        # kl_loss = K.mean(kl_loss)
        return xent_loss + kl_loss * kl_weight

    def dist_MSE(self, y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred) * dist_weight
  
# Train model in a session
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(devices = gpus, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
with strategy.scope():
    vae = VAE()

    csv_logger = tf.keras.callbacks.CSVLogger(
        filename = model_dir.joinpath(f"training/FSO_VAE-ep{epochs}-bs{batch_size}-v3.csv")
    )

    history = vae.model.fit(x=train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=[csv_logger])
    
    vae.model.save_weights(model_dir.joinpath(f"training/FSO_VAE-weights-ep{epochs}-bs{batch_size}-v3.h5"))


