# %%
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import numpy as np
from utils import prepare_corpus, prepare_model, prepare_data
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt


# %%
df_train, df_val, df_test, emotions = prepare_corpus.load_corpus('emoDB', 0)
# %%
ds, num_segs = prepare_data.segment_audio_dataset(df_train)
ds_feature = prepare_data.feature_dataset(ds)
ds_feature = ds_feature.batch(64).cache().prefetch(tf.data.experimental.AUTOTUNE)
# %%
norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
norm_layer.adapt(ds_feature.map(lambda x, y, sw: x))
model = prepare_model.emotion_mono((300, 40, 1), 7, mel_norm_layer=norm_layer)
# %%
loss_obj = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(1e-4, clipvalue=1)
# %%