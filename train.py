import os

import tensorflow as tf

import config
import dataset
import models

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)


def train(): # TODO custom loop...
    tabnet_add_dense = models.tabnet_model()
    train_dataset = dataset.make_dataset(config.TRAIN_DIR, config.COLUMNS, config.BATCH_SIZE, shuffle = True, train = True)  # MapDataset (512, 676)
    tabnet_add_dense.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])
    tabnet_add_dense.fit(train_dataset, epochs = 5, verbose = 1)
    return tabnet_add_dense

if __name__ == '__main__':
    trained_model = train()