import os

import tensorflow as tf

import config
import dataset
import models
import tqdm
import metrics
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

# TODO SEED 맞춰주기?
# Learning Rate Scheduler 사용?
# Model Checkpoint는 optimizer까지?

# Compatible with tensorflow backend
def sigmoid(x):
    return 1 / (1 + tf.exp(-x))


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = sigmoid(y_pred)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_mean(alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) - tf.reduce_mean(
            (1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0))

    return focal_loss_fixed


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=200, decay_rate=0.96, staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')


def keras_train():  # TODO custom loop...
    tabnet_add_dense = models.tabnet_model()
    train_dataset = dataset.make_dataset('train_1.csv', config.COLUMNS, config.BATCH_SIZE,
                                         onehot=True,
                                         shuffle=True,
                                         train=True)  # MapDataset (512, 676)
    validation_dataset = dataset.make_dataset('validation_1.csv', config.COLUMNS, config.BATCH_SIZE,
                                              onehot=True,
                                              shuffle=True,
                                              train=True)  # MapDataset (512, 676)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    dt_value = datetime.datetime.now()
    filename = 'checkpoint/checkpoint-epoch-{epoch:04d}.ckpt'  # TODO F1 score를 넣지는 못할까?
    checkpoint = ModelCheckpoint(filename,  # file명을 지정합니다
                                 verbose=1,
                                 period=1,
                                 save_weights_only=True
                                 )
    tabnet_add_dense.compile(
        loss=focal_loss(),
        #loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=optimizer,
        metrics=['accuracy', metrics.f1_score]
    )
    tabnet_add_dense.load_weights("checkpoint/checkpoint-epoch-0017.ckpt")
    tabnet_add_dense.fit(train_dataset, validation_data=validation_dataset,
                         epochs=100000, verbose=1,
                         callbacks=[tensorboard_callback, checkpoint],
                         steps_per_epoch=200,
                         validation_steps=3)

    return tabnet_add_dense


def custom_train():  # using custom loop
    pass
    # for epoch in range(config.EPOCHS):
    #     for images, labels in tqdm(train_ds):
    #         train_step(images, labels)
    #
    #     for test_images, test_labels in tqdm(test_ds):
    #         test_step(test_images, test_labels)
    #
    #     template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
    #     print(template.format(epoch + 1,
    #                           train_loss.result(),
    #                           train_accuracy.result() * 100,
    #                           test_loss.result(),
    #                           test_accuracy.result() * 100))


def load_and_train():
    pass


if __name__ == '__main__':
    trained_model = keras_train()
