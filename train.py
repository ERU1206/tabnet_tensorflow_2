import os

import tensorflow as tf

import config
import dataset
import models
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)



# Compatible with tensorflow backend
def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = sigmoid(y_pred)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_mean(alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1))- tf.reduce_mean((1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0))
    return focal_loss_fixed

def keras_train(): # TODO custom loop...
    tabnet_add_dense = models.tabnet_model()
    train_dataset = dataset.make_dataset(config.TRAIN_DIR, config.COLUMNS, config.BATCH_SIZE, shuffle = True, train = True)  # MapDataset (512, 676)
    tabnet_add_dense.compile(
        loss=focal_loss(),
        # loss = tfa.losses.SigmoidFocalCrossEntropy(),
        optimizer='adam',
        metrics=['accuracy'])
    tabnet_add_dense.fit(train_dataset, epochs = 5, verbose = 1)
    return tabnet_add_dense


def custom_train(): #using custom loop
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

if __name__ == '__main__':
    trained_model = keras_train()