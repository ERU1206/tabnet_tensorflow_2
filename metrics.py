import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
# CrossEntropy
loss1 = tf.keras.losses.BinaryCrossentropy(from_logits=True)

y_true = np.array([1.,
                   1.,
                   1.], np.float32)
y_pred = np.array([0.2,
                   0.2,
                   0.6], np.float32)

def recall_m(y_true, y_pred, alpha):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(tf.where((K.clip(y_true * y_pred, 0.0, 1.0))>alpha,1.0,0.0))
    all_positives = K.sum(tf.where((K.clip(y_pred, 0.0, 1.0))>alpha,1.0,0.0))

    recall = true_positives / (all_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred, alpha):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(tf.where((K.clip(y_true * y_pred, 0.0, 1.0))>alpha,1.0,0.0))

    predicted_positives = K.sum(tf.where((K.clip(y_pred, 0.0, 1.0))>alpha,1.0,0.0))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred, alpha = 0.1):
    precision = precision_m(y_true, y_pred, alpha)
    recall = recall_m(y_true, y_pred, alpha)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

f1_score(y_true, y_pred)