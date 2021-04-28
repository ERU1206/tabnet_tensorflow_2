import tensorflow_addons as tfa
import tensorflow as tf

# CrossEntropy
loss1 = tf.keras.losses.BinaryCrossentropy(from_logits=True)

metric = tfa.metrics.F1Score(num_classes=1, threshold=0.5)

