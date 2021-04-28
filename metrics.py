import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

metric = tfa.metrics.F1Score(num_classes=1, threshold=0.5)


metric.update_state(y_true, y_pred)
result = metric.result()