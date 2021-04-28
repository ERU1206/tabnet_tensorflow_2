import tensorflow_addons as tfa

metric = tfa.metrics.F1Score(num_classes=1, threshold=0.5)

metric

metric.update_state(y_true, y_pred)
result = metric.result()