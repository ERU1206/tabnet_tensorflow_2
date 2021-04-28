import models

tabnet_add_dense = models.tabnet_model() # saved file 불러오기

tabnet = tabnet_add_dense.layers(0)

# 쌓기