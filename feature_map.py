import models
import numpy as np
import vaex

checkpoint_path = "/home/ubuntu/storage1/leeloo/checkpoint/checkmate.ckpt"
test_data_dir = '/home/ubuntu/data/consulting/data/test_data_consulting.csv'
tabnet_add_dense = models.tabnet_model() # saved file 불러오기

tabnet = tabnet_add_dense.layers(0)

def test(checkpoint_path, test_data_dir):
    test_data = vaex.open(test_data_dir, convert=True).to_pandas_df()
    probs = calculate_probability(test_data, checkpoint_path)
    return probs

def calculate_probability(test_data, checkpoint_dir):
    model = models.tabnet_model()
    model.load_weights(checkpoint_dir)

    activate_model = model.layers(0)

    batch_size = 100000
    dataset_size = test_data.shape[0]
    n_batchs = test_data.shape[0] // batch_size + 1
    batch_idx = 0
    probs = []
    for chunk in np.array_split(test_data, n_batchs):
        print(f"Process batch: {batch_idx}")
        probs.append(model.predict(chunk))
        #     print(model.predict(chunk))
        batch_idx += 1

    probs = np.concatenate(probs, axis=0)
    return probs

if __name__ == '__main__':
    test(checkpoint_path, test_data_dir)