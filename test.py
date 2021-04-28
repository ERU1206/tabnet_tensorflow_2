import models
import numpy as np
import pandas as pd
import seaborn as sns
import vaex
import matplotlib.pyplot as plt

import models
import dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

checkpoint_path = "/home/ubuntu/storage1/leeloo/checkpoint/checkmate.ckpt"
test_data_dir = '/home/ubuntu/data/consulting/data/test_data_consulting.csv'
# TODO 모델 불러오기


def test(checkpoint_path, test_data_dir):
    test_data = vaex.open(test_data_dir, convert=True).to_pandas_df()
    probs = calculate_probability(test_data, checkpoint_path)
    plot(probs)
    form = summit_form(probs, test_data)
    return form

def calculate_probability(test_data, checkpoint_dir):
    model = models.tabnet_model()
    model.load_weights(checkpoint_path)

    batch_size = 100000
    dataset_size = test_data.shape[0]
    n_batchs = test_data.shape[0] // batch_size + 1
    batch_idx = 0
    probs = []
    for chunk in np.array_split(test_data, n_batchs):
        print(f"Process batch: {batch_idx}")
        print(dataset.preprocess_layer(chunk))
        probs.append(model.predict(chunk))
        #     print(model.predict(chunk))
        batch_idx += 1

    probs = np.concatenate(probs, axis=0)
    return probs

def plot(probs):
    sns.displot(probs, bins = 30, kde = False)
    plt.show()

def summit_form(prob, test_data):
    data = test_data[["order_id", "product_id"]]  # ,"label"
    cut_off = 0.5
    pred_reorder = [int(logit >= cut_off) for logit in prob]

    data = data[['order_id', 'product_id']]
    data.reset_index(drop=True, inplace=True)
    data.loc[:, 'pred'] = pred_reorder

    # reorder data
    reoder_pred_data = data[data['pred'] == 1].reset_index(drop=True)
    reoder_pred_data.loc[:, 'product_id'] = reoder_pred_data['product_id'].astype(str)
    result = reoder_pred_data.groupby('order_id')['product_id'].apply(lambda x: ' '.join(x)).reset_index()

    # none data
    no_reorder_id = list(set(data['order_id']) - set(reoder_pred_data['order_id']))
    no_reorder_result = pd.DataFrame({'order_id': no_reorder_id, 'product_id': 'None'})

    final_result = pd.concat([result, no_reorder_result], axis=0, ignore_index=True)
    final_result.rename(columns={'product_id': 'products'}, inplace=True)
    final_result['order_id'] = final_result['order_id'].astype('int')

    return final_result.sort_values('order_id').reset_index(drop=True)

if __name__ == '__main__':
    test(checkpoint_path, test_data_dir).to_csv('./submit.csv', index = False)
    print("쨔란")