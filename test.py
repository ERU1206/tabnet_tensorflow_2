import models
import numpy as np
import pandas as pd
import seaborn as sns
import vaex
import matplotlib.pyplot as plt
import config
import models
import dataset
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def test(config):
    test_data = vaex.open(config.TEST_DIR, convert=True).to_pandas_df()

    probs = calculate_probability(config)
    # plot(probs)
    return summit_form(probs, test_data)


def calculate_probability(config):
    model = models.tabnet_model()
    model.load_weights(config.CHCK_PATH)

    test_dataset = dataset.make_dataset(config.TEST_DIR, config.COLUMNS, config.TEST_BATCH_SIZE,
                                        onehot=True, train=False)
    # test_dataset = dataset.make_dataset(config.TEST_DIR, config.COLUMNS, config.TEST_BATCH_SIZE, onehot=False, train=False)
    probs = np.empty((1, 1))
    for num, chunk in enumerate(test_dataset):
        print(f"Process batch: {num + 1}")  # to 48
        prob = model(chunk).numpy()
        probs = np.append(probs, prob, axis=0)
    return probs


def plot(probs):
    sns.displot(probs, bins=30, kde=False)
    plt.show()


def summit_form(prob, test_data):
    prob = prob[1:] #고쳐주기
    data = test_data[["order_id", "product_id"]]  # ,"label"
    cut_off = 0.5
    pred_reorder = [int(logit >= cut_off) for logit in prob]

    data = data[['order_id', 'product_id']]
    data.reset_index(drop=True, inplace=True)
    data.loc[:, 'pred'] = pred_reorder

    # reorder data
    reorder_pred_data = data[data['pred'] == 1].reset_index(drop=True)
    reorder_pred_data.loc[:, 'product_id'] = reorder_pred_data['product_id'].astype(str)
    result = reorder_pred_data.groupby('order_id')['product_id'].apply(lambda x: ' '.join(x)).reset_index()

    # none data
    no_reorder_id = list(set(data['order_id']) - set(reorder_pred_data['order_id']))
    no_reorder_result = pd.DataFrame({'order_id': no_reorder_id, 'product_id': 'None'})

    final_result = pd.concat([result, no_reorder_result], axis=0, ignore_index=True)
    final_result.rename(columns={'product_id': 'products'}, inplace=True)
    final_result['order_id'] = final_result['order_id'].astype('int')

    return final_result.sort_values('order_id').reset_index(drop=True)


if __name__ == '__main__':
    # test_dataset = dataset.make_dataset(config.TEST_DIR, config.COLUMNS, config.TEST_BATCH_SIZE, train=False)
    test(config).to_csv('./submit2.csv', index=False)
