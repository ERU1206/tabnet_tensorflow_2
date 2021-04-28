from sklearn.model_selection import KFold
import vaex
import pandas as pd
import config


def split(dir):
    data = vaex.open(dir, convert=True).to_pandas_df()

    cv = KFold(n_splits=5)
    i = 1
    for t, v in cv.split(data):
        print(f"{i}th split processing...")
        train = data.iloc[t]
        train.to_csv(f'./train_{i}.csv')
        validation = data.iloc[v]
        validation.to_csv(f'./train_{i}.csv')
        i += 1


if __name__ == '__main__':
    split(config.TRAIN_DIR)
