import os

import tensorflow as tf

import config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def from_csv_dataset(data_dir, label_name, batch_size, train=True):
    if train:
        dataset = tf.data.experimental.make_csv_dataset(data_dir,
                                                        label_name=label_name,
                                                        na_value='nan',
                                                        batch_size=batch_size,
                                                        num_epochs=1)
        return dataset
    else:
        dataset = tf.data.experimental.make_csv_dataset(data_dir,
                                                        na_value='nan',
                                                        batch_size=batch_size,
                                                        num_epochs=1)  # num_epoch의 의미는?
        return dataset


class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels


class PackCategoryFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        for i in self.names:
            features[i] = tf.cast(features[i], tf.int32)

        return features, labels


class TestPackNumericFeatures(PackNumericFeatures):
    def __init__(self, names):
        super(TestPackNumericFeatures, self).__init__(names)

    def __call__(self, features):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features


class TestPackCategoryFeatures(PackCategoryFeatures):
    def __init__(self, names):
        super(TestPackCategoryFeatures, self).__init__(names)

    def __call__(self, features):
        for i in self.names:
            features[i] = tf.cast(features[i], tf.int32)

        return features


def make_dataset(directory, columns, make_batch_size, shuffle=True, train=True):
    if train:
        dataset = from_csv_dataset(directory, columns["LABEL"], make_batch_size, train=True)
        packed_train_data = dataset.map(
            PackNumericFeatures(columns["NUMERIC_FEATURES"]))

        packed_train_data = packed_train_data.map(
            PackCategoryFeatures(columns['CATEGORY_FEATURES']))

        processing_layer = preprocess_layer(columns)
        packed_train_data = packed_train_data.map(
            lambda x, y: (processing_layer(x), y))
        if shuffle:
            packed_train_data.shuffle(10000, reshuffle_each_iteration=True)
        return packed_train_data
    elif not train:
        dataset = from_csv_dataset(directory, None, make_batch_size, train=False)
        packed_test_data = dataset.map(
            TestPackNumericFeatures(columns["NUMERIC_FEATURES"]))

        packed_test_data = packed_test_data.map(
            TestPackCategoryFeatures(columns['CATEGORY_FEATURES']))

        processing_layer = preprocess_layer(columns)
        packed_test_data = packed_test_data.map(
            lambda x: processing_layer(x))
        if shuffle:
            packed_test_data.shuffle(10000, reshuffle_each_iteration=True)
        return packed_test_data
    else:
        raise ValueError('train or test')


def preprocess_layer(columns):
    numeric_columns = [
        tf.feature_column.numeric_column(
            'numeric',
            shape=[len(columns["NUMERIC_FEATURES"])]
        )]

    categorical_columns = [
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_identity(
                key=key,
                num_buckets=50, default_value=0))
        for key in columns['CATEGORY_FEATURES']]

    processing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)
    return processing_layer


if __name__ == '__main__':
    dataset = make_dataset(config.TRAIN_DIR, config.COLUMNS, config.BATCH_SIZE,
                           train=True)  # MapDataset (512, 676) # 얘 왜 안끝나?
