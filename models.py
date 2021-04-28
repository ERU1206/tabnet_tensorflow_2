import tabnet
import tensorflow as tf
import doctest

"""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


model = tabnet.TabNet(feature_columns=None,
                        num_features = 676,
                        feature_dim=256,
                        output_dim=64,
                        num_decision_steps=5,
                        relaxation_factor=1.5,
                        sparsity_coefficient=1e-5,
                        norm_type='batch',
                        batch_momentum=0.98,
                        virtual_batch_size=None,
                        num_groups=2,
                        epsilon=1e-5)

def tabnet_model():
    models = tf.keras.Sequential([
        model,
        tf.keras.layers.Dense(1),
    ])
    return models

if __name__ == '__main__':
    # TODO graph 확인하기
    models = tabnet_model()
    tensor = tf.random.normal([1,676])
    print(model(tensor))