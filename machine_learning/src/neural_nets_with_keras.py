import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras

"""
算法：Keras 实现回归任务与分类任务
"""


class FashionMnist:
    def __init__(self):
        return

    @staticmethod
    def pipline():

        # 获取数据
        fashion_mnist = keras.datasets.fashion_mnist
        (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
        assert x_train_full.shape == (60000, 28, 28)

        # 分割数据以及数据归一化
        sample_valid = 5000
        x_valid, x_train = x_train_full[:sample_valid] / 255., x_train_full[sample_valid:] / 255.
        y_valid, y_train = y_train_full[:sample_valid], y_train_full[sample_valid:]
        x_test = x_test / 255.
        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        assert class_names[y_train[0]] == "Coat"
        assert x_valid.shape == (sample_valid, 28, 28)
        assert x_test.shape == (10000, 28, 28)

        # 构建模型网络层
        method = "add"
        if method == "add":  # "api/add"
            model = keras.models.Sequential()
            model.add(keras.layers.Flatten(input_shape=[28, 28]))  # 输入层
            model.add(keras.layers.Dense(300, activation="relu"))  # 中间层节点单元数量
            model.add(keras.layers.Dense(100, activation="relu"))
            model.add(keras.layers.Dense(10, activation="softmax"))  # 输出层多分类
            keras.backend.clear_session()
        else:
            model = keras.models.Sequential([
                keras.layers.Flatten(input_shape=[28, 28]),
                keras.layers.Dense(300, activation="relu"),
                keras.layers.Dense(100, activation="relu"),
                keras.layers.Dense(10, activation="softmax")
            ])
        print(model.summary())
        assert model.layers[1].name == "dense"
        assert model.layers[1].get_weights()[0].shape == (28 * 28, 300)
        np.random.seed(42)
        tf.random.set_seed(42)

        # 设置梯度下降损失计算与评估指标
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer="sgd",
                      metrics=["accuracy"])

        # 模型训练
        history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))
        assert sorted(list(history.history.keys())) == ['accuracy', 'loss', 'val_accuracy', 'val_loss']

        # 模型评估
        test_loss, test_accuracy = model.evaluate(x_test, y_test)

        y_prob = model.predict(x_test)  # 概率
        y_pre = np.argmax(y_prob, axis=-1)  # 类别

        # 准确率计算
        print(sum(y_pre[i] == y_test[i] for i in range(y_test.shape[0]))/y_test.shape[0], test_accuracy)
        return


class TestGeneral(unittest.TestCase):

    def test_fashion_mnist(self):
        FashionMnist().pipline()
        return


if __name__ == '__main__':
    unittest.main()

