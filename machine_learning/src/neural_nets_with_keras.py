import unittest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    def pipline():  # 衣物图片多分类识别

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
        acc = sum(y_pre[i] == y_test[i] for i in range(y_test.shape[0]))/y_test.shape[0]
        assert abs(acc - test_accuracy) < 1e-2
        return


class CaliforniaHousing:
    def __init__(self):
        return

    @staticmethod
    def pipline():  # 房价回归预测

        # 读取数据并进行训练验证划分
        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full, random_state=42)

        # 标准归一化数据
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)

        np.random.seed(42)
        tf.random.set_seed(42)
        assert x_train.shape == (11610, 8)
        # 建立网络层模型与损失函数优化器
        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="relu", input_shape=x_train.shape[1:]),  # 输入层
            keras.layers.Dense(1)  # 输出层回归直接输出
        ])
        model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
        model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid))
        mse_test = model.evaluate(x_test, y_test)
        y_pre = model.predict(x_test)
        mse = sum((y_pre[i] - y_test[i])**2 for i in range(len(y_pre)))/len(y_pre)
        print(mse[0], mse_test)
        assert abs(mse[0] - mse_test) < 1e-2
        return


class TestGeneral(unittest.TestCase):

    def test_fashion_mnist(self):
        FashionMnist().pipline()
        return

    def test_california_mnist(self):
        CaliforniaHousing().pipline()
        return


if __name__ == '__main__':
    unittest.main()

