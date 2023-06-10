import logging
import math
import unittest
from functools import partial

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV


"""
算法：使用深度神经网络进行训练
"""


class ActivationFunction:
    def __init__(self):
        return

    @staticmethod
    def leaky_relu():

        def leaky_relu_activation(z, alpha=0.01):
            return np.maximum(alpha * z, z)

        assert leaky_relu_activation(1) == 1
        assert leaky_relu_activation(-1) == -0.01

        valid_sample = 5000
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        x_valid, x_train = x_train_full[:valid_sample], x_train_full[valid_sample:]
        y_valid, y_train = y_train_full[:valid_sample], y_train_full[valid_sample:]
        tf.random.set_seed(42)
        np.random.seed(42)

        # 构建序列网状模型
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(300, kernel_initializer="he_normal"),  # 初始化
            keras.layers.LeakyReLU(),  # 当前层激活函数
            keras.layers.Dense(100, kernel_initializer="he_normal"),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(10, activation="softmax")
        ])
        # 损失计算梯度函数与准确率
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                      metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=10,
                            validation_data=(x_valid, y_valid))
        test_loss, test_accuracy = model.evaluate(x_test, y_test)

        y_prob = model.predict(x_test)  # 概率
        y_pre = np.argmax(y_prob, axis=-1)  # 类别
        # 准确率计算
        acc = sum(y_pre[i] == y_test[i] for i in range(y_test.shape[0])) / y_test.shape[0]
        print(f"Check predict {acc}={test_accuracy}")
        assert abs(acc - test_accuracy) < 1e-2
        return

    @staticmethod
    def leaky_prelu():

        def leaky_prelu_activation(x, alpha=0.01):  # 等同于 leaky_prelu
            return np.maximum(0, x) + alpha*np.minimum(0, x)

        assert leaky_prelu_activation(1) == 1
        assert leaky_prelu_activation(-1) == -0.01

        valid_sample = 5000
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        x_valid, x_train = x_train_full[:valid_sample], x_train_full[valid_sample:]
        y_valid, y_train = y_train_full[:valid_sample], y_train_full[valid_sample:]
        tf.random.set_seed(42)
        np.random.seed(42)

        # 构建序列网状模型
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(300, kernel_initializer="he_normal"),
            keras.layers.PReLU(),
            keras.layers.Dense(100, kernel_initializer="he_normal"),
            keras.layers.PReLU(),
            keras.layers.Dense(10, activation="softmax")
        ])
        # 损失计算梯度函数与准确率
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                      metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=10,
                            validation_data=(x_valid, y_valid))
        test_loss, test_accuracy = model.evaluate(x_test, y_test)

        y_prob = model.predict(x_test)  # 概率
        y_pre = np.argmax(y_prob, axis=-1)  # 类别
        # 准确率计算
        acc = sum(y_pre[i] == y_test[i] for i in range(y_test.shape[0])) / y_test.shape[0]
        print(f"Check predict {acc}={test_accuracy}")
        assert abs(acc - test_accuracy) < 1e-2
        return

    @staticmethod
    def selu():

        valid_sample = 5000
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        x_valid, x_train = x_train_full[:valid_sample], x_train_full[valid_sample:]
        y_valid, y_train = y_train_full[:valid_sample], y_train_full[valid_sample:]
        tf.random.set_seed(42)
        np.random.seed(42)

        # 构建序列网状模型
        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=[28, 28]))
        model.add(keras.layers.Dense(300, activation="selu",
                                     kernel_initializer="lecun_normal"))
        for layer in range(99):
            model.add(keras.layers.Dense(100, activation="selu",
                                         kernel_initializer="lecun_normal"))
        model.add(keras.layers.Dense(10, activation="softmax"))
        # 需要将数据均值置为 0 偏差置为 1 
        pixel_means = x_train.mean(axis=0, keepdims=True)
        pixel_stds = x_train.std(axis=0, keepdims=True)
        x_train_scaled = (x_train - pixel_means) / pixel_stds
        x_valid_scaled = (x_valid - pixel_means) / pixel_stds
        x_test_scaled = (x_test - pixel_means) / pixel_stds
        # 损失计算梯度函数与准确率
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                      metrics=["accuracy"])
        model.fit(x_train_scaled, y_train, epochs=5,
                            validation_data=(x_valid_scaled, y_valid))
        test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)

        y_prob = model.predict(x_test_scaled)  # 概率
        y_pre = np.argmax(y_prob, axis=-1)  # 类别
        # 准确率计算
        acc = sum(y_pre[i] == y_test[i] for i in range(y_test.shape[0])) / y_test.shape[0]
        print(f"Check predict {acc}={test_accuracy}")
        assert abs(acc - test_accuracy) < 1e-2
        return


class BatchNormalization:
    def __init__(self):
        return

    @staticmethod
    def batch_normalization():

        valid_sample = 5000
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        x_valid, x_train = x_train_full[:valid_sample], x_train_full[valid_sample:]
        y_valid, y_train = y_train_full[:valid_sample], y_train_full[valid_sample:]
        tf.random.set_seed(42)
        np.random.seed(42)

        # 构建序列网状模型
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(300, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(100, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(10, activation="softmax")
        ])
        # 损失计算梯度函数与准确率
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                      metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=10,
                            validation_data=(x_valid, y_valid))
        test_loss, test_accuracy = model.evaluate(x_test, y_test)

        y_prob = model.predict(x_test)  # 概率
        y_pre = np.argmax(y_prob, axis=-1)  # 类别
        # 准确率计算
        acc = sum(y_pre[i] == y_test[i] for i in range(y_test.shape[0])) / y_test.shape[0]
        print(f"Check predict {acc}={test_accuracy}")
        assert abs(acc - test_accuracy) < 1e-2
        return


class ReusingPretrainedLayers:
    def __init__(self):
        return

    @staticmethod
    def reusing_pretrained_layers():
        # 使用预训练模型进行迁移
        valid_sample = 5000
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        x_valid, x_train = x_train_full[:valid_sample], x_train_full[valid_sample:]
        y_valid, y_train = y_train_full[:valid_sample], y_train_full[valid_sample:]
        tf.random.set_seed(42)
        np.random.seed(42)

        def split_dataset(x, y):
            y_5_or_6 = (y == 5) | (y == 6)  # sandals or shirts
            y_a = y[~y_5_or_6]
            y_a[y_a > 6] -= 2  # class indices 7, 8, 9 should be moved to 5, 6, 7
            y_b = (y[y_5_or_6] == 6).astype(np.float32)  # binary classification task: is it a shirt (class 6)
            return ((x[~y_5_or_6], y_a),
                    (x[y_5_or_6], y_b))

        def split_dataset2(x, y):
            y_4_or_5 = (y == 4) | (y == 5)  # sandals or shirts
            y_a = y[~y_4_or_5]
            y_a[y_a > 5] -= 2
            y_b = (y[y_4_or_5] == 5).astype(np.float32)  # binary classification task: is it a shirt (class 5)
            return ((x[~y_4_or_5], y_a),
                    (x[y_4_or_5], y_b))

        # 划分数据集分为 8 类和另外 2 类
        method = 1
        if method == 1:
            (x_train_a, y_train_a), (x_train_b, y_train_b) = split_dataset(x_train, y_train)
            (x_valid_a, y_valid_a), (x_valid_b, y_valid_b) = split_dataset(x_valid, y_valid)
            (x_test_a, y_test_a), (x_test_b, y_test_b) = split_dataset(x_test, y_test)
        else:
            (x_train_a, y_train_a), (x_train_b, y_train_b) = split_dataset2(x_train, y_train)
            (x_valid_a, y_valid_a), (x_valid_b, y_valid_b) = split_dataset2(x_valid, y_valid)
            (x_test_a, y_test_a), (x_test_b, y_test_b) = split_dataset2(x_test, y_test)

        # 先训练 A 模型
        tf.random.set_seed(42)
        np.random.seed(42)
        model_a = keras.models.Sequential()
        model_a.add(keras.layers.Flatten(input_shape=[28, 28]))
        for n_hidden in (300, 100, 50, 50, 50):
            model_a.add(keras.layers.Dense(n_hidden, activation="selu"))
        model_a.add(keras.layers.Dense(8, activation="softmax"))
        model_a.compile(loss="sparse_categorical_crossentropy",
                        optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                        metrics=["accuracy"])
        model_a.fit(x_train_a, y_train_a, epochs=20,
                    validation_data=(x_valid_a, y_valid_a))
        model_a.evaluate(x_test_a, y_test_a)
        model_a.save("./data/my_model_a.h5")

        # 再训练 B 模型
        model_b = keras.models.Sequential()
        model_b.add(keras.layers.Flatten(input_shape=[28, 28]))
        for n_hidden in (300, 100, 50, 50, 50):
            model_b.add(keras.layers.Dense(n_hidden, activation="selu"))
        model_b.add(keras.layers.Dense(1, activation="sigmoid"))
        model_b.compile(loss="binary_crossentropy",
                        optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                        metrics=["accuracy"])
        model_b.fit(x_train_b, y_train_b, epochs=20,
                              validation_data=(x_valid_b, y_valid_b))

        # 使用深拷贝赋值 A 模型的网络层与权重
        model_a = keras.models.load_model("./data/my_model_a.h5")
        model_a_clone = keras.models.clone_model(model_a)
        model_a_clone.set_weights(model_a.get_weights())
        # 修改最后一层为二分类输出
        model_b_on_a = keras.models.Sequential(model_a_clone.layers[:-1])
        model_b_on_a.add(keras.layers.Dense(1, activation="sigmoid"))

        # 基于模型 A 继续训练二分类模型 B
        model_b_on_a.compile(loss="binary_crossentropy",
                             optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                             metrics=["accuracy"])
        model_b_on_a.fit(x_train_b, y_train_b, epochs=16,
                                   validation_data=(x_valid_b, y_valid_b))
        loss_b, acc_b = model_b.evaluate(x_test_b, y_test_b)
        loss_b_on_a, acc_b_on_a = model_b_on_a.evaluate(x_test_b, y_test_b)
        print(f"loss_b={loss_b} acc_b={acc_b}")
        print(f"loss_b_on_a={loss_b_on_a} acc_b_on_a={acc_b_on_a}")
        return


class FasterOptimizers:
    def __init__(self):
        return

    @staticmethod
    def learning_rate_decay():
        # 学习率随epoch进行衰减
        valid_sample = 5000
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        x_valid, x_train = x_train_full[:valid_sample], x_train_full[valid_sample:]
        y_valid, y_train = y_train_full[:valid_sample], y_train_full[valid_sample:]
        pixel_means = x_train.mean(axis=0, keepdims=True)
        pixel_stds = x_train.std(axis=0, keepdims=True)
        x_train_scaled = (x_train - pixel_means) / pixel_stds
        x_valid_scaled = (x_valid - pixel_means) / pixel_stds
        x_test_scaled = (x_test - pixel_means) / pixel_stds
        tf.random.set_seed(42)
        np.random.seed(42)

        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, decay=1e-4)
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(10, activation="softmax")
        ])

        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        model.fit(x_train_scaled, y_train, epochs=25,
                  validation_data=(x_valid_scaled, y_valid))
        test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)

        y_prob = model.predict(x_test_scaled)  # 概率
        y_pre = np.argmax(y_prob, axis=-1)  # 类别
        # 准确率计算
        acc = sum(y_pre[i] == y_test[i] for i in range(y_test.shape[0])) / y_test.shape[0]
        print(f"Check predict {acc}={test_accuracy}")
        assert abs(acc - test_accuracy) < 1e-2
        return

    @staticmethod
    def exponential_scheduling():
        # 自定义学习率随epoch进行衰减
        valid_sample = 5000
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        x_valid, x_train = x_train_full[:valid_sample], x_train_full[valid_sample:]
        y_valid, y_train = y_train_full[:valid_sample], y_train_full[valid_sample:]
        pixel_means = x_train.mean(axis=0, keepdims=True)
        pixel_stds = x_train.std(axis=0, keepdims=True)
        x_train_scaled = (x_train - pixel_means) / pixel_stds
        x_valid_scaled = (x_valid - pixel_means) / pixel_stds
        x_test_scaled = (x_test - pixel_means) / pixel_stds
        tf.random.set_seed(42)
        np.random.seed(42)

        def exponential_decay(lr0, s):

            def exponential_decay_fn(epoch):
                return lr0 * 0.1 ** (epoch / s)

            return exponential_decay_fn

        # 自定义学习率衰减跟随 epoch
        lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay(lr0=0.01, s=20))
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
        model.fit(x_train_scaled, y_train, epochs=25,
                  validation_data=(x_valid_scaled, y_valid),
                  callbacks=[lr_scheduler])

        test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)

        y_prob = model.predict(x_test_scaled)  # 概率
        y_pre = np.argmax(y_prob, axis=-1)  # 类别
        # 准确率计算
        acc = sum(y_pre[i] == y_test[i] for i in range(y_test.shape[0])) / y_test.shape[0]
        print(f"Check predict {acc}={test_accuracy}")
        assert abs(acc - test_accuracy) < 1e-2
        return

    @staticmethod
    def exponential_scheduling_class():
        # 自定义学习率随每一次迭代进行衰减
        valid_sample = 5000
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        x_valid, x_train = x_train_full[:valid_sample], x_train_full[valid_sample:]
        y_valid, y_train = y_train_full[:valid_sample], y_train_full[valid_sample:]
        pixel_means = x_train.mean(axis=0, keepdims=True)
        pixel_stds = x_train.std(axis=0, keepdims=True)
        x_train_scaled = (x_train - pixel_means) / pixel_stds
        x_valid_scaled = (x_valid - pixel_means) / pixel_stds
        x_test_scaled = (x_test - pixel_means) / pixel_stds
        tf.random.set_seed(42)
        np.random.seed(42)

        class ExponentialDecay(keras.callbacks.Callback):
            def __init__(self, s=40000):
                super().__init__()
                self.s = s

            def on_batch_begin(self, batch, logs=None):
                # Note: the `batch` argument is reset at each epoch
                lr = keras.backend.get_value(self.model.optimizer.learning_rate)
                keras.backend.set_value(self.model.optimizer.learning_rate, lr * 0.1 ** (1 / self.s))

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                logs['lr'] = keras.backend.get_value(self.model.optimizer.learning_rate)

        # 自定义学习率衰减跟随
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(10, activation="softmax")
        ])
        lr0 = 0.01
        optimizer = keras.optimizers.Nadam(learning_rate=lr0)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        n_epochs = 25
        s = 20 * len(x_train) // 32  # number of steps in 20 epochs (batch size = 32)
        exp_decay = ExponentialDecay(s)
        model.fit(x_train_scaled, y_train, epochs=n_epochs,
                            validation_data=(x_valid_scaled, y_valid),
                            callbacks=[exp_decay])

        test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)

        y_prob = model.predict(x_test_scaled)  # 概率
        y_pre = np.argmax(y_prob, axis=-1)  # 类别
        # 准确率计算
        acc = sum(y_pre[i] == y_test[i] for i in range(y_test.shape[0])) / y_test.shape[0]
        print(f"Check predict {acc}={test_accuracy}")
        assert abs(acc - test_accuracy) < 1e-2
        return

    @staticmethod
    def piecewise_constant_scheduling():
        # 自定义学习率随epoch进行衰减
        valid_sample = 5000
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        x_valid, x_train = x_train_full[:valid_sample], x_train_full[valid_sample:]
        y_valid, y_train = y_train_full[:valid_sample], y_train_full[valid_sample:]
        pixel_means = x_train.mean(axis=0, keepdims=True)
        pixel_stds = x_train.std(axis=0, keepdims=True)
        x_train_scaled = (x_train - pixel_means) / pixel_stds
        x_valid_scaled = (x_valid - pixel_means) / pixel_stds
        x_test_scaled = (x_test - pixel_means) / pixel_stds
        tf.random.set_seed(42)
        np.random.seed(42)

        def piecewise_constant_fn(epoch):
            if epoch < 5:
                return 0.01
            elif epoch < 15:
                return 0.005
            else:
                return 0.001

        # 自定义学习率衰减跟随 epoch
        lr_scheduler = keras.callbacks.LearningRateScheduler(piecewise_constant_fn)
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
        model.fit(x_train_scaled, y_train, epochs=25,
                  validation_data=(x_valid_scaled, y_valid),
                  callbacks=[lr_scheduler])

        test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)

        y_prob = model.predict(x_test_scaled)  # 概率
        y_pre = np.argmax(y_prob, axis=-1)  # 类别
        # 准确率计算
        acc = sum(y_pre[i] == y_test[i] for i in range(y_test.shape[0])) / y_test.shape[0]
        print(f"Check predict {acc}={test_accuracy}")
        assert abs(acc - test_accuracy) < 1e-2
        return

    @staticmethod
    def performance_scheduling():
        # 自定义学习率随epoch进行衰减
        valid_sample = 5000
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        x_valid, x_train = x_train_full[:valid_sample], x_train_full[valid_sample:]
        y_valid, y_train = y_train_full[:valid_sample], y_train_full[valid_sample:]
        pixel_means = x_train.mean(axis=0, keepdims=True)
        pixel_stds = x_train.std(axis=0, keepdims=True)
        x_train_scaled = (x_train - pixel_means) / pixel_stds
        x_valid_scaled = (x_valid - pixel_means) / pixel_stds
        x_test_scaled = (x_test - pixel_means) / pixel_stds
        tf.random.set_seed(42)
        np.random.seed(42)

        # 自定义学习率衰减跟随 epoch
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(10, activation="softmax")
        ])
        optimizer = keras.optimizers.SGD(learning_rate=0.02, momentum=0.9)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        model.fit(x_train_scaled, y_train, epochs=25,
                  validation_data=(x_valid_scaled, y_valid),
                  callbacks=[lr_scheduler])

        test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)
        y_prob = model.predict(x_test_scaled)  # 概率
        y_pre = np.argmax(y_prob, axis=-1)  # 类别
        # 准确率计算
        acc = sum(y_pre[i] == y_test[i] for i in range(y_test.shape[0])) / y_test.shape[0]
        print(f"Check predict {acc}={test_accuracy}")
        assert abs(acc - test_accuracy) < 1e-2
        return


class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None, last_iterations=None, last_rate=None):
        super().__init__()
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0

    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)

    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
        self.iteration += 1
        keras.backend.set_value(self.model.optimizer.learning_rate, rate)


class OneCycleSchedulerCallBack:
    def __init__(self):
        return

    @staticmethod
    def one_cycle_train():
        valid_sample = 5000
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        x_valid, x_train = x_train_full[:valid_sample], x_train_full[valid_sample:]
        y_valid, y_train = y_train_full[:valid_sample], y_train_full[valid_sample:]
        pixel_means = x_train.mean(axis=0, keepdims=True)
        pixel_stds = x_train.std(axis=0, keepdims=True)
        x_train_scaled = (x_train - pixel_means) / pixel_stds
        x_valid_scaled = (x_valid - pixel_means) / pixel_stds
        x_test_scaled = (x_test - pixel_means) / pixel_stds
        tf.random.set_seed(42)
        np.random.seed(42)

        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                      metrics=["accuracy"])
        n_epochs = 25
        batch_size = 128
        one_cycle = OneCycleScheduler(math.ceil(len(x_train) / batch_size) * n_epochs, max_rate=0.05)
        model.fit(x_train_scaled, y_train, epochs=n_epochs, batch_size=batch_size,
                            validation_data=(x_valid_scaled, y_valid),
                            callbacks=[one_cycle])

        test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)

        y_prob = model.predict(x_test_scaled)  # 概率
        y_pre = np.argmax(y_prob, axis=-1)  # 类别
        # 准确率计算
        acc = sum(y_pre[i] == y_test[i] for i in range(y_test.shape[0])) / y_test.shape[0]
        print(f"Check predict {acc}={test_accuracy}")
        assert abs(acc - test_accuracy) < 1e-2
        return


class AvoidOverFittingRegularization:
    def __init__(self):
        return

    @staticmethod
    def regularization():
        valid_sample = 5000
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        x_valid, x_train = x_train_full[:valid_sample], x_train_full[valid_sample:]
        y_valid, y_train = y_train_full[:valid_sample], y_train_full[valid_sample:]
        pixel_means = x_train.mean(axis=0, keepdims=True)
        pixel_stds = x_train.std(axis=0, keepdims=True)
        x_train_scaled = (x_train - pixel_means) / pixel_stds
        x_valid_scaled = (x_valid - pixel_means) / pixel_stds
        x_test_scaled = (x_test - pixel_means) / pixel_stds

        tf.random.set_seed(42)
        np.random.seed(42)
        # 在除了输入层的每个层引入正则化
        regularized_dense = partial(keras.layers.Dense,
                                    activation="elu",
                                    kernel_initializer="he_normal",
                                    kernel_regularizer=keras.regularizers.l2(0.01))

        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            regularized_dense(300),
            regularized_dense(100),
            regularized_dense(10, activation="softmax")
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
        n_epochs = 25
        model.fit(x_train_scaled, y_train, epochs=n_epochs,
                            validation_data=(x_valid_scaled, y_valid))

        test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)

        y_prob = model.predict(x_test_scaled)  # 概率
        y_pre = np.argmax(y_prob, axis=-1)  # 类别
        # 准确率计算
        acc = sum(y_pre[i] == y_test[i] for i in range(y_test.shape[0])) / y_test.shape[0]
        print(f"Check predict {acc}={test_accuracy}")
        assert abs(acc - test_accuracy) < 1e-2
        return


class DropOut:
    def __init__(self):
        return

    @staticmethod
    def drop_out_pipline():
        valid_sample = 5000
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train_full = x_train_full / 255.0
        x_test = x_test / 255.0
        x_valid, x_train = x_train_full[:valid_sample], x_train_full[valid_sample:]
        y_valid, y_train = y_train_full[:valid_sample], y_train_full[valid_sample:]
        pixel_means = x_train.mean(axis=0, keepdims=True)
        pixel_stds = x_train.std(axis=0, keepdims=True)
        x_train_scaled = (x_train - pixel_means) / pixel_stds
        x_valid_scaled = (x_valid - pixel_means) / pixel_stds
        x_test_scaled = (x_test - pixel_means) / pixel_stds

        tf.random.set_seed(42)
        np.random.seed(42)
        # 在除了输入层的每个层引入正则化
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
        n_epochs = 2
        model.fit(x_train_scaled, y_train, epochs=n_epochs,
                            validation_data=(x_valid_scaled, y_valid))
        test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)

        y_prob = model.predict(x_test_scaled)  # 概率
        y_pre = np.argmax(y_prob, axis=-1)  # 类别
        # 准确率计算
        acc = sum(y_pre[i] == y_test[i] for i in range(y_test.shape[0])) / y_test.shape[0]
        print(f"Check predict {acc}={test_accuracy}")
        assert abs(acc - test_accuracy) < 1e-2
        return


class TestGeneral(unittest.TestCase):

    @unittest.skip
    def test_activation_function_1(self):
        ActivationFunction().leaky_relu()
        return

    @unittest.skip
    def test_activation_function_2(self):
        ActivationFunction().leaky_prelu()
        return

    @unittest.skip
    def test_activation_function_3(self):
        ActivationFunction().selu()
        return

    @unittest.skip
    def test_batch_normalization(self):
        BatchNormalization().batch_normalization()
        return

    @unittest.skip
    def test_reusing_pretrained_layers(self):
        ReusingPretrainedLayers().reusing_pretrained_layers()
        return

    @unittest.skip
    def test_faster_optimization_1(self):
        FasterOptimizers().learning_rate_decay()
        return

    @unittest.skip
    def test_faster_optimization_2(self):
        FasterOptimizers().exponential_scheduling()
        return

    @unittest.skip
    def test_faster_optimization_3(self):
        FasterOptimizers().piecewise_constant_scheduling()
        return

    @unittest.skip
    def test_faster_optimization_4(self):
        FasterOptimizers().performance_scheduling()
        return

    @unittest.skip
    def test_faster_optimization_5(self):
        FasterOptimizers().exponential_scheduling_class()
        return

    @unittest.skip
    def test_one_cycle_scheduler_call_back(self):
        OneCycleSchedulerCallBack().one_cycle_train()
        return

    @unittest.skip
    def test_avoid_over_fitting_regularization(self):
        AvoidOverFittingRegularization.regularization()
        return

    def test_drop_out(self):
        DropOut.drop_out_pipline()
        return


if __name__ == '__main__':
    unittest.main()
