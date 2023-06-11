import random
import time
import unittest

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

random.seed(2023)
tf.random.set_seed(2023)
np.random.seed(2023)


"""
算法：使用tensorflow进行常规的模型训练
"""


class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pre):
        error = y_true - y_pre
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))

    def get_config(self):
        return {"factor": self.factor}


class HuberMetric(keras.metrics.Mean):
    def __init__(self, threshold=1.0, name='HuberMetric', dtype=None):
        self.threshold = threshold

        def create_huber(thr=1.0):
            def huber_fn(y_true, y_pre):
                error = y_true - y_pre
                is_small_error = tf.abs(error) < thr
                squared_loss = tf.square(error) / 2
                linear_loss = thr * tf.abs(error) - thr ** 2 / 2
                return tf.where(is_small_error, squared_loss, linear_loss)

            return huber_fn

        self.huber_fn = create_huber(threshold)
        super().__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pre, sample_weight=None):
        metric = self.huber_fn(y_true, y_pre)
        super(HuberMetric, self).update_state(metric, sample_weight)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


class CustomLossFunction:
    def __init__(self):
        # 自定义损失函数进行计算
        return

    @staticmethod
    def huber_pipline():
        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)
        x_test_scaled = scaler.transform(x_test)

        def huber_fn(y_true, y_pre):
            error = y_true - y_pre
            is_small_error = tf.abs(error) < 1
            squared_loss = tf.square(error) / 2
            linear_loss = tf.abs(error) - 0.5
            return tf.where(is_small_error, squared_loss, linear_loss)

        input_shape = x_train.shape[1:]

        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                               input_shape=input_shape),
            keras.layers.Dense(1),
        ])
        model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])
        model.fit(x_train_scaled, y_train, epochs=2,
                  validation_data=(x_valid_scaled, y_valid))

        # 模型评估
        test_loss, test_mae = model.evaluate(x_test_scaled, y_test)
        y_predict = model.predict(x_test_scaled)  # 回归预测值
        # 准确率计算
        compute_loss = sum(huber_fn(y_test, y_predict))/len(y_predict)
        print(f"Check predict {compute_loss[0]}={test_loss}")
        assert abs(compute_loss[0] - test_loss) < 1e-2
        return

    @staticmethod
    def save_and_load():
        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)
        x_test_scaled = scaler.transform(x_test)

        def create_huber(threshold=1.0):
            def huber_fn(y_true, y_pre):
                error = y_true - y_pre
                is_small_error = tf.abs(error) < threshold
                squared_loss = tf.square(error) / 2
                linear_loss = threshold * tf.abs(error) - threshold ** 2 / 2
                return tf.where(is_small_error, squared_loss, linear_loss)

            return huber_fn

        input_shape = x_train.shape[1:]
        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                               input_shape=input_shape),
            keras.layers.Dense(1),
        ])
        model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=["mae"])
        model.fit(x_train_scaled, y_train, epochs=2,
                  validation_data=(x_valid_scaled, y_valid))
        model.save("./data/my_model_with_a_custom_loss_threshold_2.h5")
        model = keras.models.load_model("./data/my_model_with_a_custom_loss_threshold_2.h5",
                                        custom_objects={"huber_fn": create_huber(2.0)})
        # 模型评估
        test_loss, test_mae = model.evaluate(x_test_scaled, y_test)
        y_predict = model.predict(x_test_scaled)  # 回归预测值
        # 准确率计算
        compute_loss = sum(create_huber(2.0)(y_test, y_predict))/len(y_predict)
        print(f"Check predict {compute_loss[0]}={test_loss}")
        assert abs(compute_loss[0] - test_loss) < 1e-2
        return

    @staticmethod
    def save_and_load_class():
        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)
        x_test_scaled = scaler.transform(x_test)

        input_shape = x_train.shape[1:]
        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                               input_shape=input_shape),
            keras.layers.Dense(1),
        ])
        model.compile(loss=HuberLoss(2.), optimizer="nadam", metrics=["mae"])
        model.fit(x_train_scaled, y_train, epochs=2,
                  validation_data=(x_valid_scaled, y_valid))
        model.save("./data/my_model_with_a_custom_loss_class.h5")
        model = keras.models.load_model("./data/my_model_with_a_custom_loss_class.h5",
                                        custom_objects={"HuberLoss": HuberLoss})
        # 模型评估
        test_loss, test_mae = model.evaluate(x_test_scaled, y_test)
        y_predict = model.predict(x_test_scaled)  # 回归预测值
        # 准确率计算
        compute_loss = sum(HuberLoss(2.0).call(y_test, y_predict))/len(y_predict)
        print(f"Check predict {compute_loss[0]}={test_loss}")
        assert abs(compute_loss[0] - test_loss) < 1e-2
        return

    @staticmethod
    def other_custom_functions():
        def my_soft_plus(z):  # return value is just tf.nn.soft_plus(z)
            return tf.math.log(tf.exp(z) + 1.0)

        def my_l1_regula_rizer(weights):
            return tf.reduce_sum(tf.abs(0.01 * weights))

        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)
        x_test_scaled = scaler.transform(x_test)
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        input_shape = x_train.shape[1:]
        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                               input_shape=input_shape),
            keras.layers.Dense(1, activation=my_soft_plus,
                               kernel_regularizer=my_l1_regula_rizer),
        ])

        model.compile(loss="mse", optimizer="nadam", metrics=["mae"])
        model.fit(x_train_scaled, y_train, epochs=2,
                  validation_data=(x_valid_scaled, y_valid))
        model.save("./data/my_model_with_many_custom_parts.h5")
        model = keras.models.load_model("./data/my_model_with_many_custom_parts.h5",
                                        custom_objects={
                                            "my_l1_regula_rizer": my_l1_regula_rizer,
                                            "my_soft_plus": my_soft_plus,
                                        })
        # 模型评估
        test_loss, test_mae = model.evaluate(x_test_scaled, y_test)
        y_predict = model.predict(x_test_scaled)  # 回归预测值
        # 准确率计算
        compute_loss = sum(abs(y_predict[i]-y_test[i]) for i in range(len(y_predict)))/len(y_predict)
        print(f"Check predict {compute_loss[0]}={test_mae}")
        assert abs(compute_loss[0] - test_mae) < 1e-2
        return

    @staticmethod
    def other_custom_functions_class():
        def my_soft_plus(z):  # return value is just tf.nn.soft_plus(z)
            return tf.math.log(tf.exp(z) + 1.0)

        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)
        x_test_scaled = scaler.transform(x_test)
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        input_shape = x_train.shape[1:]
        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                               input_shape=input_shape),
            keras.layers.Dense(1, activation=my_soft_plus,
                               kernel_regularizer=MyL1Regularizer(0.01)),
        ])

        model.compile(loss="mse", optimizer="nadam", metrics=["mae"])
        model.fit(x_train_scaled, y_train, epochs=2,
                  validation_data=(x_valid_scaled, y_valid))
        model.save("./data/my_model_with_many_custom_parts.h5")
        model = keras.models.load_model("./data/my_model_with_many_custom_parts.h5",
                                        custom_objects={
                                            "MyL1Regularizer": MyL1Regularizer,
                                            "my_soft_plus": my_soft_plus,
                                        })
        # 模型评估
        test_loss, test_mae = model.evaluate(x_test_scaled, y_test)
        y_predict = model.predict(x_test_scaled)  # 回归预测值
        # 准确率计算
        compute_loss = sum(abs(y_predict[i]-y_test[i]) for i in range(len(y_predict)))/len(y_predict)
        print(f"Check predict {compute_loss[0]}={test_mae}")
        assert abs(compute_loss[0] - test_mae) < 1e-2
        return

    @staticmethod
    def other_custom_functions_class_metric():

        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)
        x_test_scaled = scaler.transform(x_test)
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        input_shape = x_train.shape[1:]
        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                               input_shape=input_shape),
            keras.layers.Dense(1),
        ])

        model.compile(loss=keras.losses.Huber(2.0), optimizer="nadam", weighted_metrics=[HuberMetric(2.0)])
        sample_weight = np.random.rand(len(y_train))
        model.fit(x_train_scaled.astype(np.float32), y_train.astype(np.float32),
                            epochs=2, sample_weight=sample_weight,
                  validation_data=(x_valid_scaled.astype(np.float32), y_valid.astype(np.float32)))
        model.save("./data/my_model_with_a_custom_metric_v2.h5")
        model = keras.models.load_model("./data/my_model_with_a_custom_metric_v2.h5",
                                        custom_objects={"HuberMetric": HuberMetric})
        # 模型评估
        test_loss, test_mae = model.evaluate(x_test_scaled, y_test)
        y_predict = model.predict(x_test_scaled)  # 回归预测值
        # 准确率计算
        compute_loss = sum(HuberMetric(2.0).huber_fn(y_test, y_predict))/len(y_predict)
        print(f"Check predict {compute_loss[0]}={test_mae}")
        assert abs(compute_loss[0] - test_mae) < 1e-2
        return


class MyDense(keras.layers.Layer):
    # 自定义层
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.kernel = None
        self.units = units
        self.bias = None
        self.activation = keras.activations.get(activation)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[batch_input_shape[-1], self.units],
            initializer="glorot_normal")
        self.bias = self.add_weight(name="bias", shape=[self.units], initializer="zeros")
        super().build(batch_input_shape)  # must be at the end

    def call(self, x, *args, **kwargs):
        return self.activation(x @ self.kernel + self.bias)

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units, "activation": keras.activations.serialize(self.activation)}


class MyMultiLayer(keras.layers.Layer):
    def call(self, x, *args, **kwargs):
        x1, x2 = x
        print("x1.shape: ", x1.shape, " x2.shape: ", x2.shape)  # Debugging of custom layer
        return x1 + x2, x1 * x2

    def compute_output_shape(self, batch_input_shape):
        batch_input_shape1, batch_input_shape2 = batch_input_shape
        return [batch_input_shape1, batch_input_shape2]
    

class AddGaussianNoise(keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, x, training=None, *args, **kwargs):
        if training:
            noise = tf.random.normal(tf.shape(x), stddev=self.stddev)
            return x + noise
        else:
            return x

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape


class CustomLayers:
    def __init__(self):
        # 自定义层
        return

    @staticmethod
    def my_dense_custom_layers():
        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)
        x_test_scaled = scaler.transform(x_test)
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        input_shape = x_train.shape[1:]
        model = keras.models.Sequential([
            MyDense(30, activation="relu", input_shape=input_shape),
            MyDense(1)
        ])
        model.compile(loss="mse", optimizer="nadam")
        model.fit(x_train_scaled, y_train, epochs=2,
                  validation_data=(x_valid_scaled, y_valid))
        model.save("./data/my_model_with_a_custom_layer.h5")
        model = keras.models.load_model("./data/my_model_with_a_custom_layer.h5",
                                        custom_objects={"MyDense": MyDense})
        model.evaluate(x_test_scaled, y_test)
        # 模型评估
        test_loss = model.evaluate(x_test_scaled, y_test)
        y_predict = model.predict(x_test_scaled)  # 回归预测值
        # 准确率计算
        compute_loss = sum(abs(y_predict[i]-y_test[i])**2 for i in range(len(y_predict)))/len(y_predict)
        print(f"Check predict {compute_loss[0]}={test_loss}")
        assert abs(compute_loss[0] - test_loss) < 1e-2

    @staticmethod
    def my_multi_layer():

        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)
        x_test_scaled = scaler.transform(x_test)

        def split_data(data):
            columns_count = data.shape[-1]
            half = columns_count // 2
            return data[:, :half], data[:, half:]

        x_train_scaled_a, x_train_scaled_b = split_data(x_train_scaled)
        x_valid_scaled_a, x_valid_scaled_b = split_data(x_valid_scaled)
        x_test_scaled_a, x_test_scaled_b = split_data(x_test_scaled)
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        input_a = keras.layers.Input(shape=x_train_scaled_a.shape[-1])
        input_b = keras.layers.Input(shape=x_train_scaled_b.shape[-1])
        hidden_a, hidden_b = MyMultiLayer()((input_a, input_b))
        hidden_a = keras.layers.Dense(30, activation='selu')(hidden_a)
        hidden_b = keras.layers.Dense(30, activation='selu')(hidden_b)
        concat = keras.layers.Concatenate()((hidden_a, hidden_b))
        output = keras.layers.Dense(1)(concat)
        model = keras.models.Model(inputs=[input_a, input_b], outputs=[output])
        
        model.compile(loss='mse', optimizer='nadam')
        model.fit((x_train_scaled_a, x_train_scaled_b), y_train, epochs=2,
                  validation_data=((x_valid_scaled_a, x_valid_scaled_b), y_valid))
        # 模型评估
        test_loss = model.evaluate((x_test_scaled_a, x_test_scaled_b), y_test)
        y_predict = model.predict((x_test_scaled_a, x_test_scaled_b))  # 回归预测值
        # 准确率计算
        compute_loss = sum(abs(y_predict[i]-y_test[i])**2 for i in range(len(y_predict)))/len(y_predict)
        print(f"Check predict {compute_loss[0]}={test_loss}")
        assert abs(compute_loss[0] - test_loss) < 1e-2

    @staticmethod
    def add_gaussian_noise():
        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)
        x_test_scaled = scaler.transform(x_test)

        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        model = keras.models.Sequential([
            AddGaussianNoise(stddev=1.0),  # 增加高斯噪声
            keras.layers.Dense(30, activation="selu"),
            keras.layers.Dense(1)
        ])

        model.compile(loss="mse", optimizer="nadam")
        model.fit(x_train_scaled, y_train, epochs=2,
                  validation_data=(x_valid_scaled, y_valid))
        # 模型评估
        test_loss = model.evaluate(x_test_scaled, y_test)
        y_predict = model.predict(x_test_scaled)  # 回归预测值
        # 准确率计算
        compute_loss = sum(abs(y_predict[i] - y_test[i]) ** 2 for i in range(len(y_predict))) / len(y_predict)
        print(f"Check predict {compute_loss[0]}={test_loss}")
        assert abs(compute_loss[0] - test_loss) < 1e-2


class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(n_neurons, activation="elu",
                                          kernel_initializer="he_normal")
                       for _ in range(n_layers)]

    def call(self, inputs, *args, **kwargs):
        z = inputs
        for layer in self.hidden:
            z = layer(z)
        return inputs + z

    
class ResidualRegressor(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(30, activation="elu",
                                          kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = keras.layers.Dense(output_dim)

    def call(self, inputs, *args, **kwargs):
        z = self.hidden1(inputs)
        for _ in range(1 + 3):
            z = self.block1(z)
        z = self.block2(z)
        return self.out(z)


class ResidualCustomModel:
    def __init__(self):
        # 自定义模型
        return

    @staticmethod
    def residual_regressor():
        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)
        x_test_scaled = scaler.transform(x_test)

        keras.backend.clear_session()

        model = ResidualRegressor(1)
        model.compile(loss="mse", optimizer="nadam")
        model.fit(x_train_scaled, y_train, epochs=5,
                  validation_data=(x_valid_scaled, y_valid))
        # 模型评估
        test_loss = model.evaluate(x_test_scaled, y_test)
        y_predict = model.predict(x_test_scaled)  # 回归预测值
        # 准确率计算
        compute_loss = sum(abs(y_predict[i] - y_test[i]) ** 2 for i in range(len(y_predict))) / len(y_predict)
        print(f"Check predict {compute_loss[0]}={test_loss}")
        assert abs(compute_loss[0] - test_loss) < 1e-2
        return

    @staticmethod
    def residual_block():
        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)
        x_test_scaled = scaler.transform(x_test)

        keras.backend.clear_session()
        block1 = ResidualBlock(2, 30)
        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal"),
            block1, block1, block1, block1,
            ResidualBlock(2, 30),
            keras.layers.Dense(1)
        ])

        model.compile(loss="mse", optimizer="nadam")
        model.fit(x_train_scaled, y_train, epochs=5,
                  validation_data=(x_valid_scaled, y_valid))
        # 模型评估
        test_loss = model.evaluate(x_test_scaled, y_test)
        y_predict = model.predict(x_test_scaled)  # 回归预测值
        # 准确率计算
        compute_loss = sum(abs(y_predict[i] - y_test[i]) ** 2 for i in range(len(y_predict))) / len(y_predict)
        print(f"Check predict {compute_loss[0]}={test_loss}")
        assert abs(compute_loss[0] - test_loss) < 1e-2
        return


class ReconstructingRegressor(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(30, activation="selu",
                                          kernel_initializer="lecun_normal")
                       for _ in range(5)]
        self.out = keras.layers.Dense(output_dim)
        self.reconstruct = None
        self.reconstruction_mean = keras.metrics.Mean(name="reconstruction_error")

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = keras.layers.Dense(n_inputs)

    def call(self, inputs, training=None, *args, **kwargs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)
        if training:
            result = self.reconstruction_mean(recon_loss)
            self.add_metric(result)
        return self.out(Z)


class ReconstructingRegressorLossAndMetrics:
    def __init__(self):
        return

    @staticmethod
    def reconstruct():
        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)
        x_test_scaled = scaler.transform(x_test)

        keras.backend.clear_session()
        model = ReconstructingRegressor(1)

        model.compile(loss="mse", optimizer="nadam")
        model.fit(x_train_scaled, y_train, epochs=15,
                  validation_data=(x_valid_scaled, y_valid))
        # 模型评估
        test_loss, error = model.evaluate(x_test_scaled, y_test)
        y_predict = model.predict(x_test_scaled)  # 回归预测值
        # 准确率计算
        compute_loss = sum(abs(y_predict[i] - y_test[i]) ** 2 for i in range(len(y_predict))) / len(y_predict)
        print(f"Check predict {compute_loss[0]}={test_loss}")
        assert abs(compute_loss[0] - test_loss) < 1e-2
        return


class CustomTrainingCircle:
    def __init__(self):
        return

    @staticmethod
    def custom_training_circle():
        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)
        x_test_scaled = scaler.transform(x_test)

        def progress_bar(iteration, total, size=30):
            running = iteration < total
            c = ">" if running else "="
            p = (size - 1) * iteration // total
            fmt = "{{:-{}d}}/{{}} [{{}}]".format(len(str(total)))
            params = [iteration, total, "=" * p + c + "." * (size - p - 1)]
            return fmt.format(*params)

        def print_status_bar(iteration, total, loss, metrics=None, size=30):
            metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                                  for m in [loss] + (metrics or [])])
            end = "" if iteration < total else "\n"
            print("\r{} - {}".format(progress_bar(iteration, total), metrics), end=end)

        keras.backend.clear_session()
        l2_reg = keras.regularizers.l2(0.05)
        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal",
                               kernel_regularizer=l2_reg),
            keras.layers.Dense(1, kernel_regularizer=l2_reg)
        ])
        mean_loss = keras.metrics.Mean(name="loss")
        mean_square = keras.metrics.Mean(name="mean_square")
        for i in range(1, 50 + 1):
            loss = 1 / i
            mean_loss(loss)
            mean_square(i ** 2)
            print_status_bar(i, 50, mean_loss, [mean_square])
            time.sleep(0.05)

        def random_batch(X, y, batch_size=32):
            idx = np.random.randint(len(X), size=batch_size)
            return X[idx], y[idx]

        n_epochs = 5
        batch_size = 32
        n_steps = len(x_train) // batch_size
        optimizer = keras.optimizers.Nadam(learning_rate=0.01)
        loss_fn = keras.losses.mean_squared_error
        mean_loss = keras.metrics.Mean()
        metrics = [keras.metrics.MeanAbsoluteError()]
        for epoch in range(1, n_epochs + 1):
            print("Epoch {}/{}".format(epoch, n_epochs))
            for step in range(1, n_steps + 1):
                x_batch, y_batch = random_batch(x_train_scaled, y_train)
                with tf.GradientTape() as tape:
                    y_pred = model(x_batch)
                    main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                    loss = tf.add_n([main_loss] + model.losses)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                for variable in model.variables:
                    if variable.constraint is not None:
                        variable.assign(variable.constraint(variable))
                mean_loss(loss)
                for metric in metrics:
                    metric(y_batch, y_pred)
                print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)
            print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
            for metric in [mean_loss] + metrics:
                metric.reset_states()

        model.compile(loss="mse", optimizer="nadam")
        # 模型评估
        test_loss = model.evaluate(x_test_scaled, y_test)
        y_predict = model.predict(x_test_scaled)  # 回归预测值
        # 准确率计算
        compute_loss = sum(abs(y_predict[i] - y_test[i]) for i in range(len(y_predict))) / len(y_predict)
        print(f"Check predict {compute_loss[0]}={test_loss}")
        return



class TestGeneral(unittest.TestCase):

    @unittest.skip
    def test_custom_loss_function_1(self):
        CustomLossFunction().huber_pipline()
        return

    @unittest.skip
    def test_custom_loss_function_2(self):
        CustomLossFunction().save_and_load()
        return

    @unittest.skip
    def test_custom_loss_function_3(self):
        CustomLossFunction().save_and_load_class()
        return

    @unittest.skip
    def test_custom_loss_function_4(self):
        CustomLossFunction().other_custom_functions()
        return

    @unittest.skip
    def test_custom_loss_function_5(self):
        CustomLossFunction().other_custom_functions_class_metric()
        return

    @unittest.skip
    def test_custom_layers_1(self):
        CustomLayers.my_dense_custom_layers()
        return

    @unittest.skip
    def test_custom_layers_2(self):
        CustomLayers.my_multi_layer()
        return

    @unittest.skip
    def test_custom_layers_3(self):
        CustomLayers.add_gaussian_noise()
        return

    @unittest.skip
    def test_residual_custom_model_1(self):
        ResidualCustomModel.residual_regressor()
        return

    @unittest.skip
    def test_residual_custom_model_2(self):
        ResidualCustomModel.residual_block()
        return

    @unittest.skip
    def test_reconstruct_regressor(self):
        ReconstructingRegressorLossAndMetrics.reconstruct()
        return

    def test_custom_training_circle(self):
        CustomTrainingCircle.custom_training_circle()
        return


if __name__ == '__main__':
    unittest.main()
