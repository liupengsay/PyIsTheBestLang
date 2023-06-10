import unittest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

    def test_custom_loss_function_4(self):
        CustomLossFunction().other_custom_functions()
        return


if __name__ == '__main__':
    unittest.main()
