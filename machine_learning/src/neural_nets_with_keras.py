import unittest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

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
        acc = sum(y_pre[i] == y_test[i] for i in range(y_test.shape[0])) / y_test.shape[0]
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
        mse = sum((y_pre[i] - y_test[i]) ** 2 for i in range(len(y_pre))) / len(y_pre)
        print(mse[0], mse_test)
        assert abs(mse[0] - mse_test) < 1e-2
        return


class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))


class WideAndDeep:
    def __init__(self):
        return

    @staticmethod
    def pipline():  # 房价回归预测使用函数式 API 与输入层拼接隐藏层

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
        input_ = keras.layers.Input(shape=x_train.shape[1:])
        hidden1 = keras.layers.Dense(30, activation="relu")(input_)  # 函数式 API
        hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
        concat = keras.layers.concatenate([input_, hidden2])  # 直接增加输入层到最后一步
        output = keras.layers.Dense(1)(concat)
        model = keras.models.Model(inputs=[input_], outputs=[output])

        model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
        model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid))
        mse_test = model.evaluate(x_test, y_test)
        y_pre = model.predict(x_test)
        mse = sum((y_pre[i] - y_test[i]) ** 2 for i in range(len(y_pre))) / len(y_pre)
        print(mse[0], mse_test)
        assert abs(mse[0] - mse_test) < 1e-2
        return

    @staticmethod
    def pipline_functional_api():  # 房价回归预测使用函数式 API 与输入层拼接隐藏层

        # 读取数据并进行训练验证划分
        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full, random_state=42)

        # 标准归一化数据
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)

        # 选取不同特征子集作为输入
        x_train_a, x_train_b = x_train[:, :5], x_train[:, 2:]
        x_valid_a, x_valid_b = x_valid[:, :5], x_valid[:, 2:]
        x_test_a, x_test_b = x_test[:, :5], x_test[:, 2:]
        np.random.seed(42)
        tf.random.set_seed(42)
        assert x_train.shape == (11610, 8)

        # 建立网络层模型与损失函数优化器
        input_a = keras.layers.Input(shape=[5], name="wide_input")
        input_b = keras.layers.Input(shape=[6], name="deep_input")
        hidden1 = keras.layers.Dense(30, activation="relu")(input_b)
        hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
        concat = keras.layers.concatenate([input_a, hidden2])
        output = keras.layers.Dense(1, name="output")(concat)
        model = keras.models.Model(inputs=[input_a, input_b], outputs=[output])

        model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
        model.fit((x_train_a, x_train_b), y_train, epochs=20,
                  validation_data=((x_valid_a, x_valid_b), y_valid))
        mse_test = model.evaluate((x_test_a, x_test_b), y_test)
        y_pre = model.predict((x_test_a, x_test_b))
        mse = sum((y_pre[i] - y_test[i]) ** 2 for i in range(len(y_pre))) / len(y_pre)
        print(mse[0], mse_test)
        assert abs(mse[0] - mse_test) < 1e-2

        # 添加使用多个用于正则化的输出
        input_a = keras.layers.Input(shape=[5], name="wide_input")
        input_b = keras.layers.Input(shape=[6], name="deep_input")
        hidden1 = keras.layers.Dense(30, activation="relu")(input_b)
        hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
        concat = keras.layers.concatenate([input_a, hidden2])
        output = keras.layers.Dense(1, name="main_output")(concat)
        aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
        model = keras.models.Model(inputs=[input_a, input_b],
                                   outputs=[output, aux_output])
        model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(learning_rate=1e-3))
        model.fit([x_train_a, x_train_b], [y_train, y_train], epochs=20,
                  validation_data=([x_valid_a, x_valid_b], [y_valid, y_valid]))
        total_loss, main_loss, aux_loss = model.evaluate(
            [x_test_a, x_test_b], [y_test, y_test])
        y_pre_main, y_pre_aux = model.predict([x_test_a, x_test_b])

        mse = sum((y_pre_main[i] - y_test[i]) ** 2 for i in range(len(y_pre_main))) / len(y_pre_main)
        print(mse[0], main_loss)
        assert abs(mse[0] - main_loss) < 1e-2

        return

    @staticmethod
    def pipline_save_and_load():  # 房价回归预测模型权重保存与加载

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
            keras.layers.Dense(30, activation="relu", input_shape=[8]),
            keras.layers.Dense(30, activation="relu"),
            keras.layers.Dense(1)
        ])
        model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
        model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))
        mse_test = model.evaluate(x_test, y_test)
        y_pre = model.predict(x_test)
        mse = sum((y_pre[i] - y_test[i]) ** 2 for i in range(len(y_pre))) / len(y_pre)
        print(mse[0], mse_test)
        assert abs(mse[0] - mse_test) < 1e-2

        model.save("./data/my_keras_model.h5")
        model = keras.models.load_model("./data/my_keras_model.h5")
        model.save_weights("./data/my_keras_weights.ckpt")
        model.load_weights("./data/my_keras_weights.ckpt")
        y_pre = model.predict(x_test)
        mse = sum((y_pre[i] - y_test[i]) ** 2 for i in range(len(y_pre))) / len(y_pre)
        print(mse[0], mse_test)
        assert abs(mse[0] - mse_test) < 1e-2
        return

    @staticmethod
    def pipline_callback():  # 房价回归预测在训练过程保存最好的结果进行回滚

        # 读取数据并进行训练验证划分
        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full, random_state=42)

        # 标准归一化数据
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)

        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)
        assert x_train.shape == (11610, 8)
        # 建立网络层模型与损失函数优化器
        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="relu", input_shape=[8]),
            keras.layers.Dense(30, activation="relu"),
            keras.layers.Dense(1)
        ])
        model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
        checkpoint_cb = keras.callbacks.ModelCheckpoint("./data/my_keras_model.h5", save_best_only=True)
        model.fit(x_train, y_train, epochs=10,
                  validation_data=(x_valid, y_valid),
                  callbacks=[checkpoint_cb])
        # 使用callback保存与读取最好的模型
        model = keras.models.load_model("./data/my_keras_model.h5")  # rollback to best model
        mse_test = model.evaluate(x_test, y_test)
        y_pre = model.predict(x_test)
        mse = sum((y_pre[i] - y_test[i]) ** 2 for i in range(len(y_pre))) / len(y_pre)
        print(mse[0], mse_test)
        assert abs(mse[0] - mse_test) < 1e-2

        # 还可以增加提前停止的callback减少无效训练
        model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                          restore_best_weights=True)
        model.fit(x_train, y_train, epochs=100,
                  validation_data=(x_valid, y_valid),
                  callbacks=[checkpoint_cb, early_stopping_cb])
        mse_test = model.evaluate(x_test, y_test)
        y_pre = model.predict(x_test)
        mse = sum((y_pre[i] - y_test[i]) ** 2 for i in range(len(y_pre))) / len(y_pre)
        print(mse[0], mse_test)
        assert abs(mse[0] - mse_test) < 1e-2

        # 自定义call_back函数进行信息获取打印
        val_train_ratio_cb = PrintValTrainRatioCallback()
        model.fit(x_train, y_train, epochs=10,
                  validation_data=(x_valid, y_valid),
                  callbacks=[val_train_ratio_cb])
        y_pre = model.predict(x_test)
        mse = sum((y_pre[i] - y_test[i]) ** 2 for i in range(len(y_pre))) / len(y_pre)
        print(mse[0], mse_test)
        assert abs(mse[0] - mse_test) < 1e-2
        return


class HyperParameterTuning:
    def __init__(self):
        return

    @staticmethod
    def pipline():
        # 超参数调优
        def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
            model = keras.models.Sequential()
            model.add(keras.layers.InputLayer(input_shape=input_shape))
            for layer in range(n_hidden):
                model.add(keras.layers.Dense(n_neurons, activation="relu"))
            model.add(keras.layers.Dense(1))
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
            model.compile(loss="mse", optimizer=optimizer)
            return model

        # 读取数据并进行训练验证划分
        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full, random_state=42)

        # 标准归一化数据
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        # 使用scikit_learn进行超参数调优
        keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
        keras_reg.fit(x_train, y_train, epochs=100,
                      validation_data=(x_valid, y_valid),
                      callbacks=[keras.callbacks.EarlyStopping(patience=10)])
        mse_test = -keras_reg.score(x_test, y_test)
        y_pre = keras_reg.predict(x_test)

        mse = sum((y_pre[i] - y_test[i]) ** 2 for i in range(len(y_pre))) / len(y_pre)
        print(mse, mse_test)
        assert abs(mse - mse_test) < 1e-2

        # 使用网格搜索进行计算构建
        param_distribs = {
            "n_hidden": [0, 1, 2, 3],
            "n_neurons": np.arange(1, 100).tolist(),
            "learning_rate": reciprocal(3e-4, 3e-2).rvs(1000).tolist(),
        }
        rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
        rnd_search_cv.fit(x_train, y_train, epochs=100,
                          validation_data=(x_valid, y_valid),
                          callbacks=[keras.callbacks.EarlyStopping(patience=10)])
        rnd_search_cv.score(x_test, y_test)
        # 选择表现最好的模型
        model = rnd_search_cv.best_estimator_.model
        mse_test = model.evaluate(x_test, y_test)
        y_pre = model.predict(x_test)
        mse = sum((y_pre[i] - y_test[i]) ** 2 for i in range(len(y_pre))) / len(y_pre)
        print(mse[0], mse_test)
        assert abs(mse[0] - mse_test) < 1e-2
        return


class TestGeneral(unittest.TestCase):

    def test_fashion_mnist(self):
        FashionMnist().pipline()
        return

    def test_california_housing(self):
        CaliforniaHousing().pipline()
        return

    def test_california_housing_wide_and_deep(self):
        WideAndDeep().pipline()
        WideAndDeep().pipline_functional_api()
        WideAndDeep().pipline_save_and_load()
        WideAndDeep().pipline_callback()
        return

    def test_hyper_parameter_tuning(self):
        HyperParameterTuning().pipline()
        return


if __name__ == '__main__':
    unittest.main()
