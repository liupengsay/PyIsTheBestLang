
import os
import random
import unittest

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


random.seed(2023)
tf.random.set_seed(2023)
np.random.seed(2023)


"""
算法：使用tf进行数据加载与处理

"""


class SplitDataToMultipleCSV:
    def __init__(self):
        return

    @staticmethod
    def save_and_load():
        housing = fetch_california_housing()
        x_train_full, x_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_mean = scaler.mean_
        x_std = scaler.scale_

        def save_to_multiple_csv_files(data, name_prefix, header=None, n_parts=10):
            housing_dir = os.path.join("datasets", "housing")
            os.makedirs(housing_dir, exist_ok=True)
            path_format = os.path.join(housing_dir, "my_{}_{:02d}.csv")

            filepaths = []
            m = len(data)
            for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
                part_csv = path_format.format(name_prefix, file_idx)
                filepaths.append(part_csv)
                with open(part_csv, "wt", encoding="utf-8") as f:
                    if header is not None:
                        f.write(header)
                        f.write("\n")
                    for row_idx in row_indices:
                        f.write(",".join([repr(col) for col in data[row_idx]]))
                        f.write("\n")
            return filepaths

        train_data = np.c_[x_train, y_train]
        valid_data = np.c_[x_valid, y_valid]
        test_data = np.c_[x_test, y_test]
        header_cols = housing.feature_names + ["MedianHouseValue"]
        header = ",".join(header_cols)
        train_filepaths = save_to_multiple_csv_files(train_data, "train", header, n_parts=20)
        valid_filepaths = save_to_multiple_csv_files(valid_data, "valid", header, n_parts=10)
        test_filepaths = save_to_multiple_csv_files(test_data, "test", header, n_parts=10)
        with open(train_filepaths[0]) as f:
            for i in range(5):
                print(f.readline(), end="")
        return


class TestGeneral(unittest.TestCase):

    def test_split_data(self):
        SplitDataToMultipleCSV().save_and_load()
        return


if __name__ == '__main__':
    unittest.main()
