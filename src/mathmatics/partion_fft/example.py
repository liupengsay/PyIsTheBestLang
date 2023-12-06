import unittest

import numpy as np

from src.mathmatics.partion_fft.problem import fft_v


class TestGeneral(unittest.TestCase):
    def test_euler_phi(self):
        x = np.array([[3, 1, 2, 4]])
        # 调用fft函数x的FFT
        x_fft = fft_v(x)

        # 输出x_fft的值
        print(list(x_fft))
        return


if __name__ == '__main__':
    unittest.main()
