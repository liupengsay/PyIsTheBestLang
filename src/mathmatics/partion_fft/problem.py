"""
Algorithm：divide_and_conquer|fft
Description：

=====================================LuoGu======================================
4721（https://www.luogu.com.cn/problem/P4721）divide_and_conquer|fft
"""

import numpy as np


def fft_v(x):
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if np.log2(n) % 1 > 0:
        raise ValueError("must be a power of 2")

    n_min = min(n, 2)

    n = np.arange(n_min)
    k = n[:, None]
    m = np.exp(-2j * np.pi * n * k / n_min)
    xx = np.dot(m, x.reshape((n_min, -1)))
    while xx.shape[0] < n:
        x_even = xx[:, :int(xx.shape[1] / 2)]
        x_odd = xx[:, int(xx.shape[1] / 2):]
        terms = np.exp(-1j * np.pi * np.arange(xx.shape[0])
                       / xx.shape[0])[:, None]
        xx = np.vstack([x_even + terms * x_odd,
                        x_even - terms * x_odd])
    return xx.ravel()