"""
算法：快速幂
功能：高效计算整数的幂次方取模
题目：xx（xx）
参考：OI WiKi（xx）
"""

import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
from functools import reduce
from operator import xor
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy


class FastPower:
    def __init__(self):
        return

    @staticmethod
    def fast_power_api(a, b, mod):
        return pow(a, b, mod)

    @staticmethod
    def fast_power(a, b, mod):
        a = a % mod
        res = 1
        while b > 0:
            if b & 1:
                res = res * a % mod
            a = a * a % mod
            b >>= 1
        return res

    @staticmethod
    def x_pow(x: float, n: int) -> float:
        def quick_mul(n):
            if n == 0:
                return 1.0
            y = quick_mul(n // 2)
            return y * y if n % 2 == 0 else y * y * x

        return quick_mul(n) if n >= 0 else 1.0 / quick_mul(-n)


class TestGeneral(unittest.TestCase):

    def test_fast_power(self):
        fp = FastPower()
        a, b, mod = random.randint(1, 123), random.randint(1, 1234), random.randint(1, 12345)
        assert fp.fast_power_api(a,b,mod) == fp.fast_power(a,b,mod)

        x, n = random.uniform(0, 1), random.randint(1, 1234)
        assert abs(fp.x_pow(x, n)-pow(x, n)) < 1e-5
        return


if __name__ == '__main__':
    unittest.main()
