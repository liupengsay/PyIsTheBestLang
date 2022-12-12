"""
算法：乘法逆元
功能：求逆元取模
题目：P3811 【模板】乘法逆元（https://www.luogu.com.cn/problem/P3811）
P5431 【模板】乘法逆元 2（https://www.luogu.com.cn/problem/P5431）
P2613 【模板】有理数取余（https://www.luogu.com.cn/problem/P2613）

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


class MultiplicativeInverse:
    def __init__(self):
        return

    @staticmethod
    def get_result(a, p):
        # 注意a和p都为正整数
        return pow(a, -1, p)


class TestGeneral(unittest.TestCase):

    def test_multiplicative_inverse(self):
        mt = MultiplicativeInverse()
        assert mt.get_result(10, 13) == 4
        assert mt.get_result(10, 1) == 0
        return


if __name__ == '__main__':
    unittest.main()
