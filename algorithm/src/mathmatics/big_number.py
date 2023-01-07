"""

"""
"""
算法：大数分解、素数判断、高精度计算
功能：xxx
题目：
Lxxxx xxxx（https://leetcode.cn/problems/shortest-palindrome/）xxxx

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
from decimal import Decimal, getcontext, MAX_PREC
getcontext().prec = MAX_PREC


class HighPrecision:
    def __init__(self):
        return

    @staticmethod
    def float_pow(r, n):
        ans = (Decimal(r) ** int(n)).normalize()
        ans = "{:f}".format(ans)
        return ans


class TestGeneral(unittest.TestCase):

    def test_high_percision(self):
        hp = HighPrecision()
        assert hp.float_pow("98.999", "5") == "9509420210.697891990494999"
        return


if __name__ == '__main__':
    unittest.main()
