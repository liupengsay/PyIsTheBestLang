"""

"""
"""
算法：xxx
功能：xxx
题目：
L0214 最短回文串（https://leetcode.cn/problems/shortest-palindrome/）计算字符串前缀最长回文子串

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


from itertools import combinations


def combination(n, k):
    return combinations(list(range(n)), k)


def test_combination():
    print([item for item in combination(4, 2)])
    return


if __name__ == '__main__':
    test_combination()


class TestGeneral(unittest.TestCase):

    def test_xxx(self):
        nt = ClassName()
        assert nt.gen_result(10 ** 11 + 131) == 66666666752
        return


if __name__ == '__main__':
    unittest.main()
