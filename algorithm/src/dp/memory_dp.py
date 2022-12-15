"""

"""
"""
算法：记忆化搜索
功能：记忆化形式的DP，可以自顶向下也可以自底向上
题目：
L2328 网格图中递增路径的数目（https://leetcode.cn/problems/number-of-increasing-paths-in-a-grid/）计算严格递增的路径数量
L2312 卖木头块（https://leetcode.cn/problems/selling-pieces-of-wood/）自顶向下搜索最佳方案
L2267 检查是否有合法括号字符串路径（https://leetcode.cn/problems/check-if-there-is-a-valid-parentheses-string-path/）记忆化搜索合法路径

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


class ClassName:
    def __init__(self):
        return

    def gen_result(self):
        return


class TestGeneral(unittest.TestCase):

    def test_xxx(self):
        nt = ClassName()
        assert nt.gen_result(10 ** 11 + 131) == 66666666752
        return


if __name__ == '__main__':
    unittest.main()
