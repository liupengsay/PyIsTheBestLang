
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

"""
算法：区间合并、区间覆盖、区间计数
功能：xxx
题目：
P2082 区间覆盖（加强版）（https://www.luogu.com.cn/problem/P2082）经典区间合并确定覆盖范围

参考：OI WiKi（xx）
"""


class RangeCoverCount:
    def __init__(self):
        return

    @staticmethod
    def range_merge(lst):

        # 合并线性区间未不相交的区间
        lst.sort(key=lambda it: it[0])
        ans = []
        x, y = lst[0]
        for a, b in lst[1:]:
            if a <= y:
                y = y if y > b else b
            else:
                ans.append([x, y])
                x, y = a, b
        ans.append([x, y])
        return ans


class TestGeneral(unittest.TestCase):

    def test_range_cover_count(self):
        rcc = RangeCoverCount()
        lst = [[1, 4], [2, 5], [3, 6], [8, 9]]
        assert rcc.range_merge(lst) == [[1, 6], [8, 9]]
        return


if __name__ == '__main__':
    unittest.main()
