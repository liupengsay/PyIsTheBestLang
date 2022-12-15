"""

"""
"""
算法：线性DP
功能：xxx
题目：
L2361 乘坐火车路线的最少费用（https://leetcode.cn/problems/minimum-costs-using-the-train-line/）当前状态只跟前一个状态有关
L2318 不同骰子序列的数目（https://leetcode.cn/problems/number-of-distinct-roll-sequences/）当前状态只跟前一个状态有关使用枚举计数
L2263 数组变为有序的最小操作次数（https://leetcode.cn/problems/make-array-non-decreasing-or-non-increasing/）当前状态只跟前一个状态有关
L2209 用地毯覆盖后的最少白色砖块（https://leetcode.cn/problems/minimum-white-tiles-after-covering-with-carpets/）前缀优化与处理进行转移
L2188 完成比赛的最少时间（https://leetcode.cn/problems/minimum-time-to-finish-the-race/）预处理DP

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


class Solution:
    def minimumCosts(self, regular: List[int], express: List[int], expressCost: int) -> List[int]:
        # L2361
        n = len(regular)
        cost = [[0, 0] for _ in range(n + 1)]
        cost[0][1] = expressCost
        for i in range(1, n + 1):
            cost[i][0] = min(cost[i - 1][0] + regular[i - 1],
                             cost[i - 1][1] + express[i - 1])
            cost[i][1] = min(cost[i][0] + expressCost,
                             cost[i - 1][1] + express[i - 1])
        return [min(c) for c in cost[1:]]


class TestGeneral(unittest.TestCase):

    def test_xxx(self):
        nt = ClassName()
        assert nt.gen_result(10 ** 11 + 131) == 66666666752
        return


if __name__ == '__main__':
    unittest.main()
