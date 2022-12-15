"""

"""
"""
算法：状态压缩DP
功能：使用二进制数字表示转移状态，计算相应的转移方程，通常可以先计算满足条件的子集，有时通过深搜回溯枚举全部子集的办法比位运算枚举效率更高
题目：
L1723 完成所有工作的最短时间（https://leetcode.cn/problems/find-minimum-time-to-finish-all-jobs/）通过位运算枚举分配工作DP最小化的最大值
L1986 完成任务的最少工作时间段（https://leetcode.cn/problems/minimum-number-of-work-sessions-to-finish-the-tasks/）预处理计算子集后进行记忆化状态转移
L0698 划分为k个相等的子集（https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/）预处理计算子集后进行记忆化状态转移

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
