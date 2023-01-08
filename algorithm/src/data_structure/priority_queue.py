"""
算法：单调队列、双端队列
功能：维护单调性，计算滑动窗口最大值最小值
题目：
P2251 质量检测（https://www.luogu.com.cn/problem/P2251）滑动区间最小值
L0239 滑动窗口最大值（https://leetcode.cn/problems/sliding-window-maximum/）滑动区间最大值
参考：OI WiKi（xx）
P1750 出栈序列（https://www.luogu.com.cn/problem/P1750）经典题目，滑动指针窗口栈加队列
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
