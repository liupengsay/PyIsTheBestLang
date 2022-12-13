"""
算法：裴蜀定理
功能：是一个关于最大公约数的定理，设 a、b是不全为零的整数，则存在整数x、y, 使得ax+by=gcd(a,b)
题目：
P4549 裴蜀定理（https://www.luogu.com.cn/problem/P4549）计算所有元素能加权生成的最小正数和即所有整数的最大公约数
参考：OI WiKi（https://oi-wiki.org/math/number-theory/bezouts/）
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


class PeiShuTheorem:
    def __init__(self, lst):
        self.lst = lst
        self.get_lst_gcd()
        return

    def get_lst_gcd(self):
        self.ans = self.lst[0]
        for g in self.lst[1:]:
            self.ans = math.gcd(self.ans, g)
        return


class TestGeneral(unittest.TestCase):

    def test_peishu_theorem(self):
        lst = [4059, -1782]
        pst = PeiShuTheorem(lst)
        assert pst.ans == 99
        return


if __name__ == '__main__':
    unittest.main()
