"""
算法：nim游戏也叫公平组合游戏，属于博弈论范畴
功能：用来判断游戏是否存在必胜态与必输态
题目：P2197 【模板】nim 游戏（https://www.luogu.com.cn/problem/P2197）
参考：OI WiKi（https://oi-wiki.org/graph/lgv/）

有一个神奇的结论：当n堆石子的数量异或和等于0时，先手必胜，否则先手必败
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


class Nim:
    def __init__(self, lst):
        self.lst = lst
        return

    def gen_result(self):
        return reduce(xor, self.lst) != 0


class TestGeneral(unittest.TestCase):
    def test_euler_phi(self):
        nim = Nim([0, 2, 3])
        assert nim.gen_result() == True
        return


if __name__ == '__main__':
    unittest.main()
