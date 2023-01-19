
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
算法：博弈类DP、玩游戏、必胜态、必输态
功能：通常使用枚举、区间DP加模拟贪心的方式，和记忆化搜索进行状态转移
题目：
P1290 欧几里德的游戏（https://www.luogu.com.cn/problem/P1290）典型的博弈DP题
375. 猜数字大小 II（https://leetcode.cn/problems/guess-number-higher-or-lower-ii/）使用区间DP求解的典型博弈DP
参考：OI WiKi（xx）
"""



class GameDP:
    def __init__(self):
        return

    @staticmethod
    def main_p1280(x, y):

        @lru_cache(None)
        def dfs(a, b):
            if a < b:
                a, b = b, a
            if a % b == 0:
                return True
            for i in range(1, a // b + 1):
                if not dfs(a - i * b, b):
                    return True
            return False

        ans = dfs(x, y)
        return ans


class TestGeneral(unittest.TestCase):

    def test_xxx(self):

        return


if __name__ == '__main__':
    unittest.main()
