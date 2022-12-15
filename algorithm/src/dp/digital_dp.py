"""

"""
"""
算法：数位DP
功能：统计满足一定条件的自然数个数，也可以根据字典序大小特点统计一些特定字符串的个数
题目：
L2376 统计特殊整数（https://leetcode.cn/problems/count-special-integers/）计算小于 n 的特殊正整数个数
L0233 数字 1 的个数
L1706 出现 2 的次数
L0600 不含连续 1 的非负整数
L0902 最大为 N 的数字组合
L1012 至少有 1 位重复的数字
L1067 范围内的数字计数
L1397 找到所有好字符串

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
    def countSpecialNumbers(self, n: int) -> int:
        s = str(n)
        @lru_cache(None)
        def f(i: int, mask: int, is_limit: bool, is_num: bool) -> int:
            if i == len(s):
                return int(is_num)
            res = 0
            if not is_num:  # 可以跳过当前数位
                res = f(i + 1, mask, False, False)
            up = int(s[i]) if is_limit else 9
            for d in range(0 if is_num else 1, up + 1):  # 枚举要填入的数字 d
                if mask >> d & 1 == 0:  # d 不在 mask 中
                    res += f(i + 1, mask | (1 << d), is_limit and d == up, True)
            return res
        return f(0, 0, True, False)


class Solution:
    def digitsCount(self, d: int, low: int, high: int) -> int:
        # L1067 范围内的数字计数
        def check(num):
            @lru_cache(None)
            def dfs(i, cnt, is_limit, is_num):
                if i == n:
                    if is_num:
                        return cnt
                    return 0
                res = 0
                if not is_num:
                    res += dfs(i + 1, 0, False, False)

                floor = 0 if is_num else 1
                ceil = int(s[i]) if is_limit else 9
                for x in range(floor, ceil + 1):
                    res += dfs(i + 1, cnt + int(x == d), is_limit and ceil == x, True)
                return res
            s = str(num)
            n = len(s)
            return dfs(0, 0, True, False)
        return check(high) - check(low - 1)


class TestGeneral(unittest.TestCase):

    def test_xxx(self):
        nt = ClassName()
        assert nt.gen_result(10 ** 11 + 131) == 66666666752
        return


if __name__ == '__main__':
    unittest.main()
