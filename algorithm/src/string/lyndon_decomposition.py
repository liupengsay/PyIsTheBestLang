"""
算法：Lyndon 分解（使用Duval 算法求解）
功能：用来将字符串s分解成Lyndon串s1s2s3...
Lyndon子串定义为：当且仅当s的字典序严格小于它的所有非平凡的（非平凡：非空且不同于自身）循环同构串时， s才是 Lyndon 串。
题目：P6657 【模板】LGV 引理（https://www.luogu.com.cn/problem/P6657）
参考：OI WiKi（https://oi-wiki.org/string/lyndon/）Duval 可以在 O(n)的时间内求出一个串的 Lyndon 分解
拓展：可用于求字符串s的最小表示法
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


class LyndonDecomposition:
    def __init__(self):
        return

    # duval_algorithm
    @staticmethod
    def solve_by_duval(s):
        n, i = len(s), 0
        factorization = []
        while i < n:
            j, k = i + 1, i
            while j < n and s[k] <= s[j]:
                if s[k] < s[j]:
                    k = i
                else:
                    k += 1
                j += 1
            while i <= k:
                factorization.append(s[i: i + j - k])
                i += j - k
        return factorization

    # smallest_cyclic_string
    @staticmethod
    def min_cyclic_string(s):
        s += s
        n = len(s)
        i, ans = 0, 0
        while i < n // 2:
            ans = i
            j, k = i + 1, i
            while j < n and s[k] <= s[j]:
                if s[k] < s[j]:
                    k = i
                else:
                    k += 1
                j += 1
            while i <= k:
                i += j - k
        return s[ans: ans + n // 2]

    @staticmethod
    def minist_express(sec):
        n = len(sec)
        k, i, j = 0, 0, 1
        while k < n and i < n and j < n:
            if sec[(i + k) % n] == sec[(j + k) % n]:
                k += 1
            else:
                if sec[(i + k) % n] > sec[(j + k) % n]:
                    i = i + k + 1
                else:
                    j = j + k + 1
                if i == j:
                    i += 1
                k = 0
        i = min(i, j)
        return sec[i:] + sec[:i]


class TestGeneral(unittest.TestCase):
    def test_solve_by_duval(self):
        ld = LyndonDecomposition()
        assert ld.solve_by_duval("ababa") == ["ab", "ab", "a"]
        return

    def test_min_cyclic_string(self):
        ld = LyndonDecomposition()
        assert ld.min_cyclic_string("ababa") == "aabab"
        assert ld.minist_express("ababa") == "aabab"
        return


if __name__ == '__main__':
    unittest.main()
