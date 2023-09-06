import bisect
import random
import re
import sys
import unittest
from typing import List, Callable
from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache, cmp_to_key
from itertools import combinations, accumulate, chain
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from heapq import heappush, heappop, heappushpop, heapify
from operator import xor, mul, add
from functools import lru_cache
from math import inf
import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy
from sortedcontainers import SortedList

# sys.set_int_max_str_digits(0)  # 大数的范围坑


def ac_max(a, b):
    return a if a > b else b


def ac_min(a, b):
    return a if a < b else b



class SegmentTreeLongestSubSameNode:
    def __init__(self, pref=(-1, 0), suf=(-1, 0), ma=0):
        self.pref = pref
        self.suf = suf
        self.ceil = ma
        return


class SegmentTreeLongestSubSame:
    # 模板：单点字母更新，最长具有相同字母的连续子数组查询
    def __init__(self, n, lst):
        self.n = n
        self.lst = lst
        self.pref = [0] * 4 * n
        self.suf = [0] * 4 * n
        self.ceil = [0] * 4 * n
        self.build()
        return

    def build(self):
        # 使用数组初始化线段树
        stack = [[0, self.n - 1, 1]]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self.make_tag(ind)
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self.push_up(ind, s, t)
        return

    def make_tag(self, i):
        self.pref[i] = 1
        self.suf[i] = 1
        self.ceil[i] = 1
        return

    def push_up(self, i, s, t):
        m = s + (t - s) // 2
        # [s, m] 与 [m+1, t]
        self.pref[i] = self.pref[2 * i]
        if self.pref[2 * i] == m - s + 1 and self.lst[m] == self.lst[m + 1]:
            self.pref[i] += self.pref[2 * i + 1]

        self.suf[i] = self.suf[2 * i + 1]
        if self.suf[2 * i + 1] == t - m and self.lst[m] == self.lst[m + 1]:
            self.suf[i] += self.suf[2 * i]

        a = -inf
        for b in [self.pref[i], self.suf[i], self.ceil[2*i], self.ceil[2*i+1]]:
            a = a if a > b else b
        if self.lst[m] == self.lst[m + 1]:
            b = self.suf[2 * i] + self.pref[2 * i + 1]
            a = a if a > b else b
        self.ceil[i] = a
        return

    def update_point(self, left, right, s, t, val, i):
        # 更新单点最小值
        self.lst[left] = val
        stack = []
        while True:
            stack.append([s, t, i])
            if left <= s <= t <= right:
                self.make_tag(i)
                break
            m = s + (t - s) // 2
            if left <= m:  # 注意左右子树的边界与范围
                s, t, i = s, m, 2 * i
            if right > m:
                s, t, i = m + 1, t, 2 * i + 1
        stack.pop()
        while stack:
            s, t, i = stack.pop()
            self.push_up(i, s, t)
        assert i == 1
        return self.ceil[1]


class Solution:
    def longestRepeating(self, s: str, queryCharacters: str, queryIndices: List[int]) -> List[int]:
        n = len(s)
        tree = SegmentTreeLongestSubSame(n, [ord(w) - ord("a") for w in s])
        ans = []
        for i, w in zip(queryIndices, queryCharacters):
            ans.append(tree.update_point(i, i, 0, n - 1, ord(w) - ord("a"), 1))
        return ans




assert Solution().longestRepeating(s = "babacc", queryCharacters = "bcb", queryIndices = [1,3,3]) == [3, 3, 4]
assert Solution().longestRepeating(s = "geuqjmt", queryCharacters = "bgemoegklm", queryIndices = [3,4,2,6,5,6,5,4,3,2]) == [1,1,2,2,2,2,2,2,2,1]
