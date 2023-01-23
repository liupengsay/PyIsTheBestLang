"""
算法：KMP算法
功能：用来处理字符串的前缀后缀相关问题
题目：
P3375 KMP字符串匹配（https://www.luogu.com.cn/problem/P3375）计算子字符串出现的位置，与最长公共前后缀的子字符串长度
L0796 旋转字符串（https://leetcode.cn/problems/rotate-string/）计算字符串是否可以旋转得到
L0025 找出字符串中第一个匹配项的下标（https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/）计算子字符串第一次出现的位置
L0214 最短回文串（https://leetcode.cn/problems/shortest-palindrome/）计算字符串前缀最长回文子串
L1392 最长快乐前缀（https://leetcode.cn/problems/longest-happy-prefix/）计算最长的公共前后缀
L2223 构造字符串的总得分和（https://leetcode.cn/problems/longest-happy-prefix/）利用扩展KMP计算Z函数
P4391 无线传输（https://www.luogu.com.cn/problem/P4391）脑经急转弯加KMP算法，最优结果为 n-pi[n-1]
参考：OI WiKi（https://oi-wiki.org/string/kmp/）
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


class KMP:
    def __init__(self):
        return

    @staticmethod
    def prefix_function(s):
        # 计算s[:i]与s[:i]的最长公共真前缀与真后缀
        n = len(s)
        pi = [0] * n
        for i in range(1, n):
            j = pi[i - 1]
            while j > 0 and s[i] != s[j]:
                j = pi[j - 1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j
        # pi[0] = 0
        return pi

    @staticmethod
    def find(s1, s2):
        # 查找 s2 在 s1 中的索引位置
        n, m = len(s1), len(s2)
        s = s2 + "#" + s1
        pi = KMP().prefix_function(s2 + "#" + s1)
        ans = []
        for i in range(m + 1, m + n + 1):
            if pi[i] == m:
                ans.append(i - m - m)
        return ans

    @staticmethod
    def z_function(s):
        # 计算 s[i:] 与 s 的最长公共前缀
        n = len(s)
        z = [0] * n
        l, r = 0, 0
        for i in range(1, n):
            if i <= r and z[i - l] < r - i + 1:
                z[i] = z[i - l]
            else:
                z[i] = max(0, r - i + 1)
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
            if i + z[i] - 1 > r:
                l = i
                r = i + z[i] - 1
        # z[0] = 0
        return z


class TestGeneral(unittest.TestCase):

    def test_prefix_function(self):
        kmp = KMP()
        assert kmp.prefix_function("abcabcd") == [0, 0, 0, 1, 2, 3, 0]
        assert kmp.prefix_function("aabaaab") == [0, 1, 0, 1, 2, 2, 3]
        return

    def test_z_function(self):
        kmp = KMP()
        assert kmp.z_function("abacaba") == [0, 0, 1, 0, 3, 0, 1]
        assert kmp.z_function("aaabaab") == [0, 2, 1, 0, 2, 1, 0]
        assert kmp.z_function("aaaaa") == [0, 4, 3, 2, 1]
        return

    def test_find(self):
        kmp = KMP()
        assert kmp.find("abababc", "aba") == [0, 2]
        assert kmp.find("aaaa", "a") == [0, 1, 2, 3]
        assert kmp.find("aaaa", "aaaa") == [0]
        return


if __name__ == '__main__':
    unittest.main()
