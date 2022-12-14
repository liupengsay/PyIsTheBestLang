"""
"""

"""
算法：马拉车算法
功能：用来处理字符串的回文相关问题
题目：
L0214 最短回文串（https://leetcode.cn/problems/shortest-palindrome/）计算字符串前缀最长回文子串

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


class ManacherPlindrome:
    def __init__(self):
        return

    @staticmethod
    def manacher(s):
        # 马拉车算法
        n = len(s)
        arm = [0] * n
        left, right = 0, -1
        for i in range(0, n):
            k = 1 if i > right else min(arm[left + right - i], right - i + 1)

            # 持续增加回文串的长度
            while 0 <= i - k and i + k < n and s[i - k] == s[i + k]:
                k += 1
            arm[i] = k

            # 更新右侧最远的回文串边界
            k -= 1
            if i + k > right:
                left = i - k
                right = i + k
        # 返回每个位置往右的臂长其中 s[i-arm[i]+1: i+arm[i]] 为回文子串范围
        return arm

    def palindrome(self, s: str) -> (list, list):
        # 获取区间的回文串信息
        n = len(s)
        # 保证所有的回文串为奇数长度，且中心为 # 的为原偶数回文子串，中心为 字母 的为原奇数回文子串
        t = "#" + "#".join(list(s)) + "#"
        dp = self.manacher(t)
        m = len(t)

        # 以当前索引作为边界开头的回文子串结束位置索引
        start = [[] for _ in range(n)]
        # 以当前索引作为边界结尾的回文子串起始位置索引
        end = [[] for _ in range(n)]

        for j in range(m):
            left = j - dp[j] + 1
            right = j + dp[j] - 1
            while left <= right:
                if t[left] != "#":
                    start[left // 2].append(right // 2)
                    end[right//2].append(left//2)
                left += 1
                right -= 1
        # 由此还可以获得以某个位置开头或者结尾的最长回文子串
        return start, end


class TestGeneral(unittest.TestCase):

    def test_manacher_palindrome(self):
        s = "abccba"
        mp = ManacherPlindrome()
        start, end = mp.palindrome(s)
        assert start == [[0, 5], [1, 4], [2, 3], [3], [4], [5]]
        assert end == [[0], [1], [2], [2, 3], [1, 4], [0, 5]]
        return


if __name__ == '__main__':
    unittest.main()
