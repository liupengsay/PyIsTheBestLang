

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

算法：字符串哈希
功能：将一定长度的字符串映射为多项式函数值，并进行比较或者计数，通常结合滑动窗口进行计算
题目：xx（xx）
L0214 最短回文串（https://leetcode.cn/problems/shortest-palindrome/）使用正向与反向字符串哈希计算字符串前缀最长回文子串
L1044 最长重复子串（https://leetcode.cn/problems/shortest-palindrome/）利用二分查找加字符串哈希确定具有最长长度的重复子串
L1316 不同的循环子字符串（https://leetcode.cn/problems/shortest-palindrome/）利用字符串哈希确定不同循环子串的个数
L2156 查找给定哈希值的子串（https://leetcode.cn/problems/find-substring-with-given-hash-value/）逆向进行字符串哈希的计算
P8835 [传智杯 #3 决赛] 子串（https://www.luogu.com.cn/record/list?user=739032&status=12&page=14）字符串哈希或者KMP查找匹配的连续子串
P6140 [USACO07NOV]Best Cow Line S（https://www.luogu.com.cn/problem/P6140）贪心模拟与字典序比较，使用字符串哈希与二分查找比较正序与倒序最长公共子串
P2870 [USACO07DEC]Best Cow Line G（https://www.luogu.com.cn/problem/P2870）贪心模拟与字典序比较，使用字符串哈希与二分查找比较正序与倒序最长公共子串
P5832 [USACO19DEC]Where Am I? B（https://www.luogu.com.cn/problem/P5832）可以使用字符串哈希进行最长的长度使得所有对应长度的子串均是唯一的


参考：OI WiKi（xx）
"""


class StringHash:
    # 注意哈希碰撞，需要取两个质数与模进行区分
    def __init__(self):
        return

    @staticmethod
    def gen_hash_prime_mod(n):
        # 也可以不提前进行计算，滚动进行还行 x=(x*p+y)%mod 更新
        p1 = random.randint(26, 100)
        p2 = random.randint(26, 100)
        mod1 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)
        mod2 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)

        dp1 = [1]
        for _ in range(1, n + 1):
            dp1.append((dp1[-1] * p1) % mod1)
        dp2 = [1]
        for _ in range(1, n + 1):
            dp2.append((dp2[-1] * p2) % mod2)
        return p1, p2, mod1, mod2, dp1, dp2


class TestGeneral(unittest.TestCase):

    def test_string_hash(self):
        n = 1000
        st = "".join([chr(random.randint(0, 25) + ord("a")) for _ in range(n)])

        # 生成哈希种子
        p1, p2 = random.randint(26, 100), random.randint(26, 100)
        mod1, mod2 = random.randint(
            10 ** 9 + 7, 2 ** 31 - 1), random.randint(10 ** 9 + 7, 2 ** 31 - 1)

        # 计算目标串的哈希状态
        target = "".join([chr(random.randint(0, 25) + ord("a"))
                         for _ in range(10)])
        h1 = h2 = 0
        for w in target:
            h1 = h1 * p1 + (ord(w) - ord("a"))
            h1 %= mod1
            h2 = h2 * p2 + (ord(w) - ord("a"))
            h2 %= mod1

        # 滑动窗口计算哈希状态
        m = len(target)
        pow1 = pow(p1, m - 1, mod1)
        pow2 = pow(p2, m - 1, mod2)
        s1 = s2 = 0
        cnt = 0
        n = len(st)
        for i in range(n):
            w = st[i]
            s1 = s1 * p1 + (ord(w) - ord("a"))
            s1 %= mod1
            s2 = s2 * p2 + (ord(w) - ord("a"))
            s2 %= mod1
            if i >= m - 1:
                if (s1, s2) == (h1, h2):
                    cnt += 1
                s1 = s1 - (ord(st[i - m + 1]) - ord("a")) * pow1
                s1 %= mod1
                s2 = s2 - (ord(st[i - m + 1]) - ord("a")) * pow2
                s2 %= mod1
            if st[i:i + m] == target:
                cnt -= 1
        assert cnt == 0
        return


if __name__ == '__main__':
    unittest.main()
