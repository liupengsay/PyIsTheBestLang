
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

from algorithm.src.fast_io import FastIO

"""
算法：马拉车算法、回文连续子串、回文不连续子串
功能：用来处理字符串的回文相关问题，可以有暴力、DP、中心扩展法、马拉车
题目：

===================================力扣===================================
5. 最长回文子串（https://leetcode.cn/problems/longest-palindromic-substring/）计算字符串的最长回文连续子串
132. 分割回文串 II（https://leetcode.cn/problems/palindrome-partitioning-ii/）经典线性 DP 与马拉车判断以每个位置为结尾的回文串
214 最短回文串（https://leetcode.cn/problems/shortest-palindrome/）计算字符串前缀最长回文子串
1745. 回文串分割 IV（https://leetcode.cn/problems/palindrome-partitioning-iv/）待确认如何使用马拉车实现线性做法
1960. 两个回文子字符串长度的最大乘积（https://leetcode.cn/problems/maximum-product-of-the-length-of-two-palindromic-substrings/）利用马拉车求解每个位置前后最长回文子串

===================================洛谷===================================
P4555 最长双回文串（https://www.luogu.com.cn/problem/P4555）计算以当前索引为开头以及结尾的最长回文子串
P1210 [USACO1.3]最长的回文 Calf Flac（https://www.luogu.com.cn/problem/P1210）寻找最长的连续回文子串
P4888 三去矩阵（https://www.luogu.com.cn/problem/P4888）中心扩展法双指针
P1872 回文串计数（https://www.luogu.com.cn/problem/P1872）回文串对数统计，利用马拉车计算以当前字母开头与结尾的回文串数
P6297 替换（https://www.luogu.com.cn/problem/P6297）中心扩展法并使用变量维护

139. 回文子串的最大长度（https://www.acwing.com/problem/content/141/）马拉车计算最长回文子串长度，也可使用二分查找加哈希

"""


class ManacherPlindrome:
    def __init__(self):
        return

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def manacher(s):
        # 马拉车算法
        n = len(s)
        arm = [0] * n
        left, right = 0, -1
        for i in range(0, n):
            a, b = arm[left + right - i], right - i + 1
            a = a if a < b else b
            k = 1 if i > right else a

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
        # 即以i为中心的最长回文子串长度
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
        return start, end

    def palindrome_longest(self, s: str) -> (list, list):
        # 获取区间的回文串信息
        n = len(s)
        # 保证所有的回文串为奇数长度，且中心为 # 的为原偶数回文子串，中心为 字母 的为原奇数回文子串
        t = "#" + "#".join(list(s)) + "#"
        dp = self.manacher(t)
        m = len(t)

        # 由此还可以获得以某个位置开头或者结尾的最长回文子串的长度
        post = [1] * n
        pre = [1] * n

        for j in range(m):
            left = j - dp[j] + 1
            right = j + dp[j] - 1
            while left <= right:
                if t[left] != "#":
                    x, y = left // 2, right // 2
                    post[x] = self.max(post[x], y - x + 1)
                    pre[y] = self.max(pre[y], y - x + 1)
                    break
                left += 1
                right -= 1
        # 由此还可以获得以某个位置开头或者结尾的最长回文子串
        for i in range(1, n):
            if i - pre[i - 1] - 1 >= 0 and s[i] == s[i - pre[i - 1] - 1]:
                pre[i] = self.max(pre[i], pre[i - 1] + 2)

        for i in range(n - 2, -1, -1):
            pre[i] = self.max(pre[i], pre[i + 1] - 2)

        for i in range(n - 2, -1, -1):
            if i + post[i + 1] + 1 < n and s[i] == s[i + post[i + 1] + 1]:
                post[i] = self.max(post[i], post[i + 1] + 2)
        for i in range(1, n):
            post[i] = self.max(post[i], post[i - 1] - 2)

        return post, pre

    def palindrome_longest_length(self, s: str) -> (list, list):
        # 计算字符串的最长回文子串长度
        t = "#" + "#".join(list(s)) + "#"
        dp = self.manacher(t)
        m = len(t)
        ans = 0
        for j in range(m):
            left = j - dp[j] + 1
            right = j + dp[j] - 1
            cur = (right - left + 1) // 2
            ans = ans if ans > cur else cur
        return ans


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_4555(s):
        # 模板：计算长度和最大的两个回文子串的长度和，转换为求字符开头以及结尾的最长回文子串
        n = len(s)
        post, pre = ManacherPlindrome().palindrome_longest(s)
        ans = max(post[i + 1] + pre[i] for i in range(n - 1))
        return ans

    @staticmethod
    def lc_5(s: str) -> str:
        # 模板：计算字符串的最长回文子串，转换为求字符开头以及结尾的最长回文子串
        post, pre = ManacherPlindrome().palindrome_longest(s)
        i = post.index(max(post))
        return s[i: i+post[i]]

    @staticmethod
    def ac_139(ac=FastIO()):
        # 模板：马拉车计算最长回文子串的长度
        ind = 0
        while True:
            s = ac.read_str()
            if s == "END":
                break
            ind += 1
            ans = ManacherPlindrome().palindrome_longest_length(s)
            ac.st(f"Case {ind}: {ans}")
        return

    @staticmethod
    def lg_p1876(ac=FastIO()):
        # 模板：回文串对数统计，利用马拉车计算以当前字母开头与结尾的回文串数
        s = ac.read_str()
        n = len(s)
        start, end = ManacherPlindrome().palindrome(s)
        start = [len(x) for x in start]
        end = [len(x) for x in end]
        pre = ans = 0
        for i in range(n):
            ans += pre * start[i]
            pre += end[i]
        ac.st(ans)
        return

    @staticmethod
    def lg_p6297(ac=FastIO()):
        # 模板：中心扩展法并使用变量维护
        n, k = ac.read_ints()
        mod = 10**9 + 7
        nums = ac.read_list_ints()
        ans = 0
        for i in range(n):

            cur = nums[i]
            rem = k
            x, y = i - 1, i + 1
            while x >= 0 and y < n:
                if nums[x] != nums[y]:
                    if not rem:
                        break
                    rem -= 1
                cur *= nums[x] * nums[y]
                x -= 1
                y += 1
            ans = ac.max(ans, cur)

            if i + 1 < n:
                cur = 0
                rem = k
                x, y = i, i + 1
                while x >= 0 and y < n:
                    if nums[x] != nums[y]:
                        if not rem:
                            break
                        rem -= 1
                    cur = cur if cur else 1
                    cur *= nums[x] * nums[y]
                    x -= 1
                    y += 1
                ans = ac.max(ans, cur)
        ac.st(ans % mod)
        return

    @staticmethod
    def lc_1960(s: str) -> int:
        # 模板：利用马拉车求解每个位置前后最长回文子串
        post, pre = ManacherPlindrome().palindrome_longest(s)

        n = len(s)
        if post[-1] % 2 == 0:
            post[-1] = 1
        for i in range(n - 2, -1, -1):
            if post[i] % 2 == 0:
                post[i] = 1
            post[i] = post[i] if post[i] > post[i + 1] else post[i + 1]

        ans = x = 0
        for i in range(n - 1):
            if pre[i] % 2 == 0:
                pre[i] = 1
            x = x if x > pre[i] else pre[i]
            ans = ans if ans > x * post[i + 1] else x * post[i + 1]


class TestGeneral(unittest.TestCase):

    def test_manacher_palindrome(self):
        s = "abccba"
        mp = ManacherPlindrome()
        start, end = mp.palindrome(s)
        assert start == [[0, 5], [1, 4], [2, 3], [3], [4], [5]]
        assert end == [[0], [1], [2], [2, 3], [1, 4], [0, 5]]
        return

    def test_luogu(self):
        s = "baacaabbacabb"
        luogu = Luogu()
        assert luogu.lg_4555(s) == 12
        return


if __name__ == '__main__':
    unittest.main()
