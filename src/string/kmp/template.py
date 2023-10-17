import unittest
from collections import Counter
from itertools import permutations

from utils.fast_io import FastIO
from src.mathmatics.fast_power import MatrixFastPower




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
    def z_function(s):
        # 计算 s[i:] 与 s 的最长公共前缀
        n = len(s)
        z = [0] * n
        left, r = 0, 0
        for i in range(1, n):
            if i <= r and z[i - left] < r - i + 1:
                z[i] = z[i - left]
            else:
                z[i] = max(0, r - i + 1)
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
            if i + z[i] - 1 > r:
                left = i
                r = i + z[i] - 1
        # z[0] = 0
        return z

    def find(self, s1, s2):
        # 查找 s2 在 s1 中的索引位置
        n, m = len(s1), len(s2)
        pi = self.prefix_function(s2 + "#" + s1)
        ans = []
        for i in range(m + 1, m + n + 1):
            if pi[i] == m:
                ans.append(i - m - m)
        return ans

    def find_longest_palindrome(self, s, pos="prefix"):
        # 计算最长前缀与最长后缀回文子串
        if pos == "prefix":
            return self.prefix_function(s + "#" + s[::-1])[-1]
        return self.prefix_function(s[::-1] + "#" + s)[-1]

