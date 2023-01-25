"""

"""
"""
算法：Trie字典树，也叫前缀树
功能：处理字符串以及结合位运算相关，01Trie通用用于查询位运算极值
题目：

P8306 字典树（https://www.luogu.com.cn/problem/P8306）

L2416 字符串的前缀分数和（https://leetcode.cn/problems/sum-of-prefix-scores-of-strings/）单词组前缀计数
P4551 最长异或路径（https://www.luogu.com.cn/problem/P4551）关键是利用异或的性质，将任意根节点作为中转站
1803. 统计异或值在范围内的数对有多少（https://leetcode.cn/problems/count-pairs-with-xor-in-a-range/）
677. 键值映射（https://leetcode.cn/problems/map-sum-pairs/）
P3864 [USACO1.2]命名那个数字 Name That Number（https://www.luogu.com.cn/problem/P3864）使用哈希枚举或者进行字典树存储

参考：OI WiKi（）
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


class TriePrefixKeyValue:
    # L677
    def __init__(self):
        self.dct = dict()
        return

    def update(self, word, val):
        # 更新单词与前缀计数
        cur = self.dct
        for w in word:
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
        cur["val"] = val
        return

    def query(self, word):
        # 查询前缀单词个数
        cur = self.dct
        for w in word:
            if w not in cur:
                return 0
            cur = cur[w]

        def dfs(dct):
            nonlocal res
            if "val" in dct:
                res += dct["val"]
            for s in dct:
                if s != "val":
                    dfs(dct[s])
            return
        res = 0
        dfs(cur)
        return res


class TrieCount:
    def __init__(self):
        self.dct = dict()
        return

    def update(self, word):
        # 更新单词与前缀计数
        cur = self.dct
        for w in word:
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
            cur["cnt"] = cur.get("cnt", 0) + 1
        return

    def query(self, word):
        # 查询前缀单词个数
        cur = self.dct
        for w in word:
            if w not in cur:
                return 0
            cur = cur[w]
        return cur["cnt"]


class TrieBit:
    def __init__(self):
        self.dct = dict()
        self.n = 32
        return

    def update(self, num):
        cur = self.dct
        for i in range(self.n, -1, -1):
            w = 1 if num & (1 << i) else 0
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
            cur["cnt"] = cur.get("cnt", 0) + 1
        return

    def query(self, num):
        cur = self.dct
        ans = 0
        for i in range(self.n, -1, -1):
            w = 1 if num & (1 << i) else 0
            if 1 - w in cur:
                cur = cur[1 - w]
                ans += 1 << i
            else:
                cur = cur[w]
        return ans

    def delete(self, num):
        cur = self.dct
        for i in range(self.n, -1, -1):
            w = num & (1 << i)
            if cur[w].get("cnt", 0) == 1:
                del cur[w]
                break
            cur = cur[w]
            cur["cnt"] -= 1
        return


class TrieBitRange:
    # 查询范围类的异或值数对L1803
    def __init__(self):
        self.dct = dict()
        self.n = len(bin(2 * 10 ** 4)) - 1
        return

    def update(self, num, c):
        cur = self.dct
        for i in range(self.n, -1, -1):
            w = 1 if num & (1 << i) else 0
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
            cur["cnt"] = cur.get("cnt", 0) + c
        return

    def query(self, num, ceil):

        def dfs(cur, i, val):
            nonlocal res
            if val + ((1 << (i + 1)) - 1) <= ceil:
                res += cur["cnt"]
                return

            if i == -1 or val > ceil:
                return

            w = 1 if num & (1 << i) else 0
            if 1 - w in cur:
                dfs(cur[1 - w], i - 1, val | (1 << i))
            if w in cur:
                dfs(cur[w], i - 1, val)
            return

        res = 0
        dfs(self.dct, self.n, 0)
        return res


class TestGeneral(unittest.TestCase):

    def test_trie_count(self):
        tc = TrieCount()
        words = ["happy", "hello", "leetcode", "let"]
        for word in words:
            tc.update(word)
        assert tc.query("h") == 2
        assert tc.query("le") == 2
        assert tc.query("lt") == 0
        return


if __name__ == '__main__':
    unittest.main()
