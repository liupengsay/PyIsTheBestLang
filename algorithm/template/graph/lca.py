"""
算法：LCA，使用Tarjan、Range Maximum Query（RMQ）、和倍增算法求解

# 最近公共祖先算法:
# 通常解决这类问题有两种方法：在线算法和离线算法
# 在线算法：每次读入一个查询，处理这个查询，给出答案
# 离线算法：一次性读入所有查询，统一进行处理，给出所有答案
# 我们接下来介绍一种离线算法：Tarjan，两种在线算法：RMQ,倍增算法
# Tarjan的时间复杂度是 O(n+q)
# RMQ是一种先进行 O(nlogn) 预处理，然后O(1)在线查询的算法。
# 倍增算法是一种时间复杂度 O((n+q)logn)的算法
# ————————————————
# 版权声明：本文为CSDN博主「weixin_42001089」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/weixin_42001089/article/details/83590686

功能：来求一棵树的最近公共祖先（LCA）
题目：P3379 【模板】最近公共祖先（LCA）（https://www.luogu.com.cn/problem/P3379）
参考：OI WiKi（）
CSDN（https://blog.csdn.net/weixin_42001089/article/details/83590686）

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


class LcaTarjan:
    def __init__(self, dct, root):
        self.dct = dct
        self.depth = dict()
        self.father = dict()
        self.maxstep = 20  # 最多向上跳2^maxstep步

        self.depth[root] = 1
        for x in dct[root]:
            self.dfs(root, x)

    def dfs(self, prev, rt):
        self.father[rt] = [prev]
        self.depth[rt] = self.depth[prev] + 1
        for i in range(1, self.maxstep):
            if self.father[rt][i - 1] in self.father and len(
                    self.father[self.father[rt][i - 1]]) >= i:
                self.father[rt].append(
                    self.father[self.father[rt][i - 1]][i - 1])
            else:
                break
        for x in self.dct[rt]:
            self.dfs(rt, x)
        return

    def LCA(self, root, m, n):
        # 因为 self.f中没有根节点，所以这里判断一下，如果其中一个是根节点，那么其LCA必是根节点，直接返回即可
        if m == root or n == root:
            return root
        if self.depth[n] < self.depth[m]:
            temp = m
            m = n
            n = temp
        # 目的就是将m和n的深度调为一样
        for i in range(len(self.father[n])):
            if self.depth[n] - self.depth[m] >= 2 ** (len(self.father[n]) - i - 1):
                n = self.father[n][len(self.father[n]) - i - 1]
        if n == m:
            return n
        # 两者一同向上跳，注意这里的length的重要性
        length = len(self.father[n])
        for i in range(length):
            if self.father[n][length - i -
                              1] != self.father[m][length - i - 1]:
                n = self.father[n][length - i - 1]
                m = self.father[m][length - i - 1]
        return self.father[m][0]


class LcaRMQ:
    def __init__(self, dct, root):
        self.dct = dct
        self.root = root

        # 记录每一个元素对应的深度
        self.R = []

        # 记录遍历过的元素对应
        self.ves = []

        # 记录每一个区段的最值，它的结构是这样{1: [1, 1, 1, 1, 1], 2: [2, 2, 2, 2, 11],.........}
        # 2: [2, 2, 2, 2, 11]比如代表的意义就是从第二个位置开始，长度为1的区间中(本身)深度最浅元素的位置是2，长度为2的区间中深度最浅元素的位置是2
        # 长度为4的区间中(本身)深度最浅元素的位置是2，长度为8的区间中(本身)深度最浅元素的位置是2，长度为16的区间中(本身)深度最浅元素的位置是11
        self.dp = {}
        # 记录每一个元素在欧拉序中出现的第一个位置
        self.first = {}

        self.dfs(root, 1)
        self.ST(len(self.R))
        return

    def dfs(self, root, depth):
        self.R.append(depth)
        self.ves.append(root)
        if root not in self.first:
            self.first[root] = len(self.ves)
        for x in self.dct[root]:
            self.dfs(x, depth + 1)
            self.R.append(depth)
            self.ves.append(root)
        return

    def ST(self, lenth):
        K = int(math.log(lenth, 2))
        for i in range(lenth):
            self.dp[i + 1] = [i + 1]
        for j in range(1, K + 1):
            i = 1
            while i + 2 ** j - 1 <= lenth:
                a = self.dp[i][j - 1]
                b = self.dp[i + 2 ** (j - 1)][j - 1]
                if self.R[a - 1] <= self.R[b - 1]:
                    self.dp[i].append(a)
                else:
                    self.dp[i].append(b)
                i += 1
        return

    def LCA(self, f, g):
        if self.first[f] < self.first[g]:
            c = self.RMQ(self.first[f], self.first[g])
        else:
            c = self.RMQ(self.first[g], self.first[f])
        return self.ves[c - 1]

    def RMQ(self, m, n):
        K = int(math.log(n - m + 1, 2))
        a = self.dp[m][K]
        b = self.dp[n - 2 ** K + 1][K]
        if self.R[a - 1] < self.R[b - 1]:
            return a
        return b


class LcaMultiply:
    def __init__(self, dct, root):
        self.dct = dct
        self.root = root


class TestGeneral(unittest.TestCase):

    def test_lca_tarjan(self):
        g = {3: [5, 1], 5: [6, 2], 2: [7, 4], 1: [0, 8]}
        dct = defaultdict(list)
        for k in g:
            dct[k].extend(g[k])
        root = 3
        lt = LcaTarjan(dct, root)
        assert lt.LCA(root, 7, 8) == root
        return


    def test_lca_rmq(self):
        g = {3: [5, 1], 5: [6, 2], 2: [7, 4], 1: [0, 8]}
        dct = defaultdict(list)
        for k in g:
            dct[k].extend(g[k])
        root = 3
        lt = LcaRMQ(dct, root)
        assert lt.LCA(7, 8) == root
        return

if __name__ == '__main__':
    unittest.main()
