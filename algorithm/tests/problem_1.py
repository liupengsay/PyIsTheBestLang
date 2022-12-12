

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

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

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


        dct = defaultdict(list)

        lca = LcaRMQ(dct, root.val)
        lca.LCA(p.val, q.val)


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().minCost(7, [1, 3, 4, 5]) == 11
        return


if __name__ == '__main__':
    unittest.main()
