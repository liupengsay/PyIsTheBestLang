

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



class Kosaraju:
    def __init__(self, n, g):
        self.n = n
        self.g = g
        self.g2 = [[] for _ in range(self.n)]
        self.vis = [False] * n
        self.s = []
        self.color = [0] * n
        self.sccCnt = 0
        self.gen_reverse_graph()
        self.kosaraju()

    def gen_reverse_graph(self):
        for i in range(self.n):
            for j in self.g[i]:
                self.g2[j].append(i)
        return

    def dfs1(self, u):
        self.vis[u] = True
        for v in self.g[u]:
            if not self.vis[v]:
                self.dfs1(v)
        self.s.append(u)
        return

    def dfs2(self, u):
        self.color[u] = self.sccCnt
        for v in self.g2[u]:
            if not self.color[v]:
                self.dfs2(v)
        return

    def kosaraju(self):
        for i in range(self.n):
            if not self.vis[i]:
                self.dfs1(i)
        for i in range(self.n - 1, -1, -1):
            if not self.color[self.s[i]]:
                self.sccCnt += 1
                self.dfs2(self.s[i])
        self.color = [c-1 for c in self.color]
        return


class Solution:
    def longestCycle(self, edges: List[int]) -> int:
        n = len(edges)
        edge = [[] for _ in range(n)]
        ans = 0
        for i in range(n):
            if edges[i] != -1:
                if i == edges[i]:
                    ans = 1
                else:
                    edge[i].append(edges[i])

        kosaraju = Kosaraju(n, edge)
        cnt = max(Counter(kosaraju.color).values())
        if cnt > 1:
            ans = cnt
        return ans if ans > 0 else -1


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().isValid("(]") == False
        return


if __name__ == '__main__':
    unittest.main()
