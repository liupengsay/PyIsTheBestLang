"""
算法：拓扑排序
功能：有向图
题目：xx（xx）
参考：OI WiKi（xx）
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


class TopologicalSort:
    def __init__(self):
        return

    @staticmethod
    def get_rank(n, edges):
        dct = [[] for _ in range(n)]
        degree = [0]*n
        for i, j in edges:
            degree[j] += 1
            dct[i].append(j)
        stack = [i for i in range(n) if not degree[i]]
        visit = [-1]*n
        step = 0
        while stack:
            for i in stack:
                visit[i] = step
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
            step += 1
        return visit


class TestGeneral(unittest.TestCase):

    def test_topological_sort(self):
        ts = TopologicalSort()
        n = 5
        edges = [[0, 1], [0, 2], [1, 4], [2, 3], [3, 4]]
        assert ts.get_rank(n, edges) == [0, 1, 1, 2, 3]
        return


if __name__ == '__main__':
    unittest.main()
