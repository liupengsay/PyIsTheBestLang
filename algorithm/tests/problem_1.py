

import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations, accumulate
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from operator import xor, mul, add
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy
from sortedcontainers import SortedList


class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        n = len(board)

        ind = [[0] * n for _ in range(n)]
        i, j = n - 1, 0
        order = 1
        cnt = 1
        dct = []
        while i >= 0 and j >= 0:
            ind[i][j] = cnt
            dct.append([i, j])
            cnt += 1
            if order:
                j += 1
                if j >= n:
                    i -= 1
                    j = n - 1
                    order = 1 - order
            else:
                j -= 1
                if j < 0:
                    i -= 1
                    j = 0
                    order = 1 - order

        stack = [[n - 1, 0]]
        visit = [[0] * n for _ in range(n)]
        visit[n - 1][0] = 1
        step = 1
        while stack:
            nex = []
            for i, j in stack:
                cur = ind[i][j]
                ceil = cur + 6 if cur + 6 < n * n else n * n
                for x in range(cur + 1, ceil + 1):
                    a, b = dct[x-1]
                    if board[a][b] != -1:
                        x = board[a][b]
                        a, b = dct[x-1]
                    if x == n * n:
                        return step
                    if not visit[a][b]:
                        visit[a][b] = step
                        nex.append([a, b])
            step += 1
            stack = nex
        return -1


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().snakesAndLadders([[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,35,-1,-1,13,-1],[-1,-1,-1,-1,-1,-1],[-1,15,-1,-1,-1,-1]])==4
        return


if __name__ == '__main__':
    unittest.main()
