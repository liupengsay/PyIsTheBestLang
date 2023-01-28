

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
    def numIslands(self, grid: List[List[str]]) -> int:

        def dfs(a, b):
            grid[a][b] = "0"
            for x, y in [[a - 1, b], [a + 1, b], [a, b - 1], [a, b + 1]]:
                if 0 <= x < m and 0 <= y < n and grid[x][y] == "1":
                    dfs(x, y)
            return

        m, n = len(grid), len(grid[0])
        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    dfs(i, j)
                    ans += 1
        return ans


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().sortArray([5, 1, 1, 2, 0, 0]) == [0, 0, 1, 1, 2, 5]
        return


if __name__ == '__main__':
    unittest.main()
