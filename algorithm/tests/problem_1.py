

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


class Solution:
    def minDistance(self, houses: List[int], k: int) -> int:
        n = len(houses)
        cost = [[0] * n for _ in range(n)]
        for i in range(n - 2, -1, -1):
            cost[i][i + 1] = houses[i + 1] - houses[i]
            for j in range(i + 2, n):
                cost[i][j] = cost[i + 1][j - 1] + houses[j] - houses[i]

        dp = [[float("inf")] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(n):
            dp[i + 1][1] = cost[0][i]
            for j in range(2, k + 1):
                dp[i + 1][j] = min((dp[x][j - 1] + cost[x][i]) for x in range(i + 1))
        return dp[n][k]



class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().maxValue(events = [[1,2,4],[3,4,3],[2,3,10]], k = 2) == 10
        return


if __name__ == '__main__':
    unittest.main()
