

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


# """
# This is BinaryMatrix's API interface.
# You should not implement it, or speculate about its implementation
# """
#class BinaryMatrix(object):
#    def get(self, row: int, col: int) -> int:
#    def dimensions(self) -> list[]:
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        inf = float("inf")
        m, n = len(word1), len(word2)
        dp = [[inf]*(n+1) for _ in range(m+1)]
        dp[0][0] = 0
        for i in range(m):
            dp[i+1][0] = i+1
            for j in range(n):
                dp[i+1][j+1] = min(dp[i][j+1]+1, dp[i+1][j]+1, dp[i][j]+int(word1[i]!=word2[j]))
        return dp[m][n]




class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().sortArray([5,1,1,2,0,0]) == [0,0,1,1,2,5]
        return


if __name__ == '__main__':
    unittest.main()
