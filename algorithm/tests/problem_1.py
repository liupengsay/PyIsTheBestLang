

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


def mmax(a, b):
    return a if a > b else b


def mmin(a, b):
    return a if a < b else b



class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float("inf")]*(amount+1)
        dp[0] = 1
        for num in coins:
            for i in range(num, amount+1):
                if dp[i-num] + 1 < dp[i]:
                    dp[i] = dp[i-num] + 1
        if dp[-1] < float("inf"):
            return dp[-1]
        return -1

class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().countPairs(nums=[1, 4, 2, 7], low=2, high=6) == 6
        return


if __name__ == '__main__':
    unittest.main()
