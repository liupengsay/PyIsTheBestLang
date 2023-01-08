
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
    def maxKelements(self, nums: List[int], k: int) -> int:

        heapq.heapify([-num for num in nums])
        ans = 0
        for _ in range(k):
            x = heapq.heappop(nums)
            ans += x
            heapq.heappush(nums, math.ceil(x/3))
        return ans


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().smallestValue(4) == 4
        return


if __name__ == '__main__':
    unittest.main()
