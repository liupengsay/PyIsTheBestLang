
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

class Solution:
    def minTime(self, time: List[int], m: int) -> int:

        def check(x):
            res = 0
            pre_max = 0
            cur = 0
            for num in time:
                if cur-pre_max + num > x:
                    res += 1
                    pre_max = 0
                    cur = 0
                else:
                    cur += num
                    pre_max = pre_max if pre_max > num else num
            return res <= m

        low = 0
        high = max(time)
        while low < high-1:
            mid = low+(high-low)//2
            if check(mid):
                high = mid
            else:
                low = mid
        return low if check(low) else high


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().maxOutput(
            n=4, edges=[[2, 0], [0, 1], [1, 3]], price=[2, 3, 1, 1]) == 6
        return


if __name__ == '__main__':
    unittest.main()
