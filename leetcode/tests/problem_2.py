
import bisect
import re
import unittest
from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet
from itertools import combinations
from sortedcontainers import SortedDict

from sortedcontainers import SortedList, SortedDict, SortedSet
from itertools import combinations, permutations
from sortedcontainers import SortedDict
from decimal import Decimal

from collections import deque

from sortedcontainers import SortedList


class Solution:
    def subarrayLCM(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            x = 1
            for j in range(i, n):
                x = math.lcm(x, nums[j])
                if x == k:
                    ans += 1
                elif x > k:
                    break
        return ans


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution()
        return


if __name__ == '__main__':
    unittest.main()
