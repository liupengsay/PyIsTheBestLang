
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
    def partitionDisjoint(self, nums: List[int]) -> int:
        n = len(nums)
        ans = n
        post = [-1]*(n+1)
        post[n] = float("inf")
        post[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            post[i] = post[i+1] if post[i+1] < nums[i] else nums[i]

        pre = float("-inf")
        for i in range(n):
            pre = pre if pre > nums[i] else nums[i]
            if pre < post[i+1]:
                return i+1


class Solution:
    def distinctPrimeFactors(self, nums: List[int]) -> int:
        ans = set()
        for num in nums:
            for x in get_prime_factor(num):
                if x > 1:
                    ans.add(x)
        return len(ans)




class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().smallestValue(4) == 4
        return


if __name__ == '__main__':
    unittest.main()
