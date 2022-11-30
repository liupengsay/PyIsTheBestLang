
import bisect
import itertools
import random
from typing import List
import heapq
import math
import re
import unittest
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import heapq
from itertools import combinations
from sortedcontainers import SortedList







class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        x = nums.index(k)
        pre = defaultdict(int)
        a = b = 0
        for i in range(x-1, -1, -1):
            if nums[i] > k:
                a += 1
            else:
                b += 1
            pre[a-b] += 1

        post = defaultdict(int)
        a = b = 0
        for i in range(x+1, n):
            if nums[i] > k:
                a += 1
            else:
                b += 1
            post[a - b] += 1

        ans = 0
        for k in pre:
            ans += pre[k]*post[-k]
            ans += pre[k]*(post[-(k-1)])
        return ans

class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().bestClosingTime("YYNY") == 2

        return


if __name__ == '__main__':
    unittest.main()
