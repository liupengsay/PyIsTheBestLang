


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
from itertools import combinations, permutations






mod = 10**9+7


class Solution:
    def countPartitions(self, nums: List[int], k: int) -> int:
        s = sum(nums)
        pre = defaultdict(int)
        pre[0] = 1
        for num in nums:
            cur = pre.copy()
            for p in pre:
                if p+num<k:
                    cur[p+num] += pre[p]
                    cur[p+num] %= mod
            pre = cur.copy()

        small = sum(pre.values())
        n = len(nums)
        total = pow(2, n, mod)
        ans = (total-2*small) % mod
        return ans



class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().isPossible(4, [[1,2],[2,3],[2,4],[3,4]]) == False

        return


if __name__ == '__main__':
    unittest.main()
