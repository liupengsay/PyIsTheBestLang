

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
    def countGood(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = j = 0
        # 元素计数
        cnt = defaultdict(int)
        # 符合条件的元素对计数
        total = 0
        for i in range(n):
            while j < n and total < k:
                total += cnt[nums[j]]  # 增加元素对计数
                cnt[nums[j]] += 1  # 增加元素个数
                j += 1
            if total >= k:  # 不少于 k 对
                ans += n - j + 1
            cnt[nums[i]] -= 1
            total -= cnt[nums[i]]
        return ans


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().isPossible(
            4, [[1, 2], [2, 3], [2, 4], [3, 4]]) == False

        return


if __name__ == '__main__':
    unittest.main()
