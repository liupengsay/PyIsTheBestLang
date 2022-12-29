

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
    def twoOutOfThree(self, nums1: List[int], nums2: List[int], nums3: List[int]) -> List[int]:
        cnt = defaultdict(int)
        for num in set(nums1):
            cnt[num] += 1

        for num in set(nums2):
            cnt[num] += 1

        for num in set(nums3):
            cnt[num] += 1

        return [num for num in cnt if cnt[num]>=2]

class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().minCost(7, [1, 3, 4, 5]) == 11
        return


if __name__ == '__main__':
    unittest.main()
