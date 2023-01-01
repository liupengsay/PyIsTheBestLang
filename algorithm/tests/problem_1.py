

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
    def lengthOfLongestSubstring(self, s: str) -> int:
        ans = j = 0
        n = len(s)
        pre =set()
        for i in range(n):
            while j < n and s[j] not in pre:
                pre.add(s[j])
                j += 1
            ans = ans if ans > j-i else j-i
            pre.discard(s[i])
        return ans


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().minCost(7, [1, 3, 4, 5]) == 11
        return


if __name__ == '__main__':
    unittest.main()
