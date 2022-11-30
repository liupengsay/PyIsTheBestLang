
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
    def appendCharacters(self, s: str, t: str) -> int:

        n = len(t)
        i = 0
        for w in s:
            if w == t[i]:
                i += 1
                if i == n:
                    return 0
        return n-i




class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().minSwap(nums1=[1, 3, 5, 4], nums2=[1, 2, 3, 7]) == 1
        return


if __name__ == '__main__':
    unittest.main()
