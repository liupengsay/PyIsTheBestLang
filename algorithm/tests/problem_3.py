

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
    def makeStringsEqual(self, s: str, target: str) -> bool:
        cnt = [0, 0]
        if len(s) != len(target):
            return False
        n = len(s)
        for i in range(n):
            if s[i] != target[i]:
                if s[i] == "1":
                    cnt[1] += 1
                else:
                    cnt[0] += 1
        one = "1" in s
        zero = "0" in s


        if cnt[0]:


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().isPossible(
            4, [[1, 2], [2, 3], [2, 4], [3, 4]]) == False

        return


if __name__ == '__main__':
    unittest.main()
