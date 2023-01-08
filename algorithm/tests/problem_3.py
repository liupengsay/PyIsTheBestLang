


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


def mmax(a, b):
    return a if a > b else b


def mmin(a, b):
    return a if a < b else b





class Solution:
    def isItPossible(self, word1: str, word2: str) -> bool:
        cnt1 = Counter(word1)
        cnt2 = Counter(word2)
        lst = [chr(i+ord("a")) for i in range(26)]
        for a in lst:
            for b in lst:
                cnt11 = cnt1.copy()
                cnt22 = cnt2.copy()
                cnt11[a] -= 1
                cnt11[b] += 1
                cnt22[a] += 1
                cnt22[b] -= 1
                if sum(cnt11[w] >= 1 for w in lst) == sum(cnt22[w] >= 1 for w in lst):
                    return True
        return False




class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().isPossible(4, [[1,2],[2,3],[2,4],[3,4]]) == False

        return


if __name__ == '__main__':
    unittest.main()
