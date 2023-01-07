

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
    def rankTeams(self, votes: List[str]) -> str:
        n = len(votes[0])
        dct = defaultdict(lambda: [0]*n)
        for s in votes:
            for i, w in enumerate(votes):
                dct[w][i] -= 1

        ans = [[cnt, w] for w, cnt in dct.keys()]
        ans.sort(key=lambda x: [x[0], x[1]])
        st = "".join([w for _, w in ans])
        return st


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().countPairs(nums=[1, 4, 2, 7], low=2, high=6) == 6
        return


if __name__ == '__main__':
    unittest.main()
