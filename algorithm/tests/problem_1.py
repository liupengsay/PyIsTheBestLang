

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
from sortedcontainers import SortedList


class Solution:
    def findingUsersActiveMinutes(self, logs: List[List[int]], k: int) -> List[int]:
        dct = defaultdict(set)
        for i, t in logs:
            dct[i].add(t)
        cnt = Counter([len(dct[i]) for i in dct])
        return [cnt[i+1] for i in range(k)]


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().sortArray([5,1,1,2,0,0]) == [0,0,1,1,2,5]
        return


if __name__ == '__main__':
    unittest.main()
