

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
    def simplifyPath(self, path: str) -> str:
        lst = [w for w in path.split("/") if w]
        stack = []
        for w in lst:
            if w == ".":
                continue
            if w == "..":
                if stack:
                    stack.pop()
            else:
                stack.append(w)
        ans = "/" + "/".join(stack)
        return ans


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().countPairs(nums=[1, 4, 2, 7], low=2, high=6) == 6
        return


if __name__ == '__main__':
    unittest.main()
