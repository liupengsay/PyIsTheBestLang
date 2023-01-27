

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
    def restoreIpAddresses(self, s: str) -> List[str]:
        lst = list(s)
        n = len(lst)
        def check(x):
            if len(x) >= 2 and x[0] == "0":
                return False
            return 0<=int(x)<=255

        ans = []
        for item in combinations(list(range(1, n)), 3):
            tmp = lst[:]
            for i in item[::-1]:
                tmp.insert(i, ".")
            t = "".join(tmp)
            cur = t.split(".")
            if all(check(x) for x in cur):
                ans.append(t)
        return ans


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().sortArray([5, 1, 1, 2, 0, 0]) == [0, 0, 1, 1, 2, 5]
        return


if __name__ == '__main__':
    unittest.main()
