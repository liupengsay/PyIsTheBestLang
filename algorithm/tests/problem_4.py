
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


from functools import cmp_to_key

from functools import cmp_to_key

from functools import cmp_to_key


class Solution:
    def findCrossingTime(self, n: int, k: int, time: List[List[int]]) -> int:

        rank = [[-time[i][0] - time[i][2], -i] for i in range(k)]
        left_person = [[0, rank[num], num] for num in range(k)]
        heapq.heapify(left_person)

        ans = 0
        right_person = []
        on = 0
        while n:
            if right_person and not left_person:
                tm, _, idx = heapq.heappop(right_person)
                on = max(on, tm)
                n -= 1
                tm = on + time[idx][2]
                on = tm
                heapq.heappush(left_person, [on + time[idx][3], rank[idx], idx])
                ans = tm + time[idx][3]
            elif right_person and left_person and left_person[0][0] >= right_person[0][0]:
                tm, _, idx = heapq.heappop(right_person)
                on = max(on, tm)
                n -= 1
                tm = on + time[idx][2]
                on = tm
                heapq.heappush(left_person, [on + time[idx][3], rank[idx], idx])
                ans = tm + time[idx][3]

            else:
                tm, _, idx = heapq.heappop(left_person)
                tm = max(tm, on)
                on = tm + time[idx][0]
                heapq.heappush(right_person, [on + time[idx][1], rank[idx], idx])

        return ans






class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().findCrossingTime(n = 1, k = 3, time = [[1,1,2,1],[1,1,3,1],[1,1,4,1]]) == 6
        return


if __name__ == '__main__':
    unittest.main()
