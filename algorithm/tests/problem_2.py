
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

from __future__ import division
import copy
import random
import heapq
import math
import sys
import bisect
import time
import datetime
from functools import lru_cache
from collections import deque
from collections import Counter
from collections import defaultdict
from itertools import combinations
from itertools import permutations
from decimal import Decimal, getcontext, MAX_PREC
from types import GeneratorType
from functools import cmp_to_key

inf = float("inf")
sys.setrecursionlimit(10000000)

getcontext().prec = MAX_PREC


def get_diff_matrix(m, n, shifts):
    # 二维差分数组
    diff = [[0] * (n + 2) for _ in range(m + 2)]
    # 索引从1开始，矩阵初始值为0
    for xa, ya, xb, yb in shifts:
        diff[xa][ya] += 1
        diff[xa][yb + 1] -= 1
        diff[xb + 1][ya] -= 1
        diff[xb + 1][yb + 1] += 1

    for i in range(1, m + 2):
        for j in range(1, n + 2):
            diff[i][j] += diff[i - 1][j] + diff[i][j - 1] - diff[i - 1][j - 1]

    for i in range(1, m + 1):
        diff[i] = diff[i][1:n + 1]

    return diff[1: m + 1]


class Solution:
    def rangeAddQueries(self, n: int, queries: List[List[int]]) -> List[List[int]]:
        queries = [[x + 1 for x in ls] for ls in queries]
        # print(queries)
        ans = get_diff_matrix(n, n, queries)
        return ans


class Solution:
    def rangeAddQueries(self, n: int, queries: List[List[int]]) -> List[List[int]]:
        queries = [[x + 1 for x in ls] + [1] for ls in queries]
        # print(queries)
        ans = get_diff_matrix(n, n, queries)
        return ans



class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().smallestValue(4) == 4
        return


if __name__ == '__main__':
    unittest.main()
