import bisect
import random
import re
import sys
import unittest
from typing import List, Callable
from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache, cmp_to_key
from itertools import combinations, accumulate, chain, count
from functools import reduce
from heapq import heappush, heappop, heappushpop, heapify
from operator import xor, mul, add, or_
from functools import lru_cache
from math import inf
import random
from itertools import permutations, combinations

from decimal import Decimal

import heapq
import copy

from src.data_structure.sorted_list.template import SortedList


# sys.set_int_max_str_digits(0)  # for big number in leet code


def max(a, b):
    return a if a > b else b


def min(a, b):
    return a if a < b else b


class Solution:

    @staticmethod
    def example(n) -> int:

        ans = [[0] * n for _ in range(n)]

        def dfs(x):
            nonlocal res
            if x == n * n or all(x==1 for x in row) or all(x==1 for x in col):
                #print([a for a in ans])
                lst = []
                for a in ans:
                    lst.extend(a)
                res.add(tuple(lst))
                return

            for i in range(n):
                for j in range(n):
                    if row[i] == 0 and col[j] == 0 and ans[i][j] == 0:
                        ans[i][j] = 1
                        row[i] = col[j] = 1
                        if i != j:
                            ans[j][i] = -1
                            row[j] = col[i] = 1
                        dfs(x+1)
                        row[i] = col[j] = 0
                        ans[i][j] = 0
                        if i != j:
                            ans[j][i] = 0
                            row[j] = col[i] = 0
            return

        res = set()
        row = [0]*n
        col = [0]*n
        dfs(0)
        return res


class TestGeneral(unittest.TestCase):

    def test_example(self):
        lst = []
        for x in range(1, 8):
            y = len(Solution().example(x))
            #print(x, y)
            lst.append(y)
        print(",".join(str(x) for x in lst))
        return


if __name__ == '__main__':
    unittest.main()
