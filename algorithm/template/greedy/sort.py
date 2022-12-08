
"""
自定义有序列表
"""

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


class DefineSortedList:
    def __init__(self, lst=[]):
        self.lst = lst
        self.lst.sort()
        return

    def add(self, num):
        if not self.lst:
            self.lst.append(num)
            return
        if self.lst[-1] <= num:
            self.lst.append(num)
            return
        if self.lst[0] >= num:
            self.lst.insert(0, num)
            return
        i = 0
        j = len(self.lst) - 1
        while i < j - 1:
            mid = i + (j - i) // 2
            if self.lst[mid] >= num:
                j = mid
            else:
                i = mid
        if self.lst[i] >= num:
            self.lst.insert(i, num)
        else:
            self.lst.insert(j, num)
        return

    def discard(self, num):
        i = 0
        j = len(self.lst) - 1
        while i < j - 1:
            mid = i + (j - i) // 2
            if self.lst[mid] > num:
                j = mid
            elif self.lst[mid] < num:
                i = mid
            else:
                self.lst.pop(mid)
                return
        if self.lst[i] == num:
            self.lst.pop(i)
        elif self.lst[j] == num:
            self.lst.pop(j)
        return

    def bisect_left(self, num):
        if not self.lst:
            return 0
        if self.lst[-1] < num:
            return len(self.lst)
        if self.lst[0] > num:
            return 0
        i = 0
        j = len(self.lst) - 1
        while i < j - 1:
            mid = i + (j - i) // 2
            if self.lst[mid] >= num:
                j = mid
            else:
                i = mid
        if self.lst[i] >= num:
            return i
        return j

    def bisect_right(self, num):
        if not self.lst:
            return 0
        if self.lst[-1] <= num:
            return len(self.lst)
        if self.lst[0] > num:
            return 0

        i = 0
        j = len(self.lst) - 1
        while i < j - 1:
            mid = i + (j - i) // 2
            if self.lst[mid] <= num:
                i = mid
            else:
                j = mid
        if self.lst[j] >= num:
            return j
        return i


class TestGeneral(unittest.TestCase):
    def test_define_sorted_list(self):
        for _ in range(10):
            floor = -10**8
            ceil = 10**8
            low = -5*10**7
            high = 6*10**8
            n = 10**4
            # add
            lst = SortedList()
            define = DefineSortedList([])
            for _ in range(n):
                num = random.randint(low, high)
                lst.add(num)
                define.add(num)
            assert all(lst[i] == define.lst[i] for i in range(n))
            # discard
            for _ in range(n):
                num = random.randint(low, high)
                lst.discard(num)
                define.discard(num)
            m = len(lst)
            assert all(lst[i] == define.lst[i] for i in range(m))
            # bisect_left
            for _ in range(n):
                num = random.randint(low, high)
                lst.add(num)
                define.add(num)
            for _ in range(n):
                num = random.randint(floor, ceil)
                assert lst.bisect_left(num) == define.bisect_left(num)
            # bisect_right
            for _ in range(n):
                num = random.randint(floor, ceil)
                assert lst.bisect_right(num) == define.bisect_right(num)
        return


# 可以考虑使用bisect实现

if __name__ == '__main__':
    unittest.main()