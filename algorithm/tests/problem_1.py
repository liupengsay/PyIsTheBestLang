

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
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        n, m = len(nums1), len(nums2)

        def get_kth_num(k):
            ind1 = ind2 = 0
            while k:
                if ind1 == n:
                    return nums2[ind2 + k - 1]
                if ind2 == m:
                    return nums1[ind1 + k - 1]
                index1 = min(ind1 + k // 2 - 1, n - 1)
                index2 = min(ind2 + k // 2 - 1, m - 1)
                val1, val2 = nums1[index1], nums2[index2]
                if val1 < val2:
                    ind1 = index1
                    k -= index1 - ind1 + 1
                    ind1 = index1 + 1
                else:
                    k -= index2 - ind2 + 1
                    ind2 = index2 + 1
            return min(nums1[ind1], nums2[ind2])

        s = n + m
        if s % 2:
            return get_kth_num(s // 2 + 1)
        return (get_kth_num(s // 2 + 1) + get_kth_num(s // 2)) / 2



class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().countPairs(nums=[1, 4, 2, 7], low=2, high=6) == 6
        return


if __name__ == '__main__':
    unittest.main()
