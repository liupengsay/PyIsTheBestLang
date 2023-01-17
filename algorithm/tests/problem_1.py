

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


class MedianFinder:

    def __init__(self):
        self.lst = SortedList()

    def addNum(self, num: int) -> None:
        self.lst.add(num)

    def findMedian(self) -> float:
        n = len(self.lst)
        if n%2:
            return self.lst[n//2]
        return (self.lst[n//2]+self.lst[n//2-1])/2



# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()

class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().sortArray([5,1,1,2,0,0]) == [0,0,1,1,2,5]
        return


if __name__ == '__main__':
    unittest.main()
