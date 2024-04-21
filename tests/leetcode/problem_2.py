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
    def example() -> int:
        return 0


class TestGeneral(unittest.TestCase):

    def test_example(self):
        assert Solution().example() == 0
        return


if __name__ == '__main__':
    unittest.main()
