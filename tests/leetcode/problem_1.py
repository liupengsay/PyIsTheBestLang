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
from functools import lru_cache
from itertools import combinations, accumulate
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from heapq import heappush, heappop, heappushpop
from operator import xor, mul, add
from functools import lru_cache
from math import inf
import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy
from sortedcontainers import SortedList

sys.set_int_max_str_digits(0)


class Solution:
    def maxSum(self, nums: List[int]) -> int:
        ans = -1
        dct = defaultdict(int)
        for i, num in enumerate(nums):
            x = max(w for w in str(nums[i]))
            if dct[x]:
                if num + dct[x] > ans:
                    ans = num + dct[x]
            if num > dct[x]:
                dct[x] = num
        return ans


assert Solution()
