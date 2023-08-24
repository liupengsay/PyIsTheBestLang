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

# sys.set_int_max_str_digits(0)  # 大数的范围坑



class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:

        color = [0]*n
        ans = []
        pre = 0
        for i, c in queries:
            if color[i]:
                if i and color[i] == color[i-1]:
                    pre -= 1
                if i+1<n and color[i] == color[i+1]:
                    pre -= 1
            color[i] = c
            if i and color[i] == color[i - 1]:
                pre += 1
            if i + 1 < n and color[i] == color[i + 1]:
                pre += 1
            ans.append(pre)
        return ans




assert Solution()