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
from heapq import heappush, heappop, heappushpop
from operator import xor, mul, add
from functools import lru_cache
from math import inf
import random
from itertools import permutations, combinations
import numpy as np
from typing import List, Callable
from decimal import Decimal

import heapq
import copy
from sortedcontainers import SortedList







class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        n = len(position)
        ind = list(range(n))
        ind.sort(key=lambda it: -position[it])
        ans = 1
        right = [target-position[ind[0]], speed[ind[0]]]
        for x in ind[1:]:
            c, d = right
            a, b = target-position[x], speed[x]
            if c*b < a*d:
                ans += 1
                right = [a, b]
        return ans





assert Solution()



