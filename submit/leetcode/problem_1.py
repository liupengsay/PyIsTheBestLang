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
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from heapq import heappush, heappop, heappushpop, heapify
from operator import xor, mul, add, or_
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


def ac_max(a, b):
    return a if a > b else b


def ac_min(a, b):
    return a if a < b else b



lst = []
for i in range(2, 65):
    ll = 1<<i
    rr = (1<<(i+1))-1
    lst.append([i, ll, rr])

    low = int(math.log(i, ll))
    high = math.ceil(math.log(i, rr))
    for j in range(low, high+1):
        aa = i**j
        bb = i**(j+1) - 1
        xx = ac.max(aa, ll)
        yy = ac.min(bb, rr)
print(lst)


# for x in range(4, 100001):
#     y = len(bin(x)[2:])-1
#     z = 0
#     while y**(z+1) <= x:
#         z += 1
#     print(x, y, z)







