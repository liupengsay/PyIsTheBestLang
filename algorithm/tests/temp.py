
from __future__ import division
import random
import math
from collections import defaultdict


import copy
import random
import heapq
import math
import sys
import bisect
import re
import time
import datetime
from functools import lru_cache
from collections import deque
from collections import Counter
from collections import defaultdict
from itertools import combinations
from itertools import permutations
from itertools import accumulate
from decimal import Decimal, getcontext, MAX_PREC
from types import GeneratorType
from functools import cmp_to_key
import functools
import datetime
import unittest
import time


inf = float("inf")
sys.setrecursionlimit(10000000)

getcontext().prec = MAX_PREC


def my_compare(x, y):
    if x < y:
        return -1
    elif x > y:
        return 1
    return 0

nums = list(range(14))[::-1]
nums.sort(key = functools.cmp_to_key(my_compare))
print(nums)

import functools

strs = [3, 4, 1, 2]


# 自定义排序规则
def my_compare(x, y):
    if x > y:
        return 1
    elif x < y:
        return -1
    return 0


# 分别使用sorted和list.sort
print(strs)
print(sorted(strs, key=functools.cmp_to_key(my_compare)))

print(strs)
strs.sort(key=functools.cmp_to_key(my_compare))
print(strs)




