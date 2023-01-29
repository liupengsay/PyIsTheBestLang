
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
import datetime
import unittest
import time


inf = float("inf")
sys.setrecursionlimit(10000000)

getcontext().prec = MAX_PREC


def fun(a, b, c,d):
    return (a|b)^(c&d)


cnt = Counter()
for a in range(3):
    for b in range(3):
        for c in range(3):
            for d in range(3):
               # print(a, b, c, d)
                s = fun(a, b, c, d)
                cnt[s] += 1
print(cnt)
