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

import bisect

import random
import math

from sortedcontainers import SortedList


# int(input().strip())
# [int(w) for w in input().strip().split() if w]
# [float(w) for w in input().strip().split() if w]
# sys.setrecursionlimit(10000000)
#n, c = [int(w) for w in input().strip().split() if w]
import numpy as np
import math
import bisect
from functools import lru_cache
from collections import defaultdict
import bisect

import math
import heapq




s1 = input().strip()
s2 = input().strip()

class KMP:
    def __init__(self):
        return

    @staticmethod
    def prefix_function(s):
        # 计算s[:i]与s[:i]的最长公共真前缀与真后缀
        n = len(s)
        pi = [0] * n
        for i in range(1, n):
            j = pi[i - 1]
            while j > 0 and s[i] != s[j]:
                j = pi[j - 1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j
        # pi[0] = 0
        return pi



ans = KMP().prefix_function(s2)
print(" ".join(str(x) for x in ans))
