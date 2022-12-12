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
import heapq

import sys
import math
sys.setrecursionlimit(1000000)
input = lambda: sys.stdin.readline()
print = lambda x: sys.stdout.write(str(x)+'\n')


class MultiplicativeInverse:
    def __init__(self):
        return

    @staticmethod
    def get_result(a, p):
        # 注意a和p都为正整数
        return pow(a, -1, p)



import numpy as np
n = int(input().strip())
grid = []
for _ in range(n):
    grid.append([int(w) for w in input().strip().split() if w])


MOD = 10**9 + 7
print(np.linalg.inv(np.array(grid)))
# 矩阵对象可以通过 .I 更方便的求逆
A = np.matrix(np.array(grid))
print([[x%MOD for x in ls] for ls in A.I])

