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
import sys
import heapq

def input(): return sys.stdin.readline()
def print(x): return sys.stdout.write(str(x) + '\n')


sys.setrecursionlimit(10000000)


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
from collections import defaultdict, Counter, deque
from functools import lru_cache
input = lambda: sys.stdin.readline()
print = lambda x: sys.stdout.write(str(x)+'\n')
sys.setrecursionlimit(10000000)

import math

n, m = map(int, input().split())
lst1 = list(map(int, input().split()))
f = [[0] * 18 for i in range(1 + n)]

for i in range(1, n + 1):
    f[i][0] = lst1[i - 1]
for j in range(1, int(math.log2(n)) + 1):  # 一定不能用ceil，必须用floor 或者int    +1
    for i in range(1, n - (1<<j) + 2):
        a = f[i][j - 1]
        b = f[i +  (1<<(j - 1))][j - 1]
        f[i][j] = a if a > b else b

q = []
res = []
for i in range(m):
    l, r = map(int, input().split())
    q.append([l ,r])
    k = int(math.log2(r - l + 1))
    a = f[l][k]
    b = f[r - (1<<k) + 1][k]
    res.append(a if a > b else b)
print(res)

