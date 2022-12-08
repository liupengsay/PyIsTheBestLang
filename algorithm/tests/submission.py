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
# [int(w) for w in input().split() if w]
# [float(w) for w in input().split() if w]
# sys.setrecursionlimit(10000000)
#n, c = [int(w) for w in input().strip().split() if w]
import numpy as np
import math
import bisect
from functools import lru_cache
from collections import defaultdict
import bisect

import math
import sys

input = lambda: sys.stdin.readline()
print = lambda x: sys.stdout.write(str(x)+'\n')


low, high = [int(w) for w in input().split() if w]


n = int(math.sqrt(high))+1
primes = [True] * (n + 1)  # 范围0到n的列表
p = 2  # 这是最小的素数
while p * p <= n:  # 一直筛到sqrt(n)就行了
    if primes[p]:  # 如果没被筛，一定是素数
        for i in range(p * 2, n + 1, p):  # 筛掉它的倍数即可
            primes[i] = False
    p += 1
primes = [element for element in range(2, n + 1) if primes[element]]  # 得到所有少于n的素数


euler_phi = [True]*(high-low+1)
for p in primes:
    for a in range(max(low//p, 2), high//p+1):
        if low<=a*p<=high:
            euler_phi[a*p-low] = False
if low == 1:
    euler_phi[1] = False
print(sum(euler_phi))
