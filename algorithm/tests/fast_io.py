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

import sys
import heapq

input = lambda: sys.stdin.readline()
print = lambda x: sys.stdout.write(x)


def main():
    n, m, b = map(int, input().split())
    cost = [0]*n
    for i in range(n):
        cost[i] = int(input())

    dct = [dict() for _ in range(n)]
    for _ in range(m):
        a, b, c = map(int, input().split())
        a -= 1
        b -= 1
        if b not in dct[a] or dct[a][b] > c:
            dct[a][b] = c
        if a not in dct[b] or dct[b][a] > c:
            dct[b][a] = c

    def check():
        visit = [float("inf")]*n
        blood = [float("-inf")]*n
        stack = [[cost[0], 0, b]]
        while stack:
            dis, i, bd = heapq.heappop(stack)
            if i == n-1:
                return str(dis)
            if dis >= visit[i] and bd <= blood[i]:
                continue
            if dis<visit[i]:
                visit[i] = dis
                blood[i] = bd
            for j in dct[i]:
                heapq.heappush(stack, [max(dis, cost[j]), j, bd-dct[i][j]])
        return "AFK"

    print(check())
    return

main()
