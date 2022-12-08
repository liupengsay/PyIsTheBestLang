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

#n, c = [int(w) for w in input().strip().split() if w]
import numpy as np
import math
import bisect

import sys
from functools import lru_cache
from collections import defaultdict
sys.setrecursionlimit(10000000)
input = lambda: sys.stdin.readline()
print = lambda x: sys.stdout.write(str(x)+'\n')


while True:
    s = input().strip()
    if not s or s == "EOF":
        break
    m, n = [int(w) for w in s.split() if w]
    lst = []
    for _ in range(m):
        s = input().strip()
        while not s:
            s = input().strip()
        lst.append(s)

    def find():
        for i in range(m):
            for j in range(n):
                if lst[i][j] == "S":
                    return [i, j]

    start = find()
    stack = [start]
    visit = {(start[0], start[1])}
    ans = ""
    while stack and not ans:
        nex = []
        for i, j in stack:
            if ans:
                break
            for x, y in [[i-1,j],[i+1,j],[i,j-1],[i,j+1]]:
                if lst[x%m][y%n] != "#":
                    if lst[x%m][y%n] == "S" and (x, y) not in visit:
                        ans = "Yes"
                        break
                    if (x, y) not in visit:
                        lst[x%m][y%n] = "S"
                        visit.add((x,y))
                        nex.append((x, y))
        stack = nex
    if ans:
        print("Yes")
    else:
        print("No")