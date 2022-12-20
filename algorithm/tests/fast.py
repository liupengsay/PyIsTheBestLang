
import sys
import heapq
sys.setrecursionlimit(10000000)


def read():
    return sys.stdin.readline()


def ac(x):
    return sys.stdout.write(str(x) + '\n')

# 快读快写套餐
map(int, read().split())
int(read())
map(int, read().split())
ac("END")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
read = lambda: sys.stdin.readline().rstrip("\r\n")

def I():
    return read()

def II():
    return int(read())

def MI():
    return map(int, read().split())

def LI():
    return list(read().split())

def LII():
    return list(map(int, read().split()))

def GMI():
    return map(lambda x: int(x) - 1, read().split())

def LGMI():
    return list(map(lambda x: int(x) - 1, read().split()))
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


# int(read().strip())
# [int(w) for w in read().strip().split() if w]
# [float(w) for w in read().strip().split() if w]
# sys.setrecursionlimit(10000000)
#n, c = [int(w) for w in read().strip().split() if w]
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
read = lambda: sys.stdin.readline()
print = lambda x: sys.stdout.write(str(x)+'\n')
sys.setrecursionlimit(10000000)

import sys
import heapq

read = lambda: sys.stdin.readline()
print = lambda x: sys.stdout.write(x)

import random
import sys
import os
import math
from collections import Counter, defaultdict, deque
from functools import lru_cache, reduce
from itertools import accumulate, combinations, permutations
from heapq import nsmallest, nlargest, heapify, heappop, heappush
from io import BytesIO, IOBase
from copy import deepcopy
import threading
import bisect
BUFSIZE = 4096


class FastIO(IOBase):
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None

    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()

    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)

class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")



class SparseTable:
    def __init__(self, data, merge_method):
        self.note = [0] * (len(data) + 1)
        self.merge_method = merge_method
        l, r, v = 1, 2, 0
        while True:
            for i in range(l, r):
                if i >= len(self.note):
                    break
                self.note[i] = v
            else:
                l *= 2
                r *= 2
                v += 1
                continue
            break
        self.ST = [[0] * len(data) for _ in range(self.note[-1]+1)]
        self.ST[0] = data
        for i in range(1, len(self.ST)):
            for j in range(len(data) - (1 << i) + 1):
                self.ST[i][j] = merge_method(self.ST[i-1][j], self.ST[i-1][j + (1 << (i-1))])

    def query(self, l, r):
        pos = self.note[r-l+1]
        return self.merge_method(self.ST[pos][l], self.ST[pos][r - (1 << pos) + 1])

n, m = MI()
height = []
for _ in range(n):
    height.append(II())
minST = SparseTable(height, min)
maxST = SparseTable(height, max)
for _ in range(m):
    l, r = GMI()
    print(maxST.query(l, r) - minST.query(l, r))