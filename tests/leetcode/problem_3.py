import bisect
import random
import re
import unittest
from bisect import bisect_left

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations, accumulate
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from heapq import heappush, heappop, heappushpop
from operator import xor, mul, add
from functools import lru_cache
from math import inf
import random
from itertools import permutations, combinations
import numpy as np
from typing import List, Callable
from decimal import Decimal

import heapq
import copy
from sortedcontainers import SortedList

from sortedcontainers import SortedList


class ExamRoom:

    def __init__(self, n: int):
        self.n = n
        self.lst = SortedList(key=lambda it: (-self.dis(it), it[0]))
        self.left = dict()
        self.right = dict()
        self.add((-1, self.n))

    def add(self, x):
        self.lst.add(x)
        self.left[x[1]] = x[0]
        self.right[x[0]] = x[1]
        return

    def delete(self, x):
        self.lst.discard(x)
        self.left.pop(x[1])
        self.right.pop(x[0])
        return

    def dis(self, x):
        if x[0] == -1 or x[1] == self.n:
            return x[1] - x[0] - 1
        return (x[1] - x[0]) // 2

    def seat(self):
        x = self.lst[0]
        if x[0] == -1:
            p = 0
        elif x[1] == self.n - 1:
            p = self.n - 1
        else:
            p = (x[1] + x[0]) // 2
        self.delete(x)
        self.add((x[0], p))
        self.add((p, x[1]))
        return p

    def leave(self, p: int) -> None:
        left, right = self.left[p], self.right[p]
        self.delete((left, p))
        self.delete((p, right))
        self.add((left, right))
        return

    # Your ExamRoom object will be instantiated and called as such:
# obj = ExamRoom(n)
# param_1 = obj.seat()
# obj.leave(p)





assert Solution().countSubstrings(s = "aba", t = "baba") == 6
