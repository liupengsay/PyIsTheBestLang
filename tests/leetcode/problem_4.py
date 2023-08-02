import bisect
import random
import re
import unittest

from typing import List, Callable, Dict
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





class StringHash:
    # 注意哈希碰撞，需要取两个质数与模进行区分
    def __init__(self, n, s):
        self.n = n
        self.p = [random.randint(26, 100), random.randint(26, 100)]
        self.mod = [random.randint(10 ** 9 + 7, 2 ** 31 - 1), random.randint(10 ** 9 + 7, 2 ** 31 - 1)]
        self.pre = [[0], [0]]
        self.pp = [[1], [1]]
        for w in s:
            for i in range(2):
                self.pre[i].append((self.pre[i][-1] * self.p[i] + ord(w) - ord("a")) % self.mod[i])
                self.pp[i].append((self.pp[i][-1] * self.p[i]) % self.mod[i])
        return

    def query(self, x, y):
        # 模板：字符串区间的哈希值，索引从 0 开始
        ans = [0, 0]
        for i in range(2):
            if x <= y:
                ans[i] = (self.pre[i][y + 1] - self.pre[i][x] * pow(self.p[i], y - x + 1, self.mod[i])) % self.mod[i]
        return ans


class Solution:
    def distinctEchoSubstrings(self, text: str) -> int:
        n = len(text)
        sh = StringHash(n, text)

        ans = set()
        for x in range(1, n//2+1):
            for i in range(n-2*x+1):
                ans1 = sh.query(i, i+x-1)
                ans2 = sh.query(i+x, i+2*x-1)
                if ans1 == ans2:
                    ans.add(tuple(ans1))
        return len(ans)




assert Solution().maximumScore(scores = [5,2,9,8,4], edges = [[0,1],[1,2],[2,3],[0,2],[1,3],[2,4]]) == 24
