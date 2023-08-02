import bisect
import random
import re
import unittest

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




class PreFixSumMatrix:
    def __init__(self, mat):
        self.mat = mat
        # 二维前缀和
        m, n = len(mat), len(mat[0])
        self.pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                self.pre[i + 1][j + 1] = self.pre[i][j + 1] + \
                    self.pre[i + 1][j] - self.pre[i][j] + mat[i][j]

    def query(self, xa, ya, xb, yb):
        # 二维子矩阵和查询，索引从 0 开始，左上角 [xa, ya] 右下角 [xb, yb]
        return self.pre[xb + 1][yb + 1] - self.pre[xb +
                                                   1][ya] - self.pre[xa][yb + 1] + self.pre[xa][ya]


class Solution:
    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        pre = PreFixSumMatrix(mat)
        m, n = len(mat), len(mat[0])
        ans= [[0]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                x1 = max(0, i-k)
                x2 = min(m-1, i+k)
                y1 = max(0, j-k)
                y2 = min(n-1, j+k)
                ans[i][j] = pre.query(x1, y1,x2,y2)
        return ans


assert Solution().minFlips("010") == 0



