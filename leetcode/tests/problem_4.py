
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
import heapq

import random

MOD = 10 ** 9 + 7


class Solution:
    def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])

        @lru_cache(None)
        def dfs(i, j):
            cnt = [0] * k
            num = grid[i][j]
            if i == m - 1 and j == n - 1:
                cnt[num % k] = 1
                return cnt
            for x, y in [[i + 1, j], [i, j + 1]]:
                if 0 <= x < m and 0 <= y < n:
                    nex = dfs(x, y)
                    for i in range(k):
                        cnt[(i + num) % k] += nex[i]
            for i in range(k):
                cnt[i] %= MOD
            return cnt

        return dfs(0, 0)[0]


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().maxHappyGroups(batchSize = 3, groups = [1,2,3,4,5,6]) == 4
        assert Solution().maxHappyGroups(batchSize = 4, groups = [1,3,2,5,2,2,1,6]) == 4
        assert Solution().maxHappyGroups(3, [844438225,657615828,355556135,491931377,644089602,30037905,863899906,246536524,682224520]) == 6
        assert Solution().maxHappyGroups(8, [244197059,419273145,329407130,44079526,351372795,200588773,340091770,851189293,909604028,621703634,959388577,989293607,325139045,263977422,358987768,108391681,584357588,656476891,621680874,867119215,639909909,98831415,263171984,236390093,21876446]) == 13
        return


if __name__ == '__main__':
    unittest.main()
