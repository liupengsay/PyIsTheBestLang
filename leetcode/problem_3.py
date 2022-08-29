
import bisect

from typing import List

import math
from collections import defaultdict, Counter
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet


MOD = 10**9 + 7


class Solution:
    def maxProductPath(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[[0, 0] for _ in range(n)] for _ in range(m)]
        dp[0][0] = [grid[0][0], grid[0][0]]
        for j in range(1, n):
            dp[0][j] = [dp[0][j - 1][0] * grid[0]
                        [j], dp[0][j - 1][1] * grid[0][j]]
        for i in range(1, m):
            dp[i][0] = [dp[i - 1][0][0] * grid[i]
                        [0], dp[i - 1][0][1] * grid[i][0]]

        for i in range(1, m):
            for j in range(1, n):
                low = min(grid[i][j] * dp[i - 1][j][0],
                          grid[i][j] * dp[i - 1][j][1],
                          grid[i][j] * dp[i][j - 1][0],
                          grid[i][j] * dp[i][j - 1][1])
                high = max(grid[i][j] * dp[i - 1][j][0],
                           grid[i][j] * dp[i - 1][j][1],
                           grid[i][j] * dp[i][j - 1][0],
                           grid[i][j] * dp[i][j - 1][1])
                dp[i][j] = [low, high]
        ans = dp[-1][-1][1]
        if ans < 0:
            return -1
        return ans % MOD


assert Solution().maxProductPath(
    [[-1, -2, -3], [-2, -3, -3], [-3, -3, -2]]) == -1
assert Solution().maxProductPath([[1, -2, 1], [1, -2, 1], [3, -4, 1]]) == 8
assert Solution().maxProductPath([[1, 3], [0, -4]]) == 0
assert Solution().maxProductPath(
    [[1, 4, 4, 0], [-2, 0, 0, 1], [1, -1, 1, 1]]) == 2
