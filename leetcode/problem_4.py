
import bisect
import random

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import heapq

from sortedcontainers import SortedList

from sortedcontainers import SortedList

class Solution:
    def maximumMinimumPath(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        visit = [[1]*n for _ in range(m)]
        stack = [[-grid[0][0], 0, 0]]
        while stack:
            dis, i, j = heapq.heappop(stack)
            dis = -dis
            if i == m-1 and j == n-1:
                return dis
            if visit[i][j] <= dis:
                continue
            visit[i][j] = dis
            for x, y in [[i-1, j], [i+1, j], [i, j-1], [i, j+1]]:
                if 0<=x<m and 0<=y<n:
                    heapq.heappush(stack, [-min(dis, grid[x][y]), x, y])



def test_solution():
    assert Solution().longestRepeating(
        "geuqjmt", "bgemoegklm", [
            3, 4, 2, 6, 5, 6, 5, 4, 3, 2]) == [
        3, 3, 4]

    return


if __name__ == '__main__':
    test_solution()
