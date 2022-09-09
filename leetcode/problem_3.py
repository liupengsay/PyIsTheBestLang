
import bisect

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import heapq
from itertools import combinations

import heapq


class Solution:
    def checkPartitioning(self, s: str) -> bool:
        n = len(s)
        dp = [[0]*n for _ in range(n)]
        for i in range(n-1, -1, -1):
            dp[i][i] = 1
            if i+1<n:
                dp[i][i+1] = int(s[i]==s[i+1])
            for j in range(i+2, n):
                dp[i][j] = dp[i+1][j-1] & int(s[i]==s[j])
        for i in range(1, n-1):
            for j in range(i+1, n):
                if dp[0][i-1] and dp[i][j-1] and dp[j][n-1]:
                    return True
        return False



def test_solution():
    assert Solution().checkPartitioning("abcbdd") == True
    return


if __name__ == '__main__':
    test_solution()
