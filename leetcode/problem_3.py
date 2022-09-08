
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


class Solution:
    def longestNiceSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        cur = 0
        ans = 1
        j = 0
        for i in range(n):
            while j < n and cur ^ nums[j] == cur + nums[j]:
                cur ^= nums[j]
                j += 1
            if j - i > ans:
                ans = j - i
            cur ^= nums[i]
        return ans


def test_solution():
    assert Solution().longestNiceSubarray([1, 3, 8, 48, 10]) == 3
    return


if __name__ == '__main__':
    test_solution()
