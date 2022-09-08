import bisect

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import random


class Solution:
    def rearrangeString(self, s: str, k: int) -> str:

        cnt = Counter(s)
        stack = []
        for w, c in cnt.items():
            heapq.heappush(stack, [0, -c, w])

        ans = ""
        ind = 0
        while stack:
            if stack[0][0] > ind:
                return ""
            while stack and stack[0][0] < ind:
                item = heapq.heappop(stack)
                item[0] = ind
                heapq.heappush(item)
            _, c, w = heapq.heappop(stack)
            c += 1
            ans += w
            ind += 1
            if c < 0:
                heapq.heappush(stack, [ind - 1 + k, c, w])
        return ans


def test_solution():
    assert Solution().minDistance([2, 3, 5, 12, 18], 2) == 9
    return


if __name__ == '__main__':
    test_solution()
