
import bisect

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict

import heapq


class Solution:
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        meetings.sort(key=lambda x: x[0])

        stack = []
        heapq.heapify(stack)

        cnt = defaultdict(int)
        null = SortedList(list(range(n)))
        for start, end in meetings:
            while stack and stack[0][0] < start:
                tm, i = heapq.heappop(stack)
                null.add(i)
            if null:
                i = null.pop(0)
                heapq.heappush(stack, [start + end - start - 1, i])
                cnt[i] += 1
            else:
                tm, i = heapq.heappop(stack)
                heapq.heappush(stack, [tm + 1 + end - start - 1, i])
                cnt[i] += 1

        val = max(cnt.values())
        for i in range(n):
            if cnt[i] == val:
                return i


def test_solution():
    assert Solution().mostBooked(2, [[0, 10], [1, 2], [12, 14], [13, 15]]) == 0
    assert Solution().mostBooked(
        4, [[18, 19], [3, 12], [17, 19], [2, 13], [7, 10]]) == 0
    return


if __name__ == '__main__':
    test_solution()
