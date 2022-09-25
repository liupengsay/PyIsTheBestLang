import bisect

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import random


from collections import defaultdict


class Solution:
    def transportationHub(self, path: List[List[int]]) -> int:
        nodes = set()
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        for a, b in path:
            in_degree[b] += 1
            out_degree[a] += 1
            nodes.add(a)
            nodes.add(b)
        for i in nodes:
            if in_degree[i] == len(nodes) - 1 and out_degree[i] == 0:
                return i
        return -1


def test_solution():
    assert Solution().minDistance([2, 3, 5, 12, 18], 2) == 9
    return


if __name__ == '__main__':
    test_solution()
