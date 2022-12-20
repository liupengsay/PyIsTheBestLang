


import bisect
import itertools
import random
from typing import List
import heapq
import math
import re
import unittest
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import heapq
from itertools import combinations, permutations



class Solution:
    def isPossible(self, n: int, edges: List[List[int]]) -> bool:
        degree = [0] * n
        edge = set()
        for a, b in edges:
            a -= 1
            b -= 1
            if a > b:
                a, b = b, a
            degree[a] += 1
            degree[b] += 1
            edge.add((a, b))

        odd = [i for i in range(n) if degree[i] % 2 == 1]

        if len(odd) > 4:
            return False

        elif len(odd) == 0:
            return True

        elif len(odd) == 2:
            i, j = odd
            if (i, j) not in edge:
                return True
            for x in range(n):
                if i != x and j != x:
                    if tuple(sorted([i, x])) not in edge and tuple(sorted([j, x])) not in edge:
                        return True
            return False

        elif len(odd) == 4:
            for item in permutations(odd, 4):
                x, y, a, b = item
                if x > y:
                    x, y = y, x
                if a > b:
                    a, b = b, a
                if (x, y) in edge or (a, b) in edge:
                    continue
                dct = defaultdict(int)
                for i in item:
                    dct[i] += 1
                if all((degree[i] + dct[i]) % 2 == 0 for i in odd):
                    return True
        return False


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().isPossible(4, [[1,2],[2,3],[2,4],[3,4]]) == False

        return


if __name__ == '__main__':
    unittest.main()
