

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

from functools import lru_cache

import random
from itertools import permutations

class Solution:
    def findEvenNumbers(self, digits: List[int]) -> List[int]:
        ans = set()
        for item in permutations(digits, 3):
            if item[0] != 0 and item[-1]%2 == 0:
                ans.add(item[0]*100+item[1]*10+item[2])
        return sorted(list(ans))


def test_solution():
    assert Solution().kSimilarity(s1="abc", s2="bca") == 2
    return


if __name__ == '__main__':
    test_solution()
