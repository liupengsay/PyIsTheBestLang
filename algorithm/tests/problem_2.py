
import bisect
import re
import unittest
from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet
from itertools import combinations
from sortedcontainers import SortedDict

from sortedcontainers import SortedList, SortedDict, SortedSet
from itertools import combinations, permutations
from sortedcontainers import SortedDict
from decimal import Decimal

from collections import deque

from sortedcontainers import SortedList


def get_prime_factor(num):
    # 质因数分解
    res = []
    for i in range(2, int(math.sqrt(num)) + 1):
        cnt = 0
        while num % i == 0:
            num //= i
            cnt += 1
        if cnt:
            res.append([i, cnt])
        if i > num:
            break
    if num != 1 or not res:
        res.append([num, 1])
    return res


class Solution:
    def distinctPrimeFactors(self, nums: List[int]) -> int:
        ans = set()
        for num in nums:
            for x in get_prime_factor(num):
                if x > 1:
                    ans.add(x)
        return len(ans)




class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().smallestValue(4) == 4
        return


if __name__ == '__main__':
    unittest.main()
