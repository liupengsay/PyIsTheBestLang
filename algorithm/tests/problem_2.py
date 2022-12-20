
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
    res = []
    for i in range(2, num):
        cnt = 0
        while num % i == 0:
            num //= i
            cnt += 1
            #print(i, num)
        if cnt:
            res.append([i, cnt])
        if i > num:
            break
    if not res:
        res = [[num, 1]]
    return res


class Solution:
    def smallestValue(self, n: int) -> int:
        while True:
            res = get_prime_factor(n)
            if len(res) == 1 and res[0][1] == 1:
                break
            n = 0
            for num, va in res:
                n += num*va
        return n



class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().smallestValue(4) == 4
        return


if __name__ == '__main__':
    unittest.main()
