
import bisect
import random
import re
import unittest
from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations, permutations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import heapq

import random
from sortedcontainers import SortedList
from operator import mul
from functools import reduce

from sortedcontainers import SortedList
from operator import mul
from functools import reduce

from functools import cmp_to_key

from sortedcontainers import SortedList
from operator import mul
from functools import reduce

from functools import cmp_to_key

from sortedcontainers import SortedList
from operator import mul
from functools import reduce

from functools import cmp_to_key


def euler_flag_prime(n):
    # 欧拉线性筛素数，返回小于等于 n 的所有素数
    flag = [False for _ in range(n + 1)]
    prime_numbers = []
    for num in range(2, n + 1):
        if not flag[num]:
            prime_numbers.append(num)
        for prime in prime_numbers:
            if num * prime > n:
                break
            flag[num * prime] = True
            if num % prime == 0:
                break
    return prime_numbers


primes = euler_flag_prime(10**6)



class Solution:
    def closestPrimes(self, left: int, right: int) -> List[int]:
        res = [x for x in primes if left <= x <= right]
        ans = []
        m = len(res)
        for i in range(m - 1):
            x, y = res[i], res[i + 1]
            if not ans or y - x < ans[1] - ans[0]:
                ans = [x, y]
        return ans if ans else [-1, -1]


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().countPartitions()

        return


if __name__ == '__main__':
    unittest.main()
