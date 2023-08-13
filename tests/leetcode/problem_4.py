import bisect
import random
import re
import sys
import unittest
from typing import List, Callable
from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations, accumulate
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from heapq import heappush, heappop, heappushpop
from operator import xor, mul, add
from functools import lru_cache
from math import inf
import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy
from sortedcontainers import SortedList

sys.set_int_max_str_digits(0)



class NumberTheoryPrimeFactor:
    def __init__(self, ceil):
        self.ceil = ceil
        self.prime_factor = [[] for _ in range(self.ceil + 1)]
        self.min_prime = [0] * (self.ceil + 1)
        self.get_min_prime_and_prime_factor()
        return

    def get_min_prime_and_prime_factor(self):
        # 模板：计算 1 到 self.ceil 所有数字的最小质数因子
        for i in range(2, self.ceil + 1):
            if not self.min_prime[i]:
                self.min_prime[i] = i
                for j in range(i * i, self.ceil + 1, i):
                    if not self.min_prime[j]:
                        self.min_prime[j] = i

        # 模板：计算 1 到 self.ceil 所有数字的质数分解（可选）
        for num in range(2, self.ceil + 1):
            i = num
            while num > 1:
                p = self.min_prime[num]
                cnt = 0
                while num % p == 0:
                    num //= p
                    cnt += 1
                self.prime_factor[i].append([p, cnt])
        return


nt = NumberTheoryPrimeFactor(10**5+10)

mod = 10**9 + 7

class Solution:
    def maximumScore(self, nums: List[int], k: int) -> int:
        n = len(nums)
        factor = [len(nt.prime_factor[x]) for x in nums]
        ans = 1

        post = [n - 1] * n
        pre = [0] * n
        stack = []
        for i in range(n):
            while stack and factor[stack[-1]] < factor[i]:
                post[stack.pop()] = i - 1
            if stack:
                pre[i] = stack[-1] + 1
            stack.append(i)

        ind = list(range(n))
        ind.sort(key=lambda it: -nums[it])
        for i in ind:
            if not k:
                break
            x = (i - pre[i] + 1) * (post[i] - i + 1)
            y = x if x < k else k
            k -= y
            ans *= pow(nums[i], y, mod)
            ans %= mod
        return ans






