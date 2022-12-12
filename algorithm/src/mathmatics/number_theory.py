

"""
自定义有序列表

线性基
"""

import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
from functools import reduce
from operator import xor
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy


class NumberTheory:
    def __init__(self):
        return

    @staticmethod
    def euler_phi(n):
        # 注意1和1互质，而大于1的质数与1不互质
        ans = n
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                ans = ans // i * (i - 1)
                while n % i == 0:
                    n = n // i
        if n > 1:
            ans = ans // n * (n - 1)
        return int(ans)

    @staticmethod
    def ouLaShai(upperBound):
        filter = [False for _ in range(upperBound + 1)]
        primeNumbers = []
        for num in range(2, upperBound + 1):
            if not filter[num]:
                primeNumbers.append(num)
            for prime in primeNumbers:
                if num * prime > upperBound:
                    break
                filter[num * prime] = True
                if num % prime == 0:  # 这句是最有意思的地方  下面解释
                    break
        return primeNumbers

    @staticmethod
    def sieve_of_eratosthenes(n):  # 埃拉托色尼筛选法，返回少于n的素数
        primes = [True] * (n + 1)  # 范围0到n的列表
        p = 2  # 这是最小的素数
        while p * p <= n:  # 一直筛到sqrt(n)就行了
            if primes[p]:  # 如果没被筛，一定是素数
                for i in range(p * 2, n + 1, p):  # 筛掉它的倍数即可
                    primes[i] = False
            p += 1
        primes = [
            element for element in range(
                2, n + 1) if primes[element]]  # 得到所有少于n的素数
        return primes


class TestGeneral(unittest.TestCase):
    def test_euler_phi(self):
        nt = NumberTheory()
        assert nt.euler_phi(10**11 + 131) == 66666666752
        return

    def test_euler_shai(self):
        nt = NumberTheory()
        assert len(nt.ouLaShai(10**6))==78498
        return

    def test_eratosthenes_shai(self):
        nt = NumberTheory()
        assert len(nt.sieve_of_eratosthenes(10**6))==78498
        return

if __name__ == '__main__':
    unittest.main()
