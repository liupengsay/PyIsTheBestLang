
"""

"""
"""
算法：数论、欧拉筛、线性筛、素数、欧拉函数、因子分解、素因子分解
功能：
题目：


参考：OI WiKi（xx）
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

    def gcd(self, x, y):
        # # 最大公约数
        if y == 0:
            return x
        else:
            return self.gcd(y, x % y)

    def lcm(self, x, y):
        # 最小公倍数
        return x * y // self.gcd(x, y)

    @staticmethod
    def factorial_zero_count(num):
        # 阶乘后的后缀零个数
        cnt = 0
        while num > 0:
            cnt += num // 5
            num //= 5
        return cnt

    @staticmethod
    def get_k_bin_of_n(n, k):
        # 整数n的k进制计算
        if n == 0:
            return [0]
        if k == 0:
            return []
        # 支持正负数
        pos = 1 if k > 0 else -1
        k = abs(k)
        lst = []
        while n:
            lst.append(n % k)
            n //= k
            n *= pos
        lst.reverse()
        return lst

    @staticmethod
    def is_prime(num):
        # 判断数是否为质数
        if num <= 1:
            return False
        for i in range(2, min(int(math.sqrt(num)) + 2, num)):
            if num % i == 0:
                return False
        return True

    @staticmethod
    def rational_number_to_fraction(st):
        """
        # 有理数循环小数化为分数
        1.2727... = (27 / 99) + 1
        1.571428571428... = (571428 / 999999) + 1
        有n位循环 = (循环部分 / n位9) + 整数部分
        最后约简
        """
        n = len(st)
        a = int(st)
        b = int("9" * n)
        c = math.gcd(a, b)
        a //= c
        b //= c
        return [a, b]

    @staticmethod
    def euler_phi(n):
        # 欧拉函数返回小于等于n的与n互质的个数
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
        # 欧拉线性筛素数
        # 说明：返回小于upperBound的所有素数
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
    def sieve_of_eratosthenes(n):  # 埃拉托色尼筛选法，返回小于等于n的素数
        primes = [True] * (n + 1)  # 范围0到n的列表
        p = 2  # 这是最小的素数
        while p * p <= n:  # 一直筛到sqrt(n)就行了
            if primes[p]:  # 如果没被筛，一定是素数
                for i in range(p * 2, n + 1, p):  # 筛掉它的倍数即可
                    primes[i] = False
            p += 1
        primes = [element for element in range(2, n + 1) if primes[element]]  # 得到所有少于n的素数
        return primes

    @staticmethod
    def get_all_factor(num):
        # 获取整数所有的因子包括1和它自己
        factor = set()
        for i in range(1, int(math.sqrt(num)) + 1):
            if num % i == 0:
                factor.add(i)
                factor.add(num // i)
        return sorted(list(factor))

    @staticmethod
    def get_prime_factor(num):
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
        if not res:
            res = [[num, 1]]
        return res

class TestGeneral(unittest.TestCase):
    def test_euler_phi(self):
        nt = NumberTheory()
        assert nt.euler_phi(10**11 + 131) == 66666666752
        return

    def test_euler_shai(self):
        nt = NumberTheory()
        correctResult_30 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        ouLaShaiResult_30 = nt.ouLaShai(30)
        assert correctResult_30 == ouLaShaiResult_30
        assert len(nt.ouLaShai(10**6)) == 78498
        return

    def test_eratosthenes_shai(self):
        nt = NumberTheory()
        assert len(nt.sieve_of_eratosthenes(10**6)) == 78498
        return

    def test_factorial_zero_count(self):
        nt = NumberTheory()
        num = random.randint(1, 100)
        s = str(math.factorial(num))
        cnt = 0
        for w in s[::-1]:
            if w == "0":
                cnt += 1
            else:
                break
        assert nt.factorial_zero_count(num) == cnt
        return

    def test_get_k_bin_of_n(self):
        nt = NumberTheory()
        num = random.randint(1, 100)
        assert nt.get_k_bin_of_n(num, 2) == [int(w) for w in bin(num)[2:]]

        assert nt.get_k_bin_of_n(4, -2) == [1, 0, 0]
        return

    def test_rational_number_to_fraction(self):
        nt = NumberTheory()
        assert nt.rational_number_to_fraction("33") == [1, 3]
        return

    def test_is_prime(self):
        nt = NumberTheory()
        assert not nt.is_prime(1)
        assert nt.is_prime(5)
        assert not nt.is_prime(51)
        return

    def test_gcd_lcm(self):
        nt = NumberTheory()
        a, b = random.randint(1, 1000), random.randint(1, 1000)
        assert nt.gcd(a, b) == math.gcd(a, b)
        assert nt.lcm(a, b) == math.lcm(a, b)
        return

    def test_get_prime_factor(self):
        nt = NumberTheory()

        num = 2
        assert nt.get_prime_factor(num) == [[2, 1]]

        num = 1
        assert nt.get_prime_factor(num) == [[1, 1]]

        num = 2*(3**2)*7*(11**3)
        assert nt.get_prime_factor(num) == [[2, 1], [3, 2], [7, 1], [11, 3]]
        return

if __name__ == '__main__':
    unittest.main()
