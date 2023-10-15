import math
import unittest
from collections import Counter
from collections import defaultdict
from functools import reduce
from itertools import permutations
from math import inf
from typing import List

from src.fast_io import FastIO
from src.mathmatics.number_theory import NumberTheory


class PrimeFactor:
    def __init__(self, ceil):
        self.ceil = ceil + 100
        # 模板：计算 1 到 self.ceil 所有数字的最小质数因子
        self.min_prime = [0] * (self.ceil + 1)
        # 模板：判断 1 到 self.ceil 所有数字是否为质数
        self.is_prime = [0] * (self.ceil + 1)
        # 模板：计算 1 到 self.ceil 所有数字的质数分解
        self.prime_factor = [[] for _ in range(self.ceil + 1)]
        # 模板：计算 1 到 self.ceil 所有数字的所有因子包含 1 和数字其本身
        self.all_factor = [[1] for _ in range(self.ceil + 1)]

        self.build()
        return

    def build(self):

        # 最小质因数与是否为质数O(nlogn)
        for i in range(2, self.ceil + 1):
            if not self.min_prime[i]:
                self.is_prime[i] = 1
                self.min_prime[i] = i
                for j in range(i * i, self.ceil + 1, i):
                    if not self.min_prime[j]:
                        self.min_prime[j] = i

        # 质因数分解O(nlogn)
        for num in range(2, self.ceil + 1):
            i = num
            while num > 1:
                p = self.min_prime[num]
                cnt = 0
                while num % p == 0:
                    num //= p
                    cnt += 1
                self.prime_factor[i].append([p, cnt])

        # 所有因数分解O(nlogn)
        for i in range(2, self.ceil + 1):
            x = 1
            while x * i <= self.ceil:
                self.all_factor[x * i].append(i)
                x += 1
        return

    def comb(self, n, m):
        # 模板：使用质因数分解的方式求解组合数学的值以及质因数分解O((n+m)log(n+m))
        cnt = defaultdict(int)
        for i in range(1, n + 1):  # n!
            for num, y in self.prime_factor[i]:
                cnt[num] += y
        for i in range(1, m + 1):  # m!
            for num, y in self.prime_factor[i]:
                cnt[num] -= y
        for i in range(1, n - m + 1):  # (n-m)!
            for num, y in self.prime_factor[i]:
                cnt[num] -= y
        ans = 1
        for w in cnt:
            ans *= w ** cnt[w]
        return ans



