import unittest
from typing import List
from collections import Counter, defaultdict
from src.fast_io import FastIO
import math
from functools import lru_cache

from src.mathmatics.number_theory import NumberTheory
from src.mathmatics.prime_factor import NumberTheoryPrimeFactor




class Combinatorics:
    def __init__(self, n, mod):
        # 模板：求全排列组合数，使用时注意 n 的取值范围
        n += 10
        self.perm = [1] * n
        self.rev = [1] * n
        self.mod = mod
        for i in range(1, n):
            # 阶乘数 i! 取模
            self.perm[i] = self.perm[i - 1] * i
            self.perm[i] %= self.mod
        self.rev[-1] = self.mod_reverse(self.perm[-1], self.mod)  # 等价于pow(self.perm[-1], -1, self.mod)
        for i in range(n - 2, 0, -1):
            self.rev[i] = (self.rev[i + 1] * (i + 1) % mod)  # 阶乘 i! 取逆元
        self.fault = [0] * n
        self.fault_perm()
        return

    def ex_gcd(self, a, b):
        # 扩展欧几里得求乘法逆元
        if b == 0:
            return 1, 0, a
        else:
            x, y, q = self.ex_gcd(b, a % b)
            x, y = y, (x - (a // b) * y)
            return x, y, q

    def mod_reverse(self, a, p):
        x, y, q = self.ex_gcd(a, p)
        if q != 1:
            raise Exception("No solution.")   # 逆元要求a与p互质
        else:
            return (x + p) % p  # 防止负数

    def comb(self, a, b):
        if a < b:
            return 0
        # 组合数根据乘法逆元求解
        res = self.perm[a] * self.rev[b] * self.rev[a - b]
        return res % self.mod

    def factorial(self, a):
        # 组合数根据乘法逆元求解
        res = self.perm[a]
        return res % self.mod

    def fault_perm(self):
        # 求错位排列组合数
        self.fault[0] = 1
        self.fault[2] = 1
        for i in range(3, len(self.fault)):
            self.fault[i] = (i - 1) * (self.fault[i - 1] + self.fault[i - 2])
            self.fault[i] %= self.mod
        return

    def inv(self, n):
        # 求 pow(n, -1, mod)
        return self.perm[n - 1] * self.rev[n] % self.mod

    def catalan(self, n):
        # 求卡特兰数
        return (self.comb(2 * n, n) - self.comb(2 * n, n - 1)) % self.mod


class Lucas:
    def __init__(self):
        # 模板：快速求Comb(a,b)%p
        return

    @staticmethod
    def lucas(self, n, m, p):
        # 模板：卢卡斯定理，求 math.comb(n, m) % p，要求p为质数
        if m == 0:
            return 1
        return ((math.comb(n % p, m % p) % p) * self.lucas(n // p, m // p, p)) % p

    @staticmethod
    def comb(n, m, p):
        # 模板：利用乘法逆元求comb(n,m)%p
        ans = 1
        for x in range(n - m + 1, n + 1):
            ans *= x
            ans %= p
        for x in range(1, m + 1):
            ans *= pow(x, -1, p)
            ans %= p
        return ans

    def lucas_iter(self, n, m, p):
        # 模板：卢卡斯定理，求 math.comb(n, m) % p，要求p为质数
        if m == 0:
            return 1
        stack = [[n, m]]
        dct = dict()
        while stack:
            n, m = stack.pop()
            if n >= 0:
                if m == 0:
                    dct[(n, m)] = 1
                    continue
                stack.append([~n, m])
                stack.append([n // p, m // p])
            else:
                n = ~n
                dct[(n, m)] = (self.comb(n % p, m % p, p) % p) * dct[(n // p, m // p)] % p
        return dct[(n, m)]

    @staticmethod
    def extend_lucas(self, n, m, p):
        # 模板：扩展卢卡斯定理，求 math.comb(n, m) % p，不要求p为质数
        return


