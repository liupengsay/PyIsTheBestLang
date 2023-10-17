import copy
import math
import random
import unittest

from utils.fast_io import FastIO
from src.string.kmp import KMP



class FastPower:
    def __init__(self):
        return

    @staticmethod
    def fast_power_api(a, b, mod):
        return pow(a, b, mod)

    @staticmethod
    def fast_power(a, b, mod):
        a = a % mod
        res = 1
        while b > 0:
            if b & 1:
                res = res * a % mod
            a = a * a % mod
            b >>= 1
        return res

    @staticmethod
    def float_fast_pow(x: float, n: int) -> float:
        # 浮点数快速幂
        def quick_mul(n):
            if n == 0:
                return 1.0
            y = quick_mul(n // 2)
            return y * y if n % 2 == 0 else y * y * x

        return quick_mul(n) if n >= 0 else 1.0 / quick_mul(-n)


class MatrixFastPower:
    def __init__(self):
        return

    @staticmethod
    def matrix_mul(a, b, mod=10**9 + 7):
        n = len(a)
        res = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    res[i][j] += (a[i][k] % mod) * (b[k][j] % mod)
                    res[i][j] %= mod
        return res

    def matrix_pow(self, base, p, mod=10**9 + 7):
        n = len(base)
        ans = [[0] * n for _ in range(n)]
        for i in range(n):
            ans[i][i] = 1
        while p:
            if p & 1:
                ans = self.matrix_mul(ans, base, mod)
            base = self.matrix_mul(base, base, mod)
            p >>= 1
        return ans


class PowerReverse:
    def __init__(self):
        return

    # 扩展欧几里得求乘法逆元
    def ex_gcd(self, a, b):
        if b == 0:
            return 1, 0, a
        else:
            x, y, q = self.ex_gcd(b, a % b)
            x, y = y, (x - (a // b) * y)
            return x, y, q

    def mod_reverse(self, a, p):
        x, y, q = self.ex_gcd(a, p)
        if q != 1:
            raise Exception("No solution.")
        else:
            return (x + p) % p  # 防止负数


