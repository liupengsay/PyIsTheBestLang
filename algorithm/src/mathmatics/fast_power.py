
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

from algorithm.src.fast_io import FastIO

"""
算法：快速幂、矩阵快速幂
功能：高效计算整数的幂次方取模
题目：

===================================力扣===================================
450. 应用操作后不同二进制字符串的数量（https://leetcode.cn/problems/number-of-distinct-binary-strings-after-applying-operations/）脑筋急转弯快速幂计算

===================================洛谷===================================
P1630 求和（https://www.luogu.com.cn/problem/P1630）快速幂计算，利用同模进行计数加和
P1939 【模板】矩阵加速（数列）（https://www.luogu.com.cn/problem/P1939）矩阵快速幂递推求解
P1962 斐波那契数列（https://www.luogu.com.cn/problem/P1962）矩阵快速幂递推求解
P3390 【模板】矩阵快速幂（https://www.luogu.com.cn/problem/P3390）矩阵快速幂计算
P3811 【模板】乘法逆元（https://www.luogu.com.cn/problem/P3811）乘法逆元模板题
P5775 [AHOI2006]斐波卡契的兔子（https://www.luogu.com.cn/problem/P5775）从背包模拟、前缀和优化、到数列变换使用矩阵快速幂再到纯模拟
P5550 Chino的数列（https://www.luogu.com.cn/problem/P5550）循环节计算也可以使用矩阵快速幂递推
P6045 后缀树（https://www.luogu.com.cn/problem/P6045）脑筋急转弯进行组合计数与快速幂枚举计算
P6075 [JSOI2015]子集选取（https://www.luogu.com.cn/problem/P6075）组合计数后进行快速幂计算
P6392 中意（https://www.luogu.com.cn/problem/P6392）公式拆解变换后进行快速幂计算

参考：OI WiKi（xx）

"""


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
    def x_pow(x: float, n: int) -> float:
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
    def matrix_mul(y, x, mod=10 ** 9 + 7):
        # 矩阵乘法函数，返回两个矩阵相乘的值，建议背诵
        return [[sum(a * b % mod for a, b in zip(col, row)) % mod for col in zip(*x)] for row in y]

    def matrix_pow(self, mat_a, n, mod=10 ** 9 + 7):

        # 矩阵快速幂算法（递归）求解矩阵的 n 次方结果
        size_ = len(mat_a)
        if n == 0:  # 返回单位矩阵
            res = [[0 for _ in range(size_)] for _ in range(size_)]
            for i in range(size_):
                res[i][i] = 1
            return res
        elif n == 1:  # 返回自己
            return mat_a

        y = self.matrix_pow(mat_a, n // 2)
        if n & 1:  # 要乘
            return self.matrix_mul(self.matrix_mul(y, y, mod), mat_a, mod)
        return self.matrix_mul(y, y, mod)  # 不乘

    @staticmethod
    def matrix_mul2(a, b, mod=10**9 + 7):
        n = len(a)
        res = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    res[i][j] += (a[i][k] % mod) * (b[k][j] % mod)
                    res[i][j] %= mod
        return res

    def matrix_pow2(self, base, p, mod=10**9 + 7):
        n = len(base)
        ans = [[0] * n for _ in range(n)]
        for i in range(n):
            ans[i][i] = 1
        while p:
            if p & 1:
                ans = self.matrix_mul2(ans, base, mod)
            base = self.matrix_mul2(base, base, mod)
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


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1630(ac=FastIO()):
        # 模板：利用取模分组计数与快速幂计算 1**b+2**b+..+a**b % mod 的值
        mod = 10**4
        for _ in range(ac.read_int()):
            a, b = ac.read_ints()
            rest = [0] + [pow(i, b, mod) for i in range(1, mod)]
            ans = sum(rest) * (a // mod) + sum(rest[:a % mod + 1])
            ac.st(ans % mod)
        return

    @staticmethod
    def lg_p1939(ac=FastIO()):
        # 模板：利用转移矩阵乘法公式和快速幂计算值
        mat = [[1, 0, 1], [1, 0, 0], [0, 1, 0]]
        lst = [1, 1, 1]
        mod = 10**9 + 7
        mfp = MatrixFastPower()
        for _ in range(ac.read_int()):
            n = ac.read_int()
            if n > 3:
                nex = mfp.matrix_pow(mat, n - 3)
                ans = sum(nex[0]) % mod
                ac.st(ans)
            else:
                ac.st(lst[n - 1])
        return


class TestGeneral(unittest.TestCase):

    def test_fast_power(self):
        fp = FastPower()
        a, b, mod = random.randint(
            1, 123), random.randint(
            1, 1234), random.randint(
            1, 12345)
        assert fp.fast_power_api(a, b, mod) == fp.fast_power(a, b, mod)

        x, n = random.uniform(0, 1), random.randint(1, 1234)
        assert abs(fp.x_pow(x, n) - pow(x, n)) < 1e-5

        mfp = MatrixFastPower()
        mat = [[1, 0, 1], [1, 0, 0], [0, 1, 0]]
        mod = 10 ** 9 + 7
        for _ in range(10):
            n = random.randint(1, 100)
            cur = copy.deepcopy(mat)
            for _ in range(1, n):
                cur = mfp.matrix_mul(cur, mat, mod)
            assert cur == mfp.matrix_pow(
                mat, n, mod) == mfp.matrix_pow2(
                mat, n, mod)

        base = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert mfp.matrix_pow(
            mat, 0, mod) == mfp.matrix_pow2(
            mat, 0, mod) == base
        return


if __name__ == '__main__':
    unittest.main()
