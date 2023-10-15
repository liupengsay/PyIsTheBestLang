import heapq
import math
import random
import unittest
from collections import deque, Counter

from src.fast_io import FastIO




class SpiralMatrix:
    def __init__(self):
        return

    @staticmethod
    def joseph_ring(n, m):
        # 模板: 约瑟夫环计算最后的幸存者
        # 0.1..n-1每次选取第m个消除之后剩下的编号
        f = 0
        for x in range(2, n + 1):
            f = (m + f) % x
        return f

    @staticmethod
    def num_to_loc(m, n, num):
        # 根据从左往右从上往下的顺序生成给定数字的行列索引
        # 0123、4567
        return [num // n, num % n]

    @staticmethod
    def loc_to_num(r, c, m, n):
        # 根据从左往右从上往下的顺序给定的行列索引生成数字
        return r * n + n

    @staticmethod
    def get_spiral_matrix_num1(m, n, r, c):  # 顺时针螺旋
        # 获取 m*n 矩阵的 [r, c] 元素位置（元素从 1 开始索引从 1 开始）
        num = 1
        while r not in [1, m] and c not in [1, n]:
            num += 2 * m + 2 * n - 4
            r -= 1
            c -= 1
            n -= 2
            m -= 2

        # 复杂度 O(m+n)
        x = y = 1
        direc = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        d = 0
        while [x, y] != [r, c]:
            a, b = direc[d]
            if not (1 <= x + a <= m and 1 <= y + b <= n):
                d += 1
                a, b = direc[d]
            x += a
            y += b
            num += 1
        return num

    @staticmethod
    def get_spiral_matrix_num2(m, n, r, c):  # 顺时针螺旋: 索引到数字序
        # 获取 m*n 矩阵的 [r, c] 元素位置（元素从 1 开始索引从 1 开始）

        rem = min(r - 1, m - r, c - 1, n - c)
        num = 2 * rem * (m - rem + 1) + 2 * rem * (n - rem + 1) - 4 * rem
        m -= 2 * rem
        n -= 2 * rem
        r -= rem
        c -= rem

        # 复杂度 O(1)
        if r == 1:
            num += c
        elif 1 < r <= m and c == n:
            num += n + (r - 1)
        elif r == m and 1 <= c <= n - 1:
            num += n + (m - 1) + (n - c)
        else:
            num += n + (m - 1) + (n - 1) + (m - r)
        return num

    @staticmethod
    def get_spiral_matrix_loc(m, n, num):  # 顺时针螺旋: 数字序到索引
        # 获取 m*n 矩阵的元素 num 位置

        def check(x):
            res = 2 * x * (m - x + 1) + 2 * x * (n - x + 1) - 4 * x
            return res < num

        low = 0
        high = max(m // 2, n // 2)
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                low = mid
            else:
                high = mid
        rem = high if check(high) else low

        num -= 2 * rem * (m - rem + 1) + 2 * rem * (n - rem + 1) - 4 * rem
        assert num > 0
        m -= 2 * rem
        n -= 2 * rem
        r = c = rem

        if num <= n:
            a = 1
            b = num
        elif n < num <= n + m - 1:
            a = num - n + 1
            b = n
        elif n + (m - 1) < num <= n + (m - 1) + (n - 1):
            a = m
            b = n - (num - n - (m - 1))
        else:
            a = m - (num - n - (n - 1) - (m - 1))
            b = 1
        return [r + a, c + b]




