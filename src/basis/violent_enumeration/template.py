import bisect
import unittest
import math
from collections import defaultdict, deque
from functools import reduce, lru_cache
from itertools import combinations, permutations
from operator import mul, or_
from typing import List

from src.fast_io import FastIO, inf




class ViolentEnumeration:
    def __init__(self):
        return

    @staticmethod
    def matrix_rotate(matrix):  # 旋转矩阵

        # 将矩阵顺时针旋转 90 度
        n = len(matrix)
        for i in range(n // 2):
            for j in range((n + 1) // 2):
                a, b, c, d = matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1], matrix[i][j]
                matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1] = a, b, c, d

        # 将矩阵逆时针旋转 90 度
        n = len(matrix)
        for i in range(n // 2):
            for j in range((n + 1) // 2):
                a, b, c, d = matrix[j][n - i - 1], matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1]
                matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1] = a, b ,c ,d

        return matrix




