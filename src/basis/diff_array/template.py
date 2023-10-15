import math
import unittest
from collections import defaultdict, deque
from itertools import accumulate
from typing import List

from src.basis.binary_search import BinarySearch
from src.fast_io import FastIO
import bisect
from math import inf



class PreFixSumMatrix:
    def __init__(self, mat):
        self.mat = mat
        # 二维前缀和
        m, n = len(mat), len(mat[0])
        self.pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                self.pre[i + 1][j + 1] = self.pre[i][j + 1] + \
                    self.pre[i + 1][j] - self.pre[i][j] + mat[i][j]

    def query(self, xa, ya, xb, yb):
        # 二维子矩阵和查询，索引从 0 开始，左上角 [xa, ya] 右下角 [xb, yb]
        return self.pre[xb + 1][yb + 1] - self.pre[xb +
                                                   1][ya] - self.pre[xa][yb + 1] + self.pre[xa][ya]


class DiffArray:
    def __init__(self):
        return

    @staticmethod
    def get_diff_array(n, shifts):
        # 一维差分数组
        diff = [0] * n
        for i, j, d in shifts:
            if j + 1 < n:
                diff[j + 1] -= d
            diff[i] += d
        for i in range(1, n):
            diff[i] += diff[i - 1]
        return diff

    @staticmethod
    def get_array_prefix_sum(n, lst):
        # 一维前缀和
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + lst[i]
        return pre

    @staticmethod
    def get_array_range_sum(pre, left, right):
        # 区间元素和
        return pre[right + 1] - pre[left]


class DiffMatrix:
    def __init__(self):
        return

    @staticmethod
    def get_diff_matrix(m, n, shifts):
        # 二维差分数组
        diff = [[0] * (n + 2) for _ in range(m + 2)]
        # 索引从 1 开始，矩阵初始值为 0
        for xa, xb, ya, yb, d in shifts:  # 注意这里的行列索引范围，是从左上角到右下角
            diff[xa][ya] += d
            diff[xa][yb + 1] -= d
            diff[xb + 1][ya] -= d
            diff[xb + 1][yb + 1] += d

        for i in range(1, m + 2):
            for j in range(1, n + 2):
                diff[i][j] += diff[i - 1][j] + \
                    diff[i][j - 1] - diff[i - 1][j - 1]

        for i in range(1, m + 1):
            diff[i] = diff[i][1:n + 1]
        return diff[1: m + 1]

    @staticmethod
    def get_diff_matrix2(m, n, shifts):
        diff = [[0] * (n + 1) for _ in range(m + 1)]
        # 二维差分，索引从 0 开始， 注意这里的行列索引范围，是从左上角到右下角
        for xa, xb, ya, yb, d in shifts:
            diff[xa][ya] += d
            diff[xa][yb + 1] -= d
            diff[xb + 1][ya] -= d
            diff[xb + 1][yb + 1] += d

        res = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                res[i + 1][j + 1] = res[i + 1][j] + \
                    res[i][j + 1] - res[i][j] + diff[i][j]
        return [item[1:] for item in res[1:]]

    @staticmethod
    def get_matrix_prefix_sum(mat):
        # 二维前缀和
        m, n = len(mat), len(mat[0])
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                pre[i + 1][j + 1] = pre[i][j + 1] + \
                    pre[i + 1][j] - pre[i][j] + mat[i][j]
        return pre

    @staticmethod
    def get_matrix_range_sum(pre, xa, ya, xb, yb):
        # 二维子矩阵和
        return pre[xb + 1][yb + 1] - pre[xb + 1][ya] - \
            pre[xa][yb + 1] + pre[xa][ya]


