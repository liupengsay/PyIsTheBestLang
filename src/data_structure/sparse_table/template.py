import math
from functools import reduce
from math import lcm, gcd
from operator import or_, and_


class SparseTable:
    def __init__(self, lst, fun):
        """static range queries can be performed as long as the range_merge_to_disjoint fun satisfies monotonicity"""
        self.fun = fun  # min max and or lcm gcd
        self.n = len(lst)
        self.m = int(math.log2(self.n))
        self.f = [0] * ((self.m + 1) * (self.n + 1))
        self.build(lst)
        return

    def build(self, lst):
        """the same as multiplication method for tree_lca"""
        for i in range(1, self.n + 1):
            self.f[i * (self.m + 1) + 0] = lst[i - 1]
        for j in range(1, self.m + 1):
            for i in range(1, self.n - (1 << j) + 2):
                a = self.f[i * (self.m + 1) + j - 1]
                b = self.f[(i + (1 << (j - 1))) * (self.m + 1) + j - 1]
                self.f[i * (self.m + 1) + j] = self.fun(a, b)
        return

    def query(self, left, right):
        """index start from 1"""
        k = int(math.log2(right - left + 1))
        a = self.f[left * (self.m + 1) + k]
        b = self.f[(right - (1 << k) + 1) * (self.m + 1) + k]
        return self.fun(a, b)


class SparseTable2D:
    def __init__(self, matrix, method="max"):
        m, n = len(matrix), len(matrix[0])
        a, b = int(math.log2(m)) + 1, int(math.log2(n)) + 1

        if method == "max":
            self.fun = self.max
        elif method == "min":
            self.fun = self.min
        elif method == "gcd":
            self.fun = self.gcd
        elif method == "lcm":
            self.fun = self.min
        elif method == "or":
            self.fun = self._or
        else:
            self.fun = self._and

        self.dp = [[[[0 for _ in range(b)] for _ in range(a)] for _ in range(1000)] for _ in range(1000)]

        for i in range(a):
            for j in range(b):
                for x in range(m - (1 << i) + 1):
                    for y in range(n - (1 << j) + 1):
                        if i == 0 and j == 0:
                            self.dp[x][y][i][j] = matrix[x][y]
                        elif i == 0:
                            self.dp[x][y][i][j] = self.fun([self.dp[x][y][i][j - 1],
                                                            self.dp[x][y + (1 << (j - 1))][i][j - 1]])
                        elif j == 0:
                            self.dp[x][y][i][j] = self.fun([self.dp[x][y][i - 1][j],
                                                            self.dp[x + (1 << (i - 1))][y][i - 1][j]])
                        else:
                            self.dp[x][y][i][j] = self.fun([self.dp[x][y][i - 1][j - 1],
                                                            self.dp[x + (1 << (i - 1))][y][i - 1][j - 1],
                                                            self.dp[x][y + (1 << (j - 1))][i - 1][j - 1],
                                                            self.dp[x + (1 << (i - 1))][y + (1 << (j - 1))][i - 1][
                                                                j - 1]])
        return

    @staticmethod
    def max(args):
        return reduce(max, args)

    @staticmethod
    def min(args):
        return reduce(min, args)

    @staticmethod
    def gcd(args):
        return reduce(gcd, args)

    @staticmethod
    def lcm(args):
        return reduce(lcm, args)

    @staticmethod
    def _or(args):
        return reduce(or_, args)

    @staticmethod
    def _and(args):
        return reduce(and_, args)

    def query(self, x, y, x1, y1):
        # index start from 0 and left up corner is (x, y) and right down corner is (x1, y1)
        k = int(math.log2(x1 - x + 1))
        p = int(math.log2(y1 - y + 1))
        ans = self.fun([self.dp[x][y][k][p],
                        self.dp[x1 - (1 << k) + 1][y][k][p],
                        self.dp[x][y1 - (1 << p) + 1][k][p],
                        self.dp[x1 - (1 << k) + 1][y1 - (1 << p) + 1][k][p]])
        return ans


class SparseTableIndex:
    def __init__(self, lst, fun="max"):
        # as long as Fun satisfies monotonicity
        # static interval queries can be performed on the index where the maximum is located
        self.fun = fun
        self.n = len(lst)
        self.lst = lst
        self.f = [[0] * (int(math.log2(self.n)) + 1)
                  for _ in range(self.n + 1)]
        self.build()
        return

    def build(self):
        for i in range(1, self.n + 1):
            self.f[i][0] = i - 1
        for j in range(1, int(math.log2(self.n)) + 1):
            for i in range(1, self.n - (1 << j) + 2):
                a = self.f[i][j - 1]
                b = self.f[i + (1 << (j - 1))][j - 1]
                if self.fun == "max":
                    self.f[i][j] = a if self.lst[a] > self.lst[b] else b
                elif self.fun == "min":
                    self.f[i][j] = a if self.lst[a] < self.lst[b] else b
        return

    def query(self, left, right):
        assert 1 <= left <= right <= self.n
        k = int(math.log2(right - left + 1))
        a = self.f[left][k]
        b = self.f[right - (1 << k) + 1][k]
        if self.fun == "max":
            return a if self.lst[a] > self.lst[b] else b
        elif self.fun == "min":
            return a if self.lst[a] < self.lst[b] else b
