import math
from functools import reduce
from math import lcm, gcd
from operator import or_, and_


class SparseTable:
    def __init__(self, lst, fun):
        """static range queries can be performed as long as the range_merge_to_disjoint fun satisfies monotonicity"""
        n = len(lst)
        self.bit = [0] * (n + 1)
        self.fun = fun
        self.n = n
        for i in range(2, n + 1):
            self.bit[i] = self.bit[i >> 1] + 1
        for i in range(n+1):
            assert self.bit[i] == (i.bit_length() - 1 if i else i.bit_length())
        self.st = [[0] * n for _ in range(self.bit[-1] + 1)]
        self.st[0] = lst
        for i in range(1, self.bit[-1] + 1):
            for j in range(n - (1 << i) + 1):
                self.st[i][j] = fun(self.st[i - 1][j], self.st[i - 1][j + (1 << (i - 1))])

    def query(self, left, right):
        """index start from 0"""
        assert 0 <= left <= right < self.n
        pos = self.bit[right - left + 1]
        return self.fun(self.st[pos][left], self.st[pos][right - (1 << pos) + 1])

    def bisect_right(self, left, val, initial):
        """index start from 0"""
        assert 0 <= left < self.n
        # find the max right such that st.query(left, right) >= val
        pos = left
        pre = initial  # 0 or (1<<32)-1
        for x in range(self.bit[-1], -1, -1):
            if pos + (1 << x) - 1 < self.n and self.fun(self.st[x][pos], pre) >= val: # can by any of >= > <= <
                pre = self.fun(self.st[x][pos], pre)
                pos += (1 << x)
        # may be pos=left and st.query(left, left) < val
        if pos > left:
            pos -= 1
        else:
            pre = self.st[0][left]
        assert left <= pos < self.n
        return pos, pre

class SparseTableIndex:
    def __init__(self, lst, fun):
        """static range queries can be performed as long as the range_merge_to_disjoint fun satisfies monotonicity"""
        n = len(lst)
        self.bit = [0] * (n + 1)
        self.n = n
        self.fun = fun
        l, r, v = 1, 2, 0
        while True:
            for i in range(l, r):
                if i >= len(self.bit):
                    break
                self.bit[i] = v
            else:
                l *= 2
                r *= 2
                v += 1
                continue
            break
        self.st = [[0] * n for _ in range(self.bit[-1] + 1)]
        self.st[0] = list(range(n))
        for i in range(1, self.bit[-1] + 1):
            for j in range(n - (1 << i) + 1):
                a, b = self.st[i - 1][j], self.st[i - 1][j + (1 << (i - 1))]
                if self.fun(lst[a], lst[b]) == lst[a]:
                    self.st[i][j] = a
                else:
                    self.st[i][j] = b
        self.lst = lst
        return

    def query(self, left, right):
        """index start from 0"""
        assert 0 <= left <= right <= self.n - 1
        pos = self.bit[right - left + 1]
        a, b = self.st[pos][left], self.st[pos][right - (1 << pos) + 1]
        if self.fun(self.lst[a], self.lst[b]) == self.lst[a]:
            return a
        return b


class SparseTable2D:
    def __init__(self, matrix, method="max"):
        m, n = len(matrix), len(matrix[0])
        a, b = int(math.log2(m)) + 1, int(math.log2(n)) + 1

        if method == "max":
            self.fun = max
        elif method == "min":
            self.fun = min
        elif method == "gcd":
            self.fun = self.gcd
        elif method == "lcm":
            self.fun = min
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
