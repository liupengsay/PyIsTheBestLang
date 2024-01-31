import bisect
import heapq
import sys
from types import GeneratorType
from functools import cmp_to_key
from collections import defaultdict, Counter, deque
import math
from functools import lru_cache
from heapq import nlargest
from functools import reduce
import random
from operator import mul
inf = float("inf")
PLATFORM = "CF"
if PLATFORM == "LUOGU":
    import numpy as np
    sys.setrecursionlimit(1000000)


class FastIO:
    def __init__(self):
        return

    @staticmethod
    def _read():
        return sys.stdin.readline().strip()

    def read_int(self):
        return int(self._read())

    def read_float(self):
        return float(self._read())

    def read_ints(self):
        return map(int, self._read().split())

    def read_floats(self):
        return map(float, self._read().split())

    def read_ints_minus_one(self):
        return map(lambda x: int(x) - 1, self._read().split())

    def read_list_ints(self):
        return list(map(int, self._read().split()))

    def read_list_floats(self):
        return list(map(float, self._read().split()))

    def read_list_ints_minus_one(self):
        return list(map(lambda x: int(x) - 1, self._read().split()))

    def read_str(self):
        return self._read()

    def read_list_strs(self):
        return self._read().split()

    def read_list_str(self):
        return list(self._read())

    @staticmethod
    def st(x):
        return sys.stdout.write(str(x) + '\n')

    @staticmethod
    def lst(x):
        return sys.stdout.write(" ".join(str(w) for w in x) + '\n')

    @staticmethod
    def round_5(f):
        res = int(f)
        if f - res >= 0.5:
            res += 1
        return res

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    @staticmethod
    def bootstrap(f, queue=[]):
        def wrappedfunc(*args, **kwargs):
            if queue:
                return f(*args, **kwargs)
            else:
                to = f(*args, **kwargs)
                while True:
                    if isinstance(to, GeneratorType):
                        queue.append(to)
                        to = next(to)
                    else:
                        queue.pop()
                        if not queue:
                            break
                        to = queue[-1].send(to)
                return to
        return wrappedfunc


class LazySegmentTree:
    def __init__(self, array):
        self.n = len(array)
        self.size = 1 << (self.n - 1).bit_length()
        self.func = min
        self.default = float("inf")
        self.data = [self.default] * (2 * self.size)
        self.lazy = [0] * (2 * self.size)
        self.process(array)

    def process(self, array):
        self.data[self.size: self.size + self.n] = array
        for i in range(self.size - 1, -1, -1):
            self.data[i] = self.func(self.data[2 * i], self.data[2 * i + 1])

    def push(self, index):
        """Push the information of the root to it's children!"""
        self.lazy[2 * index] += self.lazy[index]
        self.lazy[2 * index + 1] += self.lazy[index]
        self.data[2 * index] += self.lazy[index]
        self.data[2 * index + 1] += self.lazy[index]
        self.lazy[index] = 0

    def build(self, index):
        """Build data with the new changes!"""
        index >>= 1
        while index:
            self.data[index] = self.func(self.data[2 * index], self.data[2 * index + 1]) + self.lazy[index]
            index >>= 1

    def query(self, alpha, omega):
        """Returns the result of function over the range (inclusive)!"""
        res = self.default
        alpha += self.size
        omega += self.size + 1
        for i in reversed(range(1, alpha.bit_length())):
            self.push(alpha >> i)
        for i in reversed(range(1, (omega - 1).bit_length())):
            self.push((omega - 1) >> i)
        while alpha < omega:
            if alpha & 1:
                res = self.func(res, self.data[alpha])
                alpha += 1
            if omega & 1:
                omega -= 1
                res = self.func(res, self.data[omega])
            alpha >>= 1
            omega >>= 1
        return res

    def update(self, alpha, omega, value):
        """Increases all elements in the range (inclusive) by given value!"""
        alpha += self.size
        omega += self.size + 1
        l, r = alpha, omega
        while alpha < omega:
            if alpha & 1:
                self.data[alpha] += value
                self.lazy[alpha] += value
                alpha += 1
            if omega & 1:
                omega -= 1
                self.data[omega] += value
                self.lazy[omega] += value
            alpha >>= 1
            omega >>= 1
        self.build(l)
        self.build(r - 1)


def main(ac=FastIO()):

    n = ac.read_int()
    tree = LazySegmentTree(ac.read_list_ints())
    tot = 0
    for _ in range(ac.read_int()):
        lst = ac.read_list_ints()
        if len(lst) == 2:
            a, b = lst
            cur = [[a, b]] if a <= b else [[0, b], [a, n-1]]
            res = inf
            for x, y in cur:
                res = ac.min(res, tree.query(x, y))
            ac.st(res + tot)
        else:
            x, y, v = lst
            if x <= y:
                tree.update(x, y, v)
            else:
                tot += v
                if y + 1 <= x - 1:
                    tree.update(y + 1, x - 1, -v)
    return


main()