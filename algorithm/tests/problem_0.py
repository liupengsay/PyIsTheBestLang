
import random
import heapq
import math
import sys
import bisect
from functools import lru_cache
from collections import Counter
from collections import defaultdict
from itertools import combinations
from itertools import permutations

sys.setrecursionlimit(10000000)


class FastIO:
    def __init__(self):
        return

    @staticmethod
    def _read():
        return sys.stdin.readline().strip()

    def read_int(self):
        return int(self._read())

    def read_float(self):
        return int(self._read())

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

    def read_list_str(self):
        return self._read().split()

    @staticmethod
    def st(x):
        return sys.stdout.write(str(x) + '\n')

    @staticmethod
    def lst(x):
        return sys.stdout.write(" ".join(str(w) for w in x) + '\n')


def get_all_factor(a):
    # 获取整数所有的因子包括 1 和它自己
    factor = set()
    for i in range(1, int(math.sqrt(a)) + 1):
        if a % i == 0:
            factor.add(i)
            factor.add(a // i)
    return sorted(list(factor))


def main(ac=FastIO()):

    def check():

        exp = 0
        life = 10
        n = ac.read_int()
        for _ in range(n):
            x, a = ac.read_floats()
            if life - x <= 0:
                return exp
            life -= x
            life = min(life, 10)
            exp += max(0, a)
        return exp

    ans = int(check())
    m = 0
    while ans >= 2**m:
        ans -= 2**m
        m += 1

    ac.lst([m, ans])
    return


main()
