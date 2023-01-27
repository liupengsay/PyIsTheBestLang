from __future__ import division
import copy
import random
import heapq
import math
import sys
import bisect
import re
import time
import datetime
from functools import lru_cache
from collections import deque
from collections import Counter
from collections import defaultdict
from itertools import combinations
from itertools import permutations
from itertools import accumulate
from decimal import Decimal, getcontext, MAX_PREC
from types import GeneratorType
from functools import cmp_to_key
import datetime
import unittest
import time


inf = float("inf")
sys.setrecursionlimit(10000000)

getcontext().prec = MAX_PREC


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
    def bootstrap(f, stack=[]):
        def wrappedfunc(*args, **kwargs):
            if stack:
                return f(*args, **kwargs)
            else:
                to = f(*args, **kwargs)
                while True:
                    if isinstance(to, GeneratorType):
                        stack.append(to)
                        to = next(to)
                    else:
                        stack.pop()
                        if not stack:
                            break
                        to = stack[-1].send(to)
                return to
        return wrappedfunc

def sieve_of_eratosthenes(n):  # 埃拉托色尼筛选法，返回小于等于n的素数
    primes = [True] * (n + 1)  # 范围0到n的列表
    p = 2  # 这是最小的素数
    while p * p <= n:  # 一直筛到sqrt(n)就行了
        if primes[p]:  # 如果没被筛，一定是素数
            for i in range(p * 2, n + 1, p):  # 筛掉它的倍数即可
                primes[i] = False
        p += 1
    primes = [
        element for element in range(
            2, n + 1) if primes[element]]  # 得到所有小于等于n的素数
    return primes

dct = set(sieve_of_eratosthenes(10000))
days_dct = set(x for x in dct if 1<=x<=31)
months_dct = set(x for x in dct if 101<=x<=10000)
target = set()
for x in dct:
    if x >= 101 and int(str(x)[-2:]) in dct:
        target.add(x)
print(len(target))
leap_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
not_leap_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def is_leap_year(yy):
    # 闰年天数
    return yy % 400 == 0 or (yy % 4 == 0 and yy % 100 != 0)


def is_prime4(x):
    """https://zhuanlan.zhihu.com/p/107300262
    任何一个自然数，总可以表示成以下六种形式之一：6n，6n+1，6n+2，6n+3，6n+4，6n+5（n=0,1,2...）
    我们可以发现，除了2和3，只有形如6n+1和6n+5的数有可能是质数。
    且形如6n+1和6n+5的数如果不是质数，它们的因数也会含有形如6n+1或者6n+5的数，因此可以得到如下算法：
    """
    if x == 1:
        return False
    if (x == 2) or (x == 3):
        return True
    if (x % 6 != 1) and (x % 6 != 5):
        return False
    for i in range(5, int(math.sqrt(x)) + 1, 6):
        if (x % i == 0) or (x % (i + 2) == 0):
            return False
    return True

days = set()
for y in range(1, 10000):
    for m in target:
        if is_prime4(y*10000+m):
            days.add(str(y*10000+m))


class TrieCount:
    def __init__(self):
        self.dct = dict()
        return

    def update(self, word):
        # 更新单词与前缀计数
        cur = self.dct
        for w in word:
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
        return

    def query(self, word):
        # 查询前缀单词个数

        res = 0

        def dfs(i, cur):
            nonlocal res
            if i == 8:
                res += 1
                return
            if word[i].isnumeric():
                if word[i] in cur:
                    dfs(i+1, cur[word[i]])
            else:
                for w in cur:
                    dfs(i+1, cur[w])
            return
        dfs(0, self.dct)

        return res

print(len(days))
def main(ac=FastIO()):
    trie = TrieCount()
    for word in days:
        trie.update(word)
    assert "57070307" in days
    t = ac.read_int()
    for _ in range(t):
        s = ac.read_str()
        ac.st(trie.query(s))
    return


main()

