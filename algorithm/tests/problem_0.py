import copy
import random
import heapq
import math
import sys
import bisect
import datetime
from functools import lru_cache
from collections import deque
from collections import Counter
from collections import defaultdict
from itertools import combinations
from itertools import permutations
from decimal import Decimal, getcontext, MAX_PREC
from types import GeneratorType
from functools import cmp_to_key
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
    def mmax(a, b):
        return a if a > b else b

    @staticmethod
    def mmin(a, b):
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


def get_recent_palindrom_num(n: str) -> list:
    # 564. 寻找最近的回文数（https://leetcode.cn/problems/find-the-closest-palindrome/）
    # P1609 最小回文数（https://www.luogu.com.cn/problem/P1609）
    # 用原数的前半部分加一后的结果替换后半部分得到的回文整数。
    # 用原数的前半部分减一后的结果替换后半部分得到的回文整数。
    # 为防止位数变化导致构造的回文整数错误，因此直接构造 999…999 和 100…001 作为备选答案
    # 计算正整数 n 附近的回文数，获得最近的最小或者最大的回文数

    m = len(n)
    candidates = [10 ** (m - 1) - 1, 10 ** m + 1]
    prefix = int(n[:(m + 1) // 2])
    for x in range(prefix - 1, prefix + 2):
        y = x if m % 2 == 0 else x // 10
        while y:
            x = x * 10 + y % 10
            y //= 10
        candidates.append(x)
    return candidates


def main(ac=FastIO()):
    n = ac.read_str()
    nums = get_recent_palindrom_num(n)
    nums = [num for num in nums if num > int(n)]
    ac.st(min(nums))
    return


main()
