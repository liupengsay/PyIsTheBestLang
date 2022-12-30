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
        return list(self._read())

    def read_list_strs(self):
        return self._read().split()

    @staticmethod
    def st(x):
        return sys.stdout.write(str(x) + '\n')

    @staticmethod
    def lst(x):
        return sys.stdout.write(" ".join(str(w) for w in x) + '\n')


def main(ac=FastIO()):
    n = ac.read_int()
    cnt = [0]*26
    pre = 1
    order = 1
    ind = 0
    for _ in range(n):
        lst = ac.read_list_strs()
        cur = int(lst[1])
        x = (cur-pre+1)//26
        for i in range(26):
            cnt[i] += x
        for _ in range((cur-pre+1)%26):
            cnt[ind] += 1
            ind += order
            ind %= 26
        pre = cur + 1
        if lst[0] == "UPIT":
            ac.st(cnt[ord(lst[2])-ord("a")])
        else:
            ind -= order
            order *= -1
            ind += order
            ind %= 26
    return


main()
