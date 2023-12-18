import random
import sys
from sys import stdin, stdout
from types import GeneratorType

RANDOM = random.randint(0, 10 ** 9 + 7)


class Wrapper(int):
    # deel with hash crush
    def __init__(self, x):
        int.__init__(x)

    def __hash__(self):
        # xor the num as hashmap key
        return super(Wrapper, self).__hash__() ^ RANDOM


read = lambda: sys.stdin.readline().rstrip()


class FastIO:
    def __init__(self):
        self.random_seed = random.randint(0, 10 ** 9 + 7)
        return

    @staticmethod
    def read_int():
        return int(stdin.readline().rstrip())

    @staticmethod
    def read_float():
        return float(stdin.readline().rstrip())

    @staticmethod
    def read_list_ints():
        return list(map(int, stdin.readline().rstrip().split()))

    @staticmethod
    def read_list_floats():
        return list(map(float, stdin.readline().rstrip().split()))

    @staticmethod
    def read_list_ints_minus_one():
        return list(map(lambda x: int(x) - 1, stdin.readline().rstrip().split()))

    @staticmethod
    def read_str():
        return stdin.readline().rstrip()

    @staticmethod
    def read_list_strs():
        return stdin.readline().rstrip().split()

    @staticmethod
    def read_list_str():
        return list(stdin.readline().rstrip())

    @staticmethod
    def st(x):
        return print(x)

    @staticmethod
    def lst(x):
        return print(*x)

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
    def ceil(a, b):
        return a // b + int(a % b != 0)

    def hash_num(self, x):
        return x ^ self.random_seed

    @staticmethod
    def accumulate(nums):
        n = len(nums)
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + nums[i]
        return pre

    def inter_ask(self, lst):
        self.lst(lst)
        stdout.flush()  # which is necessary
        res = self.read_int()
        return res

    def inter_out(self, lst):
        self.lst(lst)
        stdout.flush()  # which is necessary
        return

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
