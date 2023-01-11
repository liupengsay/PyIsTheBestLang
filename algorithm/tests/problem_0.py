from __future__ import division
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


ans = 1  # 标记是第几层
ans1 = 1  # 最高层数
pr = FastIO().st


class BST(object):
    def __init__(self, data, left=None, right=None):  # BST的三个，值，左子树，右子树
        super(BST, self).__init__()
        self.data = data
        self.left = left
        self.right = right

    def insert(self, val):  # 插入函数
        global ans, ans1
        if val < self.data:  # 如果小于就放到左子树
            if self.left:  # 如果有左子树
                ans += 1  # 层数+1
                self.left.insert(val)  # 左子树调用递归
            else:  # 没有左子树
                ans += 1  # 层数+1
                self.left = BST(val)  # 把值放在这个点的左子树上
                if ans > ans1:  # 如果层数比之前最高层数高
                    ans1 = ans  # 替换
                ans = 1  # 重新开始
        else:  # 比节点的值大
            if self.right:  # 有右子树
                ans += 1  # 层数+1
                self.right.insert(val)  # 右子树调用递归
            else:  # 没有右子树
                ans += 1  # 层数+1
                self.right = BST(val)  # 将值放在右子树上
                if ans > ans1:  # 如果层数大于最高层数
                    ans1 = ans  # 覆盖
                ans = 1  # 重新开始
        return

    def post_order(self):  # 后序遍历：左，右，根
        if self.left:  # 有左子树
            self.left.post_order()  # 先遍历它的左子树
        if self.right:  # 有右子树
            self.right.post_order()  # 遍历右子树
        pr(self.data)  # 都没有或已经遍历完就输出值
        return


def main(ac=FastIO()):
    n = ac.read_int()
    nums = ac.read_list_ints()
    bst = BST(nums[0])
    for num in nums[1:]:
        bst.insert(num)
    ac.st(f"deep={ans1}")
    bst.post_order()
    return


main()
