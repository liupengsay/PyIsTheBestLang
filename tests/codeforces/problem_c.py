import bisect
import decimal
import heapq
from types import GeneratorType
from math import inf
import sys
from heapq import heappush, heappop, heappushpop
from functools import cmp_to_key
from collections import defaultdict, Counter, deque
import math
from functools import lru_cache
from heapq import nlargest
from functools import reduce
import random
from itertools import combinations, permutations
from operator import xor, add
from operator import mul
from typing import List, Callable, Dict, Set, Tuple, DefaultDict


class FastIO:
    def __init__(self):
        return

    @staticmethod
    def read_int():
        return int(sys.stdin.readline().strip())

    @staticmethod
    def read_float():
        return float(sys.stdin.readline().strip())

    @staticmethod
    def read_ints():
        return map(int, sys.stdin.readline().strip().split())

    @staticmethod
    def read_floats():
        return map(float, sys.stdin.readline().strip().split())

    @staticmethod
    def read_ints_minus_one():
        return map(lambda x: int(x) - 1, sys.stdin.readline().strip().split())

    @staticmethod
    def read_list_ints():
        return list(map(int, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_list_floats():
        return list(map(float, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_list_ints_minus_one():
        return list(map(lambda x: int(x) - 1, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_str():
        return sys.stdin.readline().strip()

    @staticmethod
    def read_list_strs():
        return sys.stdin.readline().strip().split()

    @staticmethod
    def read_list_str():
        return list(sys.stdin.readline().strip())

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

    def ask(self, lst):
        # CF交互题输出询问并读取结果
        self.lst(lst)
        sys.stdout.flush()
        res = self.read_int()
        # 记得任何一个输出之后都要 sys.stdout.flush() 刷新
        return res

    def out_put(self, lst):
        # CF交互题输出最终答案
        self.lst(lst)
        sys.stdout.flush()
        return

    @staticmethod
    def accumulate(nums):
        n = len(nums)
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + nums[i]
        return pre

    @staticmethod
    def get_random_seed():
        # 随机种子避免哈希冲突
        return random.randint(0, 10**9+7)


class SegmentTreeRangeUpdateQuerySumMinMax:
    def __init__(self, n) -> None:
        # 模板：区间值增减、区间和查询、区间最小值查询、区间最大值查询
        self.n = n
        self.lazy = [0] * (4 * self.n)  # 懒标记
        self.floor = [inf] * (4 * self.n)  # 最小值
        return

    @staticmethod
    def max(a: int, b: int) -> int:
        return a if a > b else b

    @staticmethod
    def min(a: int, b: int) -> int:
        return a if a < b else b

    def build(self, nums: List[int]) -> None:
        # 使用数组初始化线段树
        assert self.n == len(nums)
        stack = [[0, self.n - 1, 1]]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self.floor[ind] = nums[s]
                    self.lazy[ind] = nums[s]
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self.push_up(ind)
        return

    def push_down(self, i: int, s: int, m: int, t: int) -> None:
        # 下放懒标记
        if self.lazy[i]:
            self.floor[2 * i] += self.lazy[i]
            self.floor[2 * i + 1] += self.lazy[i]

            self.lazy[2 * i] += self.lazy[i]
            self.lazy[2 * i + 1] += self.lazy[i]

            self.lazy[i] = 0

    def push_up(self, i) -> None:
        self.floor[i] = self.min(self.floor[2 * i], self.floor[2 * i + 1])
        return

    def make_tag(self, i, s, t, val) -> None:
        self.floor[i] += val
        self.lazy[i] += val
        return

    def update_range(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 增减区间值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.make_tag(i, s, t, val)
                    continue

                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, ~i])

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self.push_up(i)
        return

    def update_point(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 增减单点值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始

        while True:
            if left <= s and t <= right:
                self.make_tag(i, s, t, val)
                break
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:  # 注意左右子树的边界与范围
                s, t, i = s, m, 2 * i
            if right > m:
                s, t, i = m + 1, t, 2 * i + 1
        while i > 1:
            i //= 2
            self.push_up(i)
        return

    def query_min(self, left: int, right: int, s: int, t: int, i: int) -> int:
        # 查询区间的最小值
        stack = [[s, t, i]]
        highest = inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = self.min(highest, self.floor[i])
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return highest

    def get_all_nums(self) -> List[int]:
        # 查询区间的所有值
        stack = [[0, self.n-1, 1]]
        nums = [0]*self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.floor[i]
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            stack.append([s, m, 2 * i])
            stack.append([m + 1, t, 2 * i + 1])
        return nums


class Solution:
    def __init__(self):
        return

    @staticmethod
    def ac_3805(ac=FastIO()):
        # 模板：区间增减与最小值查询
        n = ac.read_int()
        tree = SegmentTreeRangeUpdateQuerySumMinMax(n)
        tree.build(ac.read_list_ints())
        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            if len(lst) == 2:
                l, r = lst
                if l <= r:
                    ac.st(tree.query_min(l, r, 0, n-1, 1))
                else:
                    ans1 = tree.query_min(l, n-1, 0, n-1, 1)
                    ans2 = tree.query_min(0, r, 0, n-1, 1)
                    ac.st(ac.min(ans1, ans2))
            else:
                l, r, d = lst
                if l <= r:
                    tree.update_range(l, r, 0, n-1, d, 1)
                else:
                    tree.update_range(l, n-1, 0, n-1, d, 1)
                    tree.update_range(0, r, 0, n-1, d, 1)
        return


Solution().main()
