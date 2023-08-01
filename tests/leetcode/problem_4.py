import bisect
import random
import re
import unittest

from typing import List, Callable, Dict
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations, accumulate
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from heapq import heappush, heappop, heappushpop
from operator import xor, mul, add
from functools import lru_cache
from math import inf
import random
from itertools import permutations, combinations
import numpy as np
from typing import List, Callable
from decimal import Decimal

import heapq
import copy
from sortedcontainers import SortedList


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
    def print_info(st):
        st = """执行结果：
        通过
        显示详情
        查看示例代码
        00 : 00 : 04

        执行用时：
        108 ms
        , 在所有 Python3 提交中击败了
        23.15%
        的用户
        内存消耗：
        15.3 MB
        , 在所有 Python3 提交中击败了
        27.31%
        的用户
        通过测试用例：
        219 / 219"""
        lst = st.split("\n")
        lst[2] = " " + lst[2] + " "
        lst[-4] = " " + lst[-4] + " "
        lst[-9] = " " + lst[-9] + " "
        lst[4] = " " + lst[4].replace(" ", "") + " "
        lst[-1] = lst[-1].replace(" ", "")
        st1 = lst[:6]
        st2 = lst[6:11]
        st3 = lst[11:-2]
        st4 = lst[-2:]
        for s in [st1, st2, st3, st4]:
            print("- " + "".join(s))
        return

class DirectedEulerPath:
    def __init__(self, n, pairs):
        # 数组形式存储的有向连接关系
        self.n = n
        self.pairs = pairs
        # 欧拉路径上的每条边和经过的几点
        self.paths = list()
        self.nodes = list()
        self.exist = False
        self.get_euler_path()
        return

    def get_euler_path(self):
        # 存顶点的出入度
        degree = [0]*self.n
        # 存储图关系
        edge = [[] for _ in range(self.n)]
        for i, j in self.pairs:
            degree[i] += 1
            degree[j] -= 1
            edge[i].append(j)

        # 根据字典序优先访问较小的
        for i in range(self.n):
            edge[i].sort(reverse=True)

        # 寻找起始节点
        starts = []
        ends = []
        zero = 0
        for i in range(self.n):
            if degree[i] == 1:
                starts.append(i)
            elif degree[i] == -1:
                ends.append(i)
            else:
                zero += 1
        del degree

        # 图中恰好存在 1 个点出度比入度多 1（这个点即为起点） 1 个点出度比入度少 1（这个点即为终点）其余相等
        if not len(starts) == len(ends) == 1:
            if zero != self.n:
                return
            starts = [0]

        @FastIO.bootstrap
        def dfs(pre):
            # 使用深度优先搜索（Hierholzer算法）求解欧拉通路
            while edge[pre]:
                nex = edge[pre].pop()
                yield dfs(nex)
                self.nodes.append(nex)
                self.paths.append([pre, nex])
            yield

        dfs(starts[0])
        # 注意判断所有边都经过的才算欧拉路径
        self.paths.reverse()
        self.nodes.append(starts[0])
        self.nodes.reverse()
        if len(self.nodes) == len(self.pairs) + 1:
            self.exist = True
        return


class Solution:
    def validArrangement(self, pairs: List[List[int]]) -> List[List[int]]:
        nodes = set()
        for a, b in pairs:
            nodes.add(a)
            nodes.add(b)
        nodes = list(nodes)
        n = len(nodes)
        ind = {num: i for i, num in enumerate(nodes)}

        lst = [[ind[a], ind[b]] for a, b in pairs]

        ep = DirectedEulerPath(n, lst)
        ans = ep.paths
        return [[nodes[x], nodes[y]] for x, y in ans]



assert Solution().maxOutput(n = 6, edges = [[0,1],[1,2],[1,3],[3,4],[3,5]], price = [9,8,7,6,10,5]) == 24
