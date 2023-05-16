import bisect
import decimal
import heapq
from types import GeneratorType
from math import inf
import sys
from functools import cmp_to_key
from collections import defaultdict, Counter, deque
import math
from functools import lru_cache
from heapq import nlargest
from functools import reduce
import random
from itertools import combinations
from operator import xor, add
from operator import mul
from typing import List, Callable, Dict, Set, Tuple, DefaultDict


# import sys
# from collections import deque
#
# read = lambda: sys.stdin.readline()
#
# m, n = list(map(int, read().split()))
# grid = []
# for _ in range(m):
#     grid.append(list(map(int, read().split())))


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

    def ask(self, lst):
        self.lst(lst)
        sys.stdout.flush()
        res = self.read_int()
        return res

    @staticmethod
    def accumulate(nums):
        n = len(nums)
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + nums[i]
        return pre

    @staticmethod
    def plot_graph(n, edges):

        # 创建一个有向图
        g = nx.Graph()
        for i in range(n):
            # 添加节点
            g.add_node(i+1)
        for i in range(n):
            for j in edges[i]:
                g.add_edge(i+1, j+1)
        nx.draw_networkx(g)
        plt.show()
        return

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


class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):
        n = ac.read_int()
        nums = ac.read_list_ints()
        pre = ac.accumulate(nums)
        dp = [[inf] * n for _ in range(n)]
        mid = [[-1] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            dp[i][i] = 0
            for j in range(i + 1, n):
                ind = i
                for k in range(i, j):
                    cur = dp[i][k] + dp[k + 1][j] + pre[j + 1] - pre[i]
                    if cur < dp[i][j]:
                        dp[i][j] = cur
                        ind = k
                mid[i][j] = ind

        ans = []
        nums = [str(x) for x in nums]
        stack = [[0, n - 1]]
        while stack:
            i, j = stack.pop()
            if i >= 0:
                stack.append([~i, j])
                if i >= j - 1:
                    continue
                k = mid[i][j]
                stack.append([k + 1, j])
                stack.append([i, k])
            else:
                i = ~i
                if i < j:
                    nums[i] = "(" + nums[i]
                    nums[j] = nums[j] + ")"
                    ans.append(pre[j + 1] - pre[i])
        ac.st("+".join(nums))
        ac.st(sum(ans))
        ac.lst(ans)
        return


Solution().main()
