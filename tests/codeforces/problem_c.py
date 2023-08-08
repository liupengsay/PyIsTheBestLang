import heapq
from math import inf
import heapq
import random
import sys
from math import inf
from typing import List, Dict


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


class Dijkstra:
    def __init__(self):
        return

    @staticmethod
    def get_dijkstra_cnt(dct: List[int], src: int) -> (List[int], List[int]):
        # 模板: Dijkstra求最短路条数（最短路计算）
        n = len(dct)
        dis = [inf]*n
        stack = [[0, src]]
        dis[src] = 0
        cnt = [0]*n
        cnt[src] = 1
        while stack:
            d, i = heapq.heappop(stack)
            if dis[i] < d:
                continue
            for j, w in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    # 最短距离更新，重置计数
                    cnt[j] = cnt[i]
                    heapq.heappush(stack, [dj, j])
                elif dj == dis[j]:
                    # 最短距离一致，增加计数
                    cnt[j] += cnt[i]
        return cnt, dis


class Solution:
    def __init__(self):
        return

    @staticmethod
    def ac_3772(ac=FastIO()):
        # 模板：经典建立反图并使用Dijkstra最短路计数贪心模拟
        n, m = ac.read_ints()
        rev = [[] for _ in range(n)]
        for _ in range(m):
            u, v = ac.read_ints_minus_one()
            rev[v].append([u, 1])

        k = ac.read_int()
        p = ac.read_list_ints_minus_one()
        cnt, dis = Dijkstra().get_dijkstra_cnt(rev, p[-1])

        floor = 0
        for i in range(k-1):
            if k-i-1 == dis[p[i]]:
                break
            if dis[p[i-1]] == dis[p[i]] + 1:
                continue
            else:
                floor += 1

        ceil = 0
        for i in range(k-1):
            if dis[p[i]] == dis[p[i+1]] + 1 and cnt[p[i]] == cnt[p[i+1]]:
                continue
            else:
                ceil += 1
        ac.lst([floor, ceil])

        return


Solution().main()
