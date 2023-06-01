"""
"""
from algorithm.src.fast_io import FastIO
from algorithm.src.graph.union_find import UnionFind

"""

算法：拓扑排序、内向基环树（有向或者无向，连通块有k个节点以及k条边）
功能：有向图进行排序，无向图在选定根节点的情况下也可以进行拓扑排序
题目：xx（xx）
内向基环树介绍：https://leetcode.cn/problems/maximum-employees-to-be-invited-to-a-meeting/solution/nei-xiang-ji-huan-shu-tuo-bu-pai-xu-fen-c1i1b/

===================================力扣===================================
360. 图中的最长环（https://leetcode.cn/problems/longest-cycle-in-a-graph/）拓扑排序计算有向图内向基环树最长环
2392. 给定条件下构造矩阵（https://leetcode.cn/problems/build-a-matrix-with-conditions/）分别通过行列的拓扑排序来确定数字所在索引，数字可能相同，需要使用并查集
2371. 最小化网格中的最大值（https://leetcode.cn/problems/minimize-maximum-value-in-a-grid/）分别通过行列的拓扑排序来确定数字所在索引，数字都不同可以使用贪心
2127. 参加会议的最多员工数（https://leetcode.cn/problems/maximum-employees-to-be-invited-to-a-meeting/）拓扑排序确定内向基环，按照环的大小进行贪心枚举

127. 参加会议的最多员工数（https://leetcode.cn/problems/maximum-employees-to-be-invited-to-a-meeting/）
269. 火星词典（https://leetcode.cn/problems/alien-dictionary/）经典按照字典序建图，与拓扑排序的应用
6356. 收集树中金币（https://leetcode.cn/contest/weekly-contest-338/problems/collect-coins-in-a-tree/）无向图拓扑排序内向基环树


===================================洛谷===================================
P1960 郁闷的记者（https://www.luogu.com.cn/problem/P1960）计算拓扑排序是否唯一
P1992 不想兜圈的老爷爷（https://www.luogu.com.cn/problem/P1992）拓扑排序计算有向图是否有环
P2712 摄像头（https://www.luogu.com.cn/problem/P2712）拓扑排序计算非环节点数
P6145 [USACO20FEB]Timeline G（https://www.luogu.com.cn/problem/P6145）经典拓扑排序计算每个节点最晚的访问时间点
P1137 旅行计划（https://www.luogu.com.cn/problem/P1137）拓扑排序，计算可达的最长距离
P1347 排序（https://www.luogu.com.cn/problem/P1347）拓扑排序确定字典序与矛盾或者无唯一解
P1685 游览（https://www.luogu.com.cn/problem/P1685）拓扑排序计算路径条数
P3243 [HNOI2015]菜肴制作（https://www.luogu.com.cn/problem/P3243）经典反向建图拓扑排序结合二叉堆进行顺序模拟
P5536 【XR-3】核心城市（https://www.luogu.com.cn/problem/P5536）经典使用无向图拓扑排序从外到内消除最外圈的节点
P6037 Ryoku 的探索（https://www.luogu.com.cn/problem/P6037）经典无向图基环树并查集拓扑排序与环模拟计算

==================================AtCoder=================================
F - Well-defined Path Queries on a Namori（https://atcoder.jp/contests/abc266/）（无向图的内向基环树，求简单路径的树枝连通）



参考：OI WiKi（xx）
"""

import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
from functools import reduce
from operator import xor
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy


class TopologicalSort:
    def __init__(self):
        return

    @staticmethod
    def get_rank(n, edges):
        dct = [list() for _ in range(n)]
        degree = [0]*n
        for i, j in edges:
            degree[j] += 1
            dct[i].append(j)
        stack = [i for i in range(n) if not degree[i]]
        visit = [-1]*n
        step = 0
        while stack:
            for i in stack:
                visit[i] = step
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
            step += 1
        return visit

    # 内向基环树写法 https://atcoder.jp/contests/abc266/submissions/37717739
    # def main(ac=FastIO()):
    #     n = ac.read_int()
    #     edge = [[] for _ in range(n)]
    #     uf = UnionFind(n)
    #
    #     degree = [0] * n
    #     for _ in range(n):
    #         u, v = ac.read_ints_minus_one()
    #         edge[u].append(v)
    #         edge[v].append(u)
    #         degree[u] += 1
    #         degree[v] += 1
    #
    #     que = deque()
    #     for i in range(n):
    #         if degree[i] == 1:
    #             que.append(i)
    #     while que:
    #         now = que.popleft()
    #         nex = edge[now][0]
    #         degree[now] -= 1
    #         degree[nex] -= 1
    #         edge[nex].remove(now)
    #         uf.union(now, nex)
    #         if degree[nex] == 1:
    #             que.append(nex)
    #
    #     q = ac.read_int()
    #     for _ in range(q):
    #         x, y = ac.read_ints_minus_one()
    #         if uf.is_connected(x, y):
    #             ac.st("Yes")
    #         else:
    #             ac.st("No")
    #     return

    @staticmethod
    def is_topology_unique(dct, degree, n):

        # 保证存在拓扑排序的情况下判断是否唯一
        ans = []
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            ans.extend(stack)
            if len(stack) > 1:
                return False
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return True

    @staticmethod
    def is_topology_loop(edge, degree, n):

        # 使用拓扑排序判断有向图是否存在环
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            nex = []
            for i in stack:
                for j in edge[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return sum(degree) == 0

class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_2360(edges: List[int]) -> int:
        # 模板：拓扑排序计算有向图内向基环树最长环
        n = len(edges)
        # 记录入度
        degree = defaultdict(int)
        for i in range(n):
            if edges[i] != -1:
                degree[edges[i]] += 1

        # 先消除无环部分
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            nex = []
            for i in stack:
                if edges[i] != -1:
                    degree[edges[i]] -= 1
                    if not degree[edges[i]]:
                        nex.append(edges[i])
            stack = nex

        # 注意没有自环
        visit = [int(degree[i] == 0) for i in range(n)]
        ans = -1
        for i in range(n):
            cnt = 0
            while not visit[i]:
                visit[i] = 1
                cnt += 1
                i = edges[i]
            if cnt and cnt > ans:
                ans = cnt
        return ans

    @staticmethod
    def lg_p1137(ac=FastIO()):
        # 模板：拓扑排序计算最长链条
        n, m = ac.read_ints()
        dct = [[] for _ in range(n)]
        degree = [0]*n
        for _ in range(m):
            i, j = ac.read_ints()
            degree[j-1] += 1
            dct[i-1].append(j-1)
        cnt = [1]*n
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            i = stack.pop()
            for j in dct[i]:
                cnt[j] = max(cnt[j], cnt[i]+1)
                degree[j] -= 1
                if not degree[j]:
                    stack.append(j)
        for a in cnt:
            ac.st(a)
        return

    @staticmethod
    def lg_p1347(ac=FastIO()):
        # 模板：拓扑排序确定字典序与矛盾或者无唯一解
        n, m = [int(w) for w in input().strip().split() if w]
        dct = defaultdict(list)
        degree = defaultdict(int)

        def check(dct, degree, nodes, x):
            stack = [k for k in nodes if not degree[k]]
            m = len(nodes)
            res = []
            unique = True
            while stack:
                res.extend(stack)
                if len(stack) > 1:
                    unique = False
                nex = []
                for i in stack:
                    for j in dct[i]:
                        degree[j] -= 1
                        if not degree[j]:
                            nex.append(j)
                stack = nex
            if unique and len(res) == n:
                s = "".join(res)
                return True, f"Sorted sequence determined after {x} relations: {s}."
            if len(res) < m:
                return True, f"Inconsistency found after {x} relations."
            return False, "Sorted sequence cannot be determined."

        nodes = set()
        res_ans = ""
        for x in range(1, m+1):
            s = input().strip()
            if res_ans:
                continue
            dct[s[0]].append(s[2])
            degree[s[2]] += 1
            nodes.add(s[0])
            nodes.add(s[2])
            flag, ans = check(copy.deepcopy(dct), copy.deepcopy(degree), copy.deepcopy(nodes), x)
            if flag:
                res_ans = ans
        if res_ans:
            print(res_ans)
        else:
            print("Sorted sequence cannot be determined.")
        return

    @staticmethod
    def lg_p1685(ac=FastIO()):
        # 模板：拓扑排序计算经过每条边的路径条数
        n, m, s, e, t = ac.read_ints()
        s -= 1
        e -= 1
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(m):
            i, j, w = ac.read_ints()
            i -= 1
            j -= 1
            dct[i].append([j, w])
            degree[j] += 1
        mod = 10000

        # 记录总时间与路径条数
        time = [0] * n
        cnt = [0] * n
        stack = [s]
        cnt[s] = 1
        while stack:
            i = stack.pop()
            for j, w in dct[i]:
                degree[j] -= 1
                if not degree[j]:
                    stack.append(j)
                cnt[j] += cnt[i]
                time[j] += cnt[i] * w + time[i]
                time[j] %= mod
        # 减去回头的时间
        ans = time[e] + (cnt[e] - 1) * t
        ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def lg_p3243(ac=FastIO()):
        # 模板：经典反向建图拓扑排序结合二叉堆进行顺序模拟
        for _ in range(ac.read_int()):
            n, m = ac.read_ints()
            dct = [[] for _ in range(n)]
            degree = [0] * n
            for _ in range(m):
                i, j = ac.read_ints_minus_one()
                dct[j].append(i)
                degree[i] += 1
            ans = []
            stack = [-i for i in range(n) if not degree[i]]
            heapq.heapify(stack)
            while stack:
                # 优先选择入度为 0 且编号最大的
                i = -heapq.heappop(stack)
                ans.append(i)
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        heapq.heappush(stack, -j)
            if len(ans) == n:
                # 翻转后则字典序最小
                ans.reverse()
                ac.lst([x + 1 for x in ans])
            else:
                ac.st("Impossible!")
        return

    @staticmethod
    def lg_p5536_1(ac=FastIO()):
        # 模板：使用直径的贪心方式选取以树的直径中点向外辐射的节点
        n, k = ac.read_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        x, y, _, path = TreeDiameterInfo().get_diameter_info(dct)

        # 树形 DP 计算每个节点的深度与子树最大节点深度
        root = path[len(path) // 2]
        deep = [0] * n
        max_deep = [0] * n
        stack = [[root, -1, 0]]
        while stack:
            i, fa, d = stack.pop()
            if i >= 0:
                stack.append([~i, fa, d])
                deep[i] = d
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i, d + 1])
            else:
                i = ~i
                max_deep[i] = deep[i]
                for j in dct[i]:
                    if j != fa:
                        max_deep[i] = ac.max(max_deep[i], max_deep[j])

        # 选取 k 个节点后剩下的节点的最大值
        lst = [max_deep[i] - deep[i] for i in range(n)]
        lst.sort(reverse=True)
        ac.st(lst[k] + 1)
        return

    @staticmethod
    def lg_p5536_2(ac=FastIO()):

        # 模板：使用无向图拓扑排序从外到内消除最外圈的节点
        n, k = ac.read_ints()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(n - 1):
            i, j = ac.read_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
            degree[i] += 1
            degree[j] += 1

        # 按照度为 1 进行 BFS 消除
        rem = n - k
        ans = 0
        stack = deque([[i, 1] for i in range(n) if degree[i] == 1])
        while rem:
            i, d = stack.popleft()
            ans = d
            rem -= 1
            for j in dct[i]:
                if degree[j] > 1:
                    degree[j] -= 1
                    if degree[j] == 1:
                        stack.append([j, d + 1])
        ac.st(ans)
        return

    @staticmethod
    def lg_p6037(ac=FastIO()):
        # 模板：经典无向图基环树并查集拓扑排序与环模拟计算
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        # 首先分割连通分量
        uf = UnionFind(n)
        degree = [0] * n
        edge = []
        for _ in range(n):
            u, v, w, p = ac.read_ints()
            u -= 1
            v -= 1
            dct[u].append([v, w, p])
            dct[v].append([u, w, p])
            uf.union(u, v)
            edge.append([u, v, w])
            degree[u] += 1
            degree[v] += 1
        # 其次对每个分量计算结果
        part = uf.get_root_part()
        ans = [-1] * n
        for p in part:
            # 拓扑排序找出环
            stack = deque([i for i in part[p] if degree[i] == 1])
            visit = set()
            while stack:
                i = stack.popleft()
                visit.add(i)
                for j, _, _ in dct[i]:
                    degree[j] -= 1
                    if degree[j] == 1:
                        stack.append(j)
            # 根据环贪心计算结果
            circle = [i for i in part[p] if i not in visit]
            s = sum(w for i, _, w in edge if uf.find(i) == p)
            for x in circle:
                cur = [[j, w, p] for j, w, p in dct[x] if degree[j] > 1]
                cur.sort(key=lambda it: it[-1])
                ans[x] = s - cur[0][1]
            stack = deque(circle)
            while stack:
                i = stack.popleft()
                for j, _, _ in dct[i]:
                    if ans[j] == -1:
                        ans[j] = ans[i]
                        stack.append(j)
        for a in ans:
            ac.st(a)
        return


class TestGeneral(unittest.TestCase):

    def test_topological_sort(self):
        ts = TopologicalSort()
        n = 5
        edges = [[0, 1], [0, 2], [1, 4], [2, 3], [3, 4]]
        assert ts.get_rank(n, edges) == [0, 1, 1, 2, 3]
        return


if __name__ == '__main__':
    unittest.main()
