"""
"""
import math
from math import inf

from src.dp.tree_dp import TreeDiameterInfo
from src.fast_io import FastIO
from src.graph.union_find import UnionFind

"""

算法：拓扑排序、内向基环树（有向或者无向，连通块有k个节点以及k条边）、bfs序、拓扑序
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
2603. 收集树中金币（https://leetcode.cn/contest/weekly-contest-338/problems/collect-coins-in-a-tree/）无向图拓扑排序内向基环树
2204. 无向图中到环的距离（https://leetcode.cn/problems/distance-to-a-cycle-in-undirected-graph/https://leetcode.cn/problems/distance-to-a-cycle-in-undirected-graph/）无向图拓扑排序
1857. 有向图中最大颜色值（https://leetcode.cn/problems/largest-color-value-in-a-directed-graph/）经典拓扑排序DP
1932. 合并多棵二叉搜索树（https://leetcode.cn/problems/merge-bsts-to-create-single-bst/）经典连通性、拓扑排序与二叉搜索树判断
1591. 奇怪的打印机 II（https://leetcode.cn/contest/biweekly-contest-35/problems/strange-printer-ii/）经典建图判断拓扑排序是否无环


===================================洛谷===================================
P1960 郁闷的记者（https://www.luogu.com.cn/problem/P1960）计算拓扑排序是否唯一
P1992 不想兜圈的老爷爷（https://www.luogu.com.cn/problem/P1992）拓扑排序计算有向图是否有环
P2712 摄像头（https://www.luogu.com.cn/problem/P2712）拓扑排序计算非环节点数
P6145 [USACO20FEB]Timeline G（https://www.luogu.com.cn/problem/P6145）经典拓扑排序计算每个节点最晚的访问时间点
P1137 旅行计划（https://www.luogu.com.cn/problem/P1137）拓扑排序，计算可达的最长距离
P1347 排序（https://www.luogu.com.cn/problem/P1347）拓扑排序确定字典序与矛盾或者无唯一解
P1685 游览（https://www.luogu.com.cn/problem/P1685）经典DAG拓扑排序DP计算路径条数与耗时
P3243 [HNOI2015]菜肴制作（https://www.luogu.com.cn/problem/P3243）经典反向建图拓扑排序结合二叉堆进行顺序模拟
P5536 【XR-3】核心城市（https://www.luogu.com.cn/problem/P5536）经典使用无向图拓扑排序从外到内消除最外圈的节点
P6037 Ryoku 的探索（https://www.luogu.com.cn/problem/P6037）经典无向图基环树并查集拓扑排序与环模拟计算
P6255 [ICPC2019 WF]Dead-End Detector（https://www.luogu.com.cn/problem/P6255）简单无向图并查集计算连通块后使用拓扑排序寻找环的信息
P6417 [COCI2014-2015#1] MAFIJA（https://www.luogu.com.cn/problem/P6417）有向图基环树贪心应用拓扑排序由外向内
P6560 [SBCOI2020] 时光的流逝（https://www.luogu.com.cn/problem/P6560）经典反向建图拓扑排序与博弈必胜态
P8655 [蓝桥杯 2017 国 B] 发现环（https://www.luogu.com.cn/problem/P8655）使用拓扑排序计算有向基环树的环
P8943 Deception Point（https://www.luogu.com.cn/problem/P8943）经典无向图基环树博弈


==================================AtCoder=================================
F - Well-defined Path Queries on a Namori（https://atcoder.jp/contests/abc266/）（无向图的内向基环树，求简单路径的树枝连通）

==================================AcWing=================================
3696. 构造有向无环图（https://www.acwing.com/problem/content/description/3699/）经典bfs序即拓扑序与DAG构造
3828. 行走路径（https://www.acwing.com/problem/content/description/3831/）有向图DAG拓扑排序DP模板题并判断有无环
4626. 最小移动距离（https://www.acwing.com/problem/content/description/4629/）有向图内向基环树判断每个环的大小

参考：OI WiKi（xx）
"""

import unittest

from typing import List, Optional
from collections import defaultdict, deque

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

    @staticmethod
    def count_dag_path(n, edges):
        # 模板: 计算有向无环连通图路径条数
        edge = [[] for _ in range(n)]
        degree = [0] * n
        for i, j in edges:
            edge[i].append(j)
            degree[j] += 1
        cnt = [0] * n
        stack = [i for i in range(n) if not degree[i]]
        for x in stack:
            cnt[x] = 1
        while stack:  # 也可以使用深搜
            nex = []
            for i in stack:
                for j in edge[i]:
                    degree[j] -= 1
                    cnt[j] += cnt[i]
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return cnt


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
        return all(x==0 for x in degree)

    @staticmethod
    def bfs_topologic_order(n, dct, degree):
        # 拓扑排序判断有向图是否存在环，同时记录节点的拓扑顺序
        order = [0] * n
        stack = [i for i in range(n) if degree[i] == 0]
        ind = 0
        while stack:
            nex = []
            for i in stack:
                order[i] = ind
                ind += 1
                for j in dct[i]:
                    degree[j] -= 1
                    if degree[j] == 0:
                        nex.append(j)
            stack = nex[:]
        if any(d > 0 for d in degree):
            return []
        return order


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def __init__(self):
        return

    @staticmethod
    def ac_3696(ac=FastIO()):
        # 模板：经典bfs序即拓扑序与DAG构造
        for _ in range(ac.read_int()):
            def check():
                n, m = ac.read_ints()
                dct = [[] for _ in range(n)]
                degree = [0] * n
                edges = []
                ans = []
                for _ in range(m):
                    t, x, y = ac.read_ints()
                    x -= 1
                    y -= 1
                    if t == 1:
                        ans.append([x, y])
                        dct[x].append(y)
                        degree[y] += 1
                    else:
                        edges.append([x, y])

                # 拓扑排序判断有向图是否存在环，同时记录节点的拓扑顺序
                order = [0] * n
                stack = [i for i in range(n) if degree[i] == 0]
                ind = 0
                while stack:
                    nex = []
                    for i in stack:
                        order[i] = ind
                        ind += 1
                        for j in dct[i]:
                            degree[j] -= 1
                            if degree[j] == 0:
                                nex.append(j)
                    stack = nex[:]
                if any(degree[x] > 0 for x in range(n)):
                    ac.st("NO")
                    return

                # 按照拓扑序依次给与方向
                ac.st("YES")
                for x, y in edges:
                    if order[x] < order[y]:
                        ac.lst([x + 1, y + 1])
                    else:
                        ac.lst([y + 1, x + 1])
                for x, y in ans:
                    ac.lst([x + 1, y + 1])
                return

            check()
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
            # 每次都判断结果
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
            # 稳定的拓扑排序
            if unique and len(res) == n:
                s = "".join(res)
                return True, f"Sorted sequence determined after {x} relations: {s}."
            # 存在环
            if len(res) < m:
                return True, f"Inconsistency found after {x} relations."
            # 不稳定的拓扑排序
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
                # 更新 i 到 j 的路径条数与相应耗时
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

    @staticmethod
    def lg_p6255(ac=FastIO()):
        # 模板：简单无向图并查集计算连通块后使用拓扑排序寻找环的信息
        n, m = ac.read_ints()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        uf = UnionFind(n)
        for _ in range(m):  # 简单无向图即没有环套环
            i, j = ac.read_ints_minus_one()
            degree[j] += 1
            degree[i] += 1
            dct[i].append(j)
            dct[j].append(i)
            uf.union(i, j)
        # 计算连通块
        part = uf.get_root_part()
        ans = []
        for p in part:
            lst = part[p]
            # 拓扑排序找环
            nodes = [i for i in lst if degree[i] == 1]
            stack = nodes[:]
            visit = set()
            cnt = 0
            while stack:
                cnt += len(stack)
                for i in stack:
                    visit.add(i)
                nex = []
                for i in stack:
                    for j in dct[i]:
                        degree[j] -= 1
                        if degree[j] == 1:
                            nex.append(j)
                stack = nex[:]
            if cnt == len(part[p]):
                # 没有环则所有外围点出发的边都是死路
                for i in nodes:
                    for j in dct[i]:
                        ans.append([i + 1, j + 1])
            else:
                # 有环则所有环上的点
                for i in lst:
                    if i not in visit:
                        for j in dct[i]:
                            if j in visit:
                                ans.append([i + 1, j + 1])
        ans.sort()
        ac.st(len(ans))
        for a in ans:
            ac.lst(a)
        return

    @staticmethod
    def lg_p6417(ac=FastIO()):
        # 模板：有向图基环树贪心应用拓扑排序由外向内
        n = ac.read_int()
        dct = [ac.read_int() - 1 for _ in range(n)]
        degree = [0] * n
        for i in range(n):
            degree[dct[i]] += 1
        # 外层直接作为坏蛋
        stack = [x for x in range(n) if not degree[x]]
        visit = [-1] * n
        ans = len(stack)
        for i in stack:
            visit[i] = 1
        while stack:
            nex = []
            for i in stack:
                degree[dct[i]] -= 1
                # 确定坏蛋或者是平民的角色后根据度进行角色指定
                if (not degree[dct[i]] or visit[i] == 1) and visit[dct[i]] == -1:
                    if visit[i] == 1:
                        # 父亲是坏蛋必然是平民
                        visit[dct[i]] = 0
                    else:
                        # 入度为 0 优先指定为坏蛋
                        ans += 1
                        visit[dct[i]] = 1
                    nex.append(dct[i])
            stack = nex[:]

        for i in range(n):
            x = 0
            # 计算剩余环的大小
            while visit[i] == -1:
                visit[i] = 1
                x += 1
                i = dct[i]
            # 环内的坏蛋最多个数
            ans += x // 2
        ac.st(ans)
        return

    @staticmethod
    def lg_p6560(ac=FastIO()):
        # 模板：经典反向建图拓扑排序与博弈必胜态
        n, m, q = ac.read_ints()
        dct = [[] for _ in range(n)]
        degree = [[0, -1, 0] for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_ints_minus_one()
            dct[j].append(i)
            degree[i][0] += 1

        visit = [[-1, 0] for _ in range(n)]
        for ind in range(q):
            s, e = ac.read_ints_minus_one()
            visit[s] = [ind, 0]
            stack = deque([x for x in range(n) if not degree[x][0] or x == e])
            for i in stack:
                visit[i] = [ind, -1]
            while stack and not visit[s][1]:
                i = stack.popleft()
                for j in dct[i]:
                    if visit[j][0] != ind:
                        visit[j] = [ind, 0]
                    if degree[j][1] != ind:
                        degree[j][1] = ind
                        degree[j][2] = degree[j][0]
                    if visit[j][1]:
                        continue
                    degree[j][2] -= 1
                    if visit[i][1] == -1:
                        visit[j][1] = 1
                        stack.append(j)
                    elif not degree[j][2] and visit[i][1] == 1:
                        visit[j][1] = -1
                        stack.append(j)
            ac.st(visit[s][1])
        return

    @staticmethod
    def lg_p8655(ac=FastIO()):
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        degree = [0]*n
        for _ in range(n):
            i, j = ac.read_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
            degree[i] += 1
            degree[j] += 1
        stack = [x for x in range(n) if degree[x]==1]
        while stack:
            y = stack.pop()
            for z in dct[y]:
                degree[z] -= 1
                if degree[z] == 1:
                    stack.append(z)
        ans = [x+1 for x in range(n) if degree[x] == 2]
        ac.lst(ans)
        return

    @staticmethod
    def lg_p8943(ac=FastIO()):
        # 模板：经典无向图基环树博弈
        n, q = ac.read_ints()
        degree = [0] * n
        dct = [[] for _ in range(n)]
        for _ in range(n):
            x, y = ac.read_ints_minus_one()
            dct[x].append(y)
            dct[y].append(x)
            degree[x] += 1
            degree[y] += 1

        # 找出环
        stack = deque([i for i in range(n) if degree[i] == 1])
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                degree[j] -= 1
                if degree[j] == 1:
                    stack.append(j)

        dis = [inf] * n
        stack = [i for i in range(n) if degree[i] > 1]
        path = [[stack[0], -1]]
        while True:
            for j in dct[path[-1][0]]:
                if degree[j] > 1 and j != path[-1][1]:
                    path.append([j, path[-1][0]])
                    break
            if path[-1][0] == path[0][0]:
                path.pop()
                break
        path = [p for p, _ in path]
        ind = {num: i for i, num in enumerate(path)}

        # 计算每个点到环上的祖先节点与距离
        parent = [-1] * n
        for i in stack:
            parent[i] = i
        d = 0
        while stack:
            nex = []
            for i in stack:
                dis[i] = d
            for i in stack:
                for j in dct[i]:
                    if dis[j] == inf:
                        parent[j] = parent[i]
                        nex.append(j)
            stack = nex[:]
            d += 1

        for _ in range(q):
            x, y = ac.read_ints_minus_one()
            if dis[x] == 0:
                ac.st("Survive" if x != y else "Deception")
                continue

            a = parent[x]
            b = parent[y]
            # 只有先到达环才可能幸免
            dis_y = dis[y] + ac.min(len(path) - abs(ind[a] - ind[b]), abs(ind[a] - ind[b]))
            if dis[x] < dis_y:
                ac.st("Survive")
            else:
                ac.st("Deception")
        return

    @staticmethod
    def lc_2204(n: int, edges: List[List[int]]) -> List[int]:
        # 模板：无向图拓扑排序
        dct = [[] for _ in range(n)]
        degree = [0]*n
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)
            degree[i] += 1
            degree[j] += 1
        stack = deque([i for i in range(n) if degree[i] == 1])
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                degree[j] -= 1
                if degree[j] == 1:
                    stack.append(j)

        circle = deque([i for i in range(n) if degree[i] > 1])
        ans = [-1]*n
        for i in circle:
            ans[i] = 0
        while circle:
            i = circle.popleft()
            for j in dct[i]:
                if ans[j] == -1:
                    ans[j] = ans[i] + 1
                    circle.append(j)
        return ans

    @staticmethod
    def lc_1857(colors: str, edges: List[List[int]]) -> int:

        # 模板：经典拓扑排序DP
        n = len(colors)
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for i, j in edges:
            if i == j:
                return -1
            dct[i].append(j)
            degree[j] += 1

        cnt = [[0] * 26 for _ in range(n)]
        stack = deque([i for i in range(n) if not degree[i]])
        for i in stack:
            cnt[i][ord(colors[i]) - ord("a")] += 1
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                degree[j] -= 1
                for c in range(26):
                    a, b = cnt[j][c], cnt[i][c]
                    cnt[j][c] = a if a > b else b
                if not degree[j]:
                    cnt[j][ord(colors[j]) - ord("a")] += 1
                    stack.append(j)
        if not all(x == 0 for x in degree):
            return -1
        return max(max(c) for c in cnt)

    @staticmethod
    def lc_1932(trees: List[TreeNode]) -> Optional[TreeNode]:
        # 模板：经典连通性、拓扑排序与二叉搜索树判断

        nodes = set()
        dct = defaultdict(list)
        degree = defaultdict(int)
        for root in trees:

            def dfs(node):
                if not node:
                    return
                nodes.add(node.val)
                if node.left:
                    x = node.left.val
                    dct[node.val].append(x)
                    degree[x] += 1
                    dfs(node.left)
                if node.right:
                    x = node.right.val
                    dct[node.val].append(x)
                    degree[x] += 1
                    dfs(node.right)
                return

            dfs(root)

        nodes = list(nodes)
        m = len(nodes)
        ind = {num: i for i, num in enumerate(nodes)}

        # 连通性
        uf = UnionFind(m)
        for x in dct:
            for y in dct[x]:
                uf.union(ind[x], ind[y])
        if uf.part != 1:
            return

        # 二叉性与拓扑排序唯一根
        for num in nodes:
            if len(dct[num]) > 2:
                return
        stack = [num for num in nodes if not degree[num]]
        if len(stack) != 1:
            return
        r = stack[0]
        while stack:
            nex = []
            for x in stack:
                for y in dct[x]:
                    degree[y] -= 1
                    if not degree[y]:
                        nex.append(y)
            stack = nex[:]
        if not all(degree[x] == 0 for x in nodes):
            return

        # 二叉搜索特性

        def dfs(x, floor, ceil):
            nonlocal ans
            if not ans:
                return
            if not floor < x < ceil:
                ans = False
                return
            node = TreeNode(x)
            for y in dct[x]:
                if y < x:
                    node.left = dfs(y, floor, x)
                else:
                    node.right = dfs(y, x, ceil)
            return node

        ans = True
        root = dfs(r, -inf, inf)

        if ans:
            return root
        return

    @staticmethod
    def ac_3828(ac=FastIO()):
        # 模板：有向图DAG拓扑排序DP模板题并判断有无环
        m, n = ac.read_ints()
        ind = {w: i for i, w in enumerate("QWER")}
        grid = [ac.read_str() for _ in range(m)]

        # 建图
        dct = [[] for _ in range(m * n)]
        degree = [0] * (m * n)
        for i in range(m):
            for j in range(n):
                x = ind[grid[i][j]]
                for a, b in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                    if 0 <= a < m and 0 <= b < n:
                        y = ind[grid[a][b]]
                        if y == (x + 1) % 4:
                            dct[i * n + j].append(a * n + b)
                            degree[a * n + b] += 1

        # 拓扑排序DP
        pre = [0] * (m * n)
        stack = [i for i in range(m * n) if not degree[i]]
        for i in stack:
            if grid[i // n][i % n] == "Q":
                pre[i] = 1
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if pre[i] > pre[j]:
                        pre[j] = pre[i]
                    if not degree[j]:
                        if not pre[j]:
                            if grid[j // n][j % n] == "Q":
                                pre[j] = 1
                        else:
                            pre[j] += 1
                        nex.append(j)
            stack = nex[:]
        if any(d > 0 for d in degree):
            ac.st("infinity")
            return
        ans = max(x // 4 for x in pre)
        ac.st(ans if ans else "none")
        return

    @staticmethod
    def ac_4626(ac=FastIO()):
        # 模板：有向图内向基环树判断每个环的大小
        n = ac.read_int()
        a = ac.read_list_ints_minus_one()
        dct = [[] for _ in range(n)]
        degree = [0]*n
        for i in range(n):
            dct[i].append(a[i])
            degree[a[i]] += 1
        # 全是环上的点才行
        if any(d == 0 for d in degree):
            ac.st(-1)
            return
        # 使用BFS判断每个环的大小并累计结果
        ans = 1
        for i in range(n):
            if degree[i] == 0:
                continue
            stack = [i]
            degree[i] = 0
            cur = 1
            while stack:
                x = stack.pop()
                for j in dct[x]:
                    if degree[j] > 0:
                        degree[j] = 0
                        cur += 1
                        stack.append(j)
            if cur % 2:
                ans = math.lcm(cur, ans)
            else:
                ans = math.lcm(cur//2, ans)
        ac.st(ans)
        return

    @staticmethod
    def lc_1591(grid: List[List[int]]) -> bool:
        # 模板：经典建图判断拓扑排序是否无环
        color = defaultdict(list)
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                color[grid[i][j]].append([i, j])

        pos = defaultdict(list)
        for c in color:
            lst = color[c]
            x1 = min(x for x, _ in lst)
            x2 = max(x for x, _ in lst)

            y1 = min(x for _, x in lst)
            y2 = max(x for _, x in lst)
            pos[c] = [x1, x2, y1, y2]

        degree = defaultdict(int)
        dct = defaultdict(list)
        for x in pos:
            a1, a2, b1, b2 = pos[x]
            for y in pos:
                if x != y:
                    for a, b in color[y]:
                        if a1 <= a <= a2 and b1 <= b <= b2:
                            dct[x].append(y)
                            degree[y] += 1
        stack = [x for x in pos if not degree[x]]
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex[:]
        return all(degree[x] == 0 for x in pos)


class TestGeneral(unittest.TestCase):

    def test_topological_sort(self):
        ts = TopologicalSort()
        n = 5
        edges = [[0, 1], [0, 2], [1, 4], [2, 3], [3, 4]]
        assert ts.get_rank(n, edges) == [0, 1, 1, 2, 3]
        return


if __name__ == '__main__':
    unittest.main()
