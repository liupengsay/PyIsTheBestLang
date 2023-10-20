"""

算法：强连通分量、2-SAT、最大环、最小环
功能：用来求解有向图的强连通分量，可以将一张图的每个强连通分量都缩成一个点，然后这张图会变成一个 DAG，可以进行拓扑排序以及更多其他操作
定义：有向图 G 强连通是指 G 中任意两个结点连通，强连通分量（Strongly Connected Components，SCC）是极大的强连通子图
距离：求一条路径，可以经过重复结点，要求经过的不同结点数量最多
2-SAT：简单的说就是给出 n 个集合，每个集合有两个元素，已知若干个 <a,b>，表示 a 与 b 矛盾（其中 a 与 b 属于不同的集合）。然后从每个集合选择一个元素，判断能否一共选 n 个两两不矛盾的元素。显然可能有多种选择方案，一般题中只需要求出一种即可。
题目：

===================================力扣===================================
2360. 图中的最长环（https://leetcode.cn/problems/longest-cycle-in-a-graph/）求最长的环长度（有向图scc、内向基环树没有环套环，N个节点N条边，也可以使用拓扑排序）

===================================洛谷===================================
P3387 【模板】缩点 （https://www.luogu.com.cn/problem/solution/P3387）允许多次经过点和边求一条路径最大权值和、强连通分量
P2661 [NOIP2015 提高组] 信息传递（https://www.luogu.com.cn/problem/P2661）求最小的环长度（有向图、内向基环树没有环套环，N个节点N条边，也可以使用拓扑排序）
P4089 [USACO17DEC]The Bovine Shuffle S（https://www.luogu.com.cn/problem/P4089）求所有环的长度和，注意自环
P5145 漂浮的鸭子（https://www.luogu.com.cn/problem/P5145）内向基环树求最大权值和的环

P4782 【模板】2-SAT 问题（https://www.luogu.com.cn/problem/P4782）2-SAT 问题模板题
P5782 [POI2001] 和平委员会（https://www.luogu.com.cn/problem/P5782）2-SAT 问题模板题
P4171 [JSOI2010] 满汉全席（https://www.luogu.com.cn/problem/P4171）2-SAT 问题模板题
================================CodeForces================================
C. Engineer Artem（https://codeforces.com/problemset/problem/1438/C）2-SAT 问题模板题


参考：OI WiKi（https://oi-wiki.org/graph/scc/）
"""

from collections import deque

from graph.scc.template import Tarjan
from utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p3387(ac=FastIO()):
        # 模板：有向图使用强连通分量将环进行缩点后求最长路
        n, m = ac.read_list_ints()
        weight = ac.read_list_ints()
        edge = [set() for _ in range(n)]
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            edge[x].add(y)
        edge = [list(e) for e in edge]

        # 求得强连通分量后进行重新建图，这里也可以使用 Kosaraju 算法
        tarjan = Tarjan(edge)
        ind = [-1] * n
        m = len(tarjan.scc)
        point = [0] * m
        degree = [0] * m
        dct = [[] for _ in range(n)]
        for i, ls in enumerate(tarjan.scc):
            for j in ls:
                ind[j] = i
                point[i] += weight[j]
        for i in range(n):
            for j in edge[i]:
                u, v = ind[i], ind[j]
                if u != v:
                    dct[u].append(v)
        for i in range(m):
            for j in dct[i]:
                degree[j] += 1

        # 拓扑排序求最长路，这里也可以使用深搜
        visit = [0] * m
        stack = deque([i for i in range(m) if not degree[i]])
        for i in stack:
            visit[i] = point[i]
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                w = point[j]
                degree[j] -= 1
                if visit[i] + w > visit[j]:
                    visit[j] = visit[i] + w
                if not degree[j]:
                    stack.append(j)
        ac.st(max(visit))
        return

    @staticmethod
    def lc_2360(edges):

        # 模板: 求内向基环树的最大权值和环 edge表示有向边 i 到 edge[i] 而 dct表示对应的边权值
        def largest_circle(n, edge, dct):

            def dfs(x, sum_):
                nonlocal ans
                if x == st:
                    ans = ans if ans > sum_ else sum_
                    return
                # 访问过
                if a[x] or b[x]:
                    return
                a[x] = 1
                dfs(edge[x], sum_ + dct[x])
                a[x] = 0
                return

            a = [0] * n
            b = [0] * n
            ans = 0
            for st in range(n):
                dfs(edge[st], dct[st])
                b[st] = 1
            return ans

        # 经典题目也可用 scc 或者拓扑排序求解
        return largest_circle(len(edges), edges, [1] * len(edges))
