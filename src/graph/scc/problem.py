"""

算法：强连通分量、2-SAT、最大环、最小环
功能：用来求解有向图的强连通分量，可以将一张图的每个强连通分量都缩成一个点，然后这张图会变成一个 DAG，可以进行拓扑排序以及更多其他操作
定义：有向图 G 强连通是指 G 中任意两个结点连通，强连通分量（Strongly Connected Components，SCC）是极大的强连通子图
距离：求一条路径，可以经过重复结点，要求经过的不同结点数量最多
2-SAT：简单的说就是给出 n 个集合，每个集合有两个元素，已知若干个 <a,b>，表示 a 与 b 矛盾（其中 a 与 b 属于不同的集合）。然后从每个集合选择一个元素，判断能否一共选 n 个两两不矛盾的元素。显然可能有多种选择方案，一般题中只需要求出一种即可。
题目：

===================================LeetCode===================================
2360（https://leetcode.com/problems/longest-cycle-in-a-graph/）求最长的环长度（有向图scc、内向基环树没有环套环，N个节点N条边，也可以使用拓扑排序）

===================================LuoGu==================================
3387（https://www.luogu.com.cn/problem/solution/P3387）允许多次经过点和边求一条路径最大权值和、强连通分量
2661（https://www.luogu.com.cn/problem/P2661）求最小的环长度（有向图、内向基环树没有环套环，N个节点N条边，也可以使用拓扑排序）
4089（https://www.luogu.com.cn/problem/P4089）求所有环的长度和，注意自环
5145（https://www.luogu.com.cn/problem/P5145）内向基环树求最大权值和的环

4782（https://www.luogu.com.cn/problem/P4782）2-SAT 问题模板题
5782（https://www.luogu.com.cn/problem/P5782）2-SAT 问题模板题
4171（https://www.luogu.com.cn/problem/P4171）2-SAT 问题模板题
================================CodeForces================================
C. Engineer Artem（https://codeforces.com/problemset/problem/1438/C）2-SAT 问题模板题


参考：OI WiKi（https://oi-wiki.org/graph/scc/）
"""

from collections import deque

from src.graph.scc.template import Tarjan, Kosaraju
from src.utils.fast_io import FastIO


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


class TwoSAT:
    def __init__(self):
        return

    @staticmethod
    def cf_1438c(ac=FastIO()):
        for _ in range(ac.read_int()):
            m, n = ac.read_list_ints()
            grid = [ac.read_list_ints() for _ in range(m)]
            # 建图并把索引编码
            edge = [[] for _ in range(2 * m * n)]
            for i in range(m):
                for j in range(n):
                    if i + 1 < m:
                        x, y = i * n + j, i * n + n + j
                        for a, b in [[0, 0], [0, 1], [1, 0], [1, 1]]:
                            if grid[i][j] + a == grid[i + 1][j] + b:
                                edge[x * 2 + a].append(y * 2 + 1 - b)
                                edge[y * 2 + b].append(x * 2 + 1 - a)
                    if j + 1 < n:
                        x, y = i * n + j, i * n + 1 + j
                        for a, b in [[0, 0], [0, 1], [1, 0], [1, 1]]:
                            if grid[i][j] + a == grid[i][j + 1] + b:
                                edge[x * 2 + a].append(y * 2 + 1 - b)
                                edge[y * 2 + b].append(x * 2 + 1 - a)

            #####################################################
            # 按照强连通进行缩点
            tarjan = Tarjan(edge)
            # 进行方案赋予，先出现的确定值
            ans = [0] * m * n
            pre = set()
            for sc in tarjan.scc:
                for node in sc:
                    i = node // 2
                    if i not in pre:
                        ans[i] = node % 2
                    pre.add(i)

            for x in range(m * n):
                grid[x // n][x % n] += ans[x]
            for g in grid:
                ac.lst(g)
            #####################################################
            # Kosaraju算法
            kosaraju = Kosaraju(2 * m * n, edge)
            # 注意是小于符号
            ans = [int(kosaraju.color[2 * i] < kosaraju.color[2 * i + 1]) for i in range(m * n)]
            for x in range(m * n):
                grid[x // n][x % n] += ans[x]
            for g in grid:
                ac.lst(g)
            return grid

    @staticmethod
    def luogu_4782(ac=FastIO()):
        n, m = ac.read_list_ints()
        # 建图并把索引编码
        edge = [[] for _ in range(2 * n)]
        for _ in range(m):
            i, a, j, b = ac.read_list_ints()
            i -= 1
            j -= 1
            edge[i * 2 + 1 - a].append(j * 2 + b)
            edge[j * 2 + 1 - b].append(i * 2 + a)

        #####################################################
        # 按照强连通进行缩点检验是否存在冲突
        tarjan = Tarjan(edge)
        for sc in tarjan.scc:
            pre = set()
            for node in sc:
                # 条件相反的两个点不能在一个强连通分量
                if node // 2 in pre:
                    ac.st("IMPOSSIBLE")
                    return
                pre.add(node // 2)

        # 进行方案赋予，先出现的确定值
        ac.st("POSSIBLE")
        ans = [0] * n
        pre = set()
        for sc in tarjan.scc:
            for node in sc:
                i = node // 2
                if i not in pre:
                    ans[i] = node % 2
                pre.add(i)
        ac.lst(ans)

        #####################################################
        # Kosaraju算法（与上面的算法二选一）
        kosaraju = Kosaraju(2 * n, edge)
        for i in range(n):
            if kosaraju.color[i * 2] == kosaraju.color[i * 2 + 1]:
                ac.st("IMPOSSIBLE")
                return

        ac.st("POSSIBLE")
        ans = [int(kosaraju.color[2 * i] < kosaraju.color[2 * i + 1])
               for i in range(n)]
        ac.lst(ans)
        return

    @staticmethod
    def luogu_p5782(ac=FastIO()):
        n, m = ac.read_list_ints()
        # 建图并把索引编码
        edge = [[] for _ in range(4 * n)]
        for _ in range(m):
            a, b = ac.read_list_ints_minus_one()
            edge[a * 2 + 1].append(b * 2)
            edge[b * 2 + 1].append(a * 2)

        # 同一党派内只允许一个人参加
        for i in range(n):
            a, b = 2 * i, 2 * i + 1
            edge[a * 2 + 1].append(b * 2)
            edge[a * 2].append(b * 2 + 1)
            edge[b * 2 + 1].append(a * 2)
            edge[b * 2].append(a * 2 + 1)

        #####################################################
        # 按照强连通进行缩点
        tarjan = Tarjan(edge)
        for sc in tarjan.scc:
            pre = set()
            for node in sc:
                # 条件相反的两个点不能在一个强连通分量
                if node // 2 in pre:
                    ac.st("NIE")
                    return
                pre.add(node // 2)

        # 进行方案赋予，先出现的确定值
        ans = [0] * 2 * n
        pre = set()
        for sc in tarjan.scc:
            for node in sc:
                i = node // 2
                if i not in pre:
                    ans[i] = node % 2
                pre.add(i)
        res = [i + 1 for i in range(2 * n) if ans[i]]
        for a in res:
            ac.st(a)
        #####################################################
        # Kosaraju算法
        kosaraju = Kosaraju(4 * n, edge)
        for i in range(2 * n):
            if kosaraju.color[i * 2] == kosaraju.color[i * 2 + 1]:
                ac.st("NIE")
                return
        # 注意是小于符号
        ans = [int(kosaraju.color[2 * i] < kosaraju.color[2 * i + 1]) for i in range(2 * n)]
        res = [i + 1 for i in range(2 * n) if ans[i]]
        for a in res:
            ac.st(a)
        return

    @staticmethod
    def luogu_4171(ac=FastIO()):
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            # 建图并把索引编码
            edge = [[] for _ in range(2 * n)]
            for _ in range(m):
                lst = ac.read_list_strs()
                i = int(lst[0][1:]) - 1
                j = int(lst[1][1:]) - 1
                a = 1 if lst[0][0] == "h" else 0
                b = 1 if lst[1][0] == "h" else 0
                edge[i * 2 + (1 - a)].append(j * 2 + b)
                edge[j * 2 + (1 - b)].append(i * 2 + a)

            #####################################################
            # # 按照强连通进行缩点
            tarjan = Tarjan(edge)
            ans = True
            for sc in tarjan.scc:
                pre = set()
                for node in sc:
                    # 条件相反的两个点不能在一个强连通分量
                    if node // 2 in pre:
                        ans = False
                    pre.add(node // 2)
            ac.st("GOOD" if ans else "BAD")

            #####################################################
            # Kosaraju算法
            kosaraju = Kosaraju(2 * n, edge)
            ans = True
            for i in range(n):
                if kosaraju.color[i * 2] == kosaraju.color[i * 2 + 1]:
                    ans = False
                    break
            ac.st("GOOD" if ans else "BAD")
        return