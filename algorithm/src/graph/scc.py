
import unittest

from collections import defaultdict

"""

算法：强连通分量、2-SAT、最大环、最小环
功能：用来求解有向图的强连通分量，可以将一张图的每个强连通分量都缩成一个点，然后这张图会变成一个 DAG，可以进行拓扑排序以及更多其他操作
定义：有向图 G 强连通是指 G 中任意两个结点连通，强连通分量（Strongly Connected Components，SCC）是极大的强连通子图
距离：求一条路径，可以经过重复结点，要求经过的不同结点数量最多

题目：

===================================力扣===================================
2360 图中的最长环（https://leetcode.cn/problems/longest-cycle-in-a-graph/）求最长的环长度（有向图、内向基环树没有环套环，N个节点N条边，也可以使用拓扑排序）

===================================洛谷===================================
P3387 缩点 （https://www.luogu.com.cn/problem/solution/P3387）允许多次经过点和边求一条路径最大权值和
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


class Kosaraju:
    def __init__(self, n, g):
        self.n = n
        self.g = g
        self.g2 = [list() for _ in range(self.n)]
        self.vis = [False] * n
        self.s = list()
        self.color = [0] * n
        self.sccCnt = 0
        self.gen_reverse_graph()
        self.kosaraju()

    def gen_reverse_graph(self):
        for i in range(self.n):
            for j in self.g[i]:
                self.g2[j].append(i)
        return

    def dfs1(self, u):
        self.vis[u] = True
        for v in self.g[u]:
            if not self.vis[v]:
                self.dfs1(v)
        self.s.append(u)
        return

    def dfs2(self, u):
        self.color[u] = self.sccCnt
        for v in self.g2[u]:
            if not self.color[v]:
                self.dfs2(v)
        return

    def kosaraju(self):
        for i in range(self.n):
            if not self.vis[i]:
                self.dfs1(i)
        for i in range(self.n - 1, -1, -1):
            if not self.color[self.s[i]]:
                self.sccCnt += 1
                self.dfs2(self.s[i])
        self.color = [c-1 for c in self.color]
        return

    def gen_new_edges(self, weight):
        color = defaultdict(list)
        dct = dict()
        for i in range(self.n):
            j = self.color[i]
            dct[i] = j
            color[j].append(i)
        k = len(color)
        new_weight = [sum(weight[i] for i in color[j]) for j in range(k)]

        new_edges = [set() for _ in range(k)]
        for i in range(self.n):
            for j in self.g[i]:
                if dct[i] != dct[j]:
                    new_edges[dct[i]].add(dct[j])
        return new_weight, new_edges


class Tarjan:
    def __init__(self, edge):
        self.edge = edge
        self.n = len(edge)
        self.dfn = [0] * self.n
        self.low = [0] * self.n
        self.visit = [0] * self.n
        self.stamp = 0
        self.visit = [0]*self.n
        self.stack = []
        self.scc = []
        for i in range(self.n):
            if not self.visit[i]:
                self.tarjan(i)

    def tarjan(self, u):
        self.dfn[u], self.low[u] = self.stamp, self.stamp
        self.stamp += 1
        self.stack.append(u)
        self.visit[u] = 1
        for v in self.edge[u]:
            if not self.visit[v]:  # 未访问
                self.tarjan(v)
                self.low[u] = min(self.low[u], self.low[v])
            elif self.visit[v] == 1:
                self.low[u] = min(self.low[u], self.dfn[v])

        if self.dfn[u] == self.low[u]:
            cur = []
            # 栈中u之后的元素是一个完整的强连通分量
            while True:
                cur.append(self.stack.pop())
                self.visit[cur[-1]] = 2  # 节点已弹出，归属于现有强连通分量
                if cur[-1] == u:
                    break
            self.scc.append(cur)
        return


class InwardBaseTree:
    def __init__(self):
        return

    @staticmethod
    def largest_circle(n, edge, dct):
        # 模板: 求内向基环树的最大权值和环 edge表示有向边 i=>edge[i] 而 dct表示对应的边权值

        def dfs(x, sum_):
            nonlocal ans
            if x == st:
                ans = ans if ans > sum_ else sum_
                return
            if a[x] or b[x]:
                return
            a[x] = 1
            dfs(edge[x], sum_+dct[x])
            a[x] = 0
            return

        a = [0]*n
        b = [0]*n
        ans = 0
        for st in range(n):
            dfs(edge[st], dct[st])
            b[st] = 1
        return ans


class TwoSAT:
    def __init__(self):
        return

    @staticmethod
    def cf_1438c(m, n, grid):
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
        ans = [0] * m*n
        pre = set()
        for sc in tarjan.scc:
            for node in sc:
                i = node // 2
                if i not in pre:
                    ans[i] = node % 2
                pre.add(i)

        for x in range(m * n):
            grid[x // n][x % n] += ans[x]

        #####################################################
        # Kosaraju算法
        kosaraju = Kosaraju(2 * m * n, edge)
        # 注意是小于符号
        ans = [int(kosaraju.color[2 * i] < kosaraju.color[2 * i + 1]) for i in range(m * n)]
        for x in range(m * n):
            grid[x // n][x % n] += ans[x]
        return grid

    @staticmethod
    def luogu_4782(n, pairs):
        # 建图并把索引编码
        edge = [[] for _ in range(2 * n)]
        for i, a, j, b in pairs:
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
                    # ac.st("IMPOSSIBLE")
                    return
                pre.add(node // 2)

        # 进行方案赋予，先出现的确定值
        # ac.st("POSSIBLE")
        ans = [0] * n
        pre = set()
        for sc in tarjan.scc:
            for node in sc:
                i = node // 2
                if i not in pre:
                    ans[i] = node % 2
                pre.add(i)
        # ac.lst(ans)

        #####################################################
        # Kosaraju算法（与上面的算法二选一）
        kosaraju = Kosaraju(2 * n, edge)
        for i in range(n):
            if kosaraju.color[i * 2] == kosaraju.color[i * 2 + 1]:
                # ac.st("IMPOSSIBLE")
                return

        # ac.st("POSSIBLE")
        ans = [int(kosaraju.color[2 * i] < kosaraju.color[2 * i + 1])
               for i in range(n)]
        # ac.lst(ans)
        return

    @staticmethod
    def luogu_p5782(n, pairs):

        # 建图并把索引编码
        edge = [[] for _ in range(4 * n)]
        for a, b in pairs:
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
                    # ac.st("NIE")
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
        res = [i+1 for i in range(2 * n) if ans[i]]

        #####################################################
        # Kosaraju算法
        kosaraju = Kosaraju(4 * n, edge)
        for i in range(2 * n):
            if kosaraju.color[i * 2] == kosaraju.color[i * 2 + 1]:
                # ac.st("NIE")
                return
        # 注意是小于符号
        ans = [int(kosaraju.color[2 * i] < kosaraju.color[2 * i + 1]) for i in range(2 * n)]
        res = [i + 1 for i in range(2 * n) if ans[i]]
        return

    @staticmethod
    def luogu_4171(n, pairs):
        # 建图并把索引编码
        edge = [[] for _ in range(2 * n)]
        for lst in pairs:
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
        # ac.st("GOOD" if ans else "BAD")

        #####################################################
        # Kosaraju算法
        kosaraju = Kosaraju(2 * n, edge)
        ans = True
        for i in range(n):
            if kosaraju.color[i * 2] == kosaraju.color[i * 2 + 1]:
                ans = False
                break
        # ac.st("GOOD" if ans else "BAD")
        return


class TestGeneral(unittest.TestCase):

    def test_directed_graph(self):
        # 有向无环图
        edge = [[1, 2], [], [3], []]
        n = 4
        kosaraju = Kosaraju(n, edge)
        assert len(set(kosaraju.color)) == 4
        tarjan = Tarjan(edge)
        assert len(tarjan.scc) == 4

        # 有向有环图
        edge = [[1, 2], [2], [0, 3], []]
        n = 4
        kosaraju = Kosaraju(n, edge)
        assert len(set(kosaraju.color)) == 2
        tarjan = Tarjan(edge)
        assert len(tarjan.scc) == 2
        return

    def test_undirected_graph(self):
        # 无向有环图
        edge = [[1, 2], [0, 2, 3], [0, 1], [1, 4], [3]]
        n = 5
        kosaraju = Kosaraju(n, edge)
        assert len(set(kosaraju.color)) == 1
        tarjan = Tarjan(edge)
        assert len(tarjan.scc) == 1
        return


if __name__ == '__main__':
    unittest.main()
