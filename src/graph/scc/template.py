
import unittest

from collections import defaultdict, deque

from utils.fast_io import FastIO



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
        self.visit = [0] * self.n
        self.stack = []
        self.scc = []
        for i in range(self.n):
            if not self.visit[i]:
                self.tarjan(i)

    @FastIO.bootstrap
    def tarjan(self, u):
        self.dfn[u], self.low[u] = self.stamp, self.stamp
        self.stamp += 1
        self.stack.append(u)
        self.visit[u] = 1
        for v in self.edge[u]:
            if not self.visit[v]:  # 未访问
                yield self.tarjan(v)
                self.low[u] = FastIO.min(self.low[u], self.low[v])
            elif self.visit[v] == 1:
                self.low[u] = FastIO.min(self.low[u], self.dfn[v])

        if self.dfn[u] == self.low[u]:
            cur = []
            # 栈中u之后的元素是一个完整的强连通分量
            while True:
                cur.append(self.stack.pop())
                self.visit[cur[-1]] = 2  # 节点已弹出，归属于现有强连通分量
                if cur[-1] == u:
                    break
            self.scc.append(cur)
        yield



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
            a, b = ac.read_ints_minus_one()
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
        res = [i+1 for i in range(2 * n) if ans[i]]
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
