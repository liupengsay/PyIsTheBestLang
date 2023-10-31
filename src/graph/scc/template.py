from collections import defaultdict

from src.utils.fast_io import FastIO


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
        self.color = [c - 1 for c in self.color]
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
            if not self.visit[v]:
                yield self.tarjan(v)
                self.low[u] = FastIO.min(self.low[u], self.low[v])
            elif self.visit[v] == 1:
                self.low[u] = FastIO.min(self.low[u], self.dfn[v])

        if self.dfn[u] == self.low[u]:
            cur = []
            # the element after u in the stack is a complete strongly connected component
            while True:
                cur.append(self.stack.pop())
                # the node has popped up and belongs to the existing strongly connected component
                self.visit[cur[-1]] = 2
                if cur[-1] == u:
                    break
            self.scc.append(cur)
        yield
