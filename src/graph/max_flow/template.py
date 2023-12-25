from collections import deque

from src.utils.fast_io import inf


class DinicMaxflow:
    def __init__(self, n):
        self.n = n
        self.G = [[] for _ in range(n)]
        self.depth = [0] * n
        self.cur = [0] * n

    def add_edge(self, u, v, cap):
        self.G[u].append([v, cap, len(self.G[v])])
        self.G[v].append([u, 0, len(self.G[u]) - 1])

    def bfs(self, s, t):
        self.depth = [-1] * self.n
        self.depth[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v, cap, _ in self.G[u]:
                if self.depth[v] == -1 and cap > 0:
                    self.depth[v] = self.depth[u] + 1
                    q.append(v)
        return self.depth[t] != -1

    def dfs(self, s, t, ff=inf):
        stack = [(s, -1, ff)]
        visit = {(s, -1, ff): 0}
        while stack:
            u, fa, f = stack[-1]
            if u == t:
                visit[(u, fa, f)] = f
                stack.pop()
                continue
            flag = 0
            while self.cur[u] < len(self.G[u]):
                v, cap, rev = self.G[u][self.cur[u]]
                if cap > 0 and self.depth[v] == self.depth[u] + 1:
                    x, y = f - visit[(u, fa, f)], cap
                    x = x if x < y else y
                    if (v, u, x) not in visit:
                        stack.append((v, u, x))
                        flag = 1
                        visit[(v, u, x)] = 0
                        break
                    else:
                        df = visit[(v, u, x)]
                        if df > 0:
                            self.G[u][self.cur[u]][1] -= df
                            self.G[v][rev][1] += df
                            visit[(u, fa, f)] += df
                            if visit[(u, fa, f)] == f:
                                break
                self.cur[u] += 1
            if flag:
                continue
            stack.pop()
        return visit[(s, -1, ff)]

    def max_flow(self, s, t):
        total_flow = 0
        while self.bfs(s, t):
            self.cur = [0] * self.n
            flow = self.dfs(s, t)
            while flow > 0:
                total_flow += flow
                flow = self.dfs(s, t)
        return total_flow
