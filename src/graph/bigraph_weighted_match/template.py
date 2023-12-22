from collections import defaultdict, deque
from src.utils.fast_io import inf

import numpy as np


class BipartiteMatching:
    def __init__(self, n, m):
        self._n = n
        self._m = m
        self._to = [[] for _ in range(n)]

    def add_edge(self, a, b):
        self._to[a].append(b)

    def solve(self):
        n, m, to = self._n, self._m, self._to
        prev = [-1] * n
        root = [-1] * n
        p = [-1] * n
        q = [-1] * m
        updated = True
        while updated:
            updated = False
            s = []
            s_front = 0
            for i in range(n):
                if p[i] == -1:
                    root[i] = i
                    s.append(i)
            while s_front < len(s):
                v = s[s_front]
                s_front += 1
                if p[root[v]] != -1:
                    continue
                for u in to[v]:
                    if q[u] == -1:
                        while u != -1:
                            q[u] = v
                            p[v], u = u, p[v]
                            v = prev[v]
                        updated = True
                        break
                    u = q[u]
                    if prev[u] != -1:
                        continue
                    prev[u] = v
                    root[u] = root[v]
                    s.append(u)
            if updated:
                for i in range(n):
                    prev[i] = -1
                    root[i] = -1
        return [(v, p[v]) for v in range(n) if p[v] != -1]


class Hungarian:
    def __init__(self):
        # Bipartite graph maximum math without weight
        return

    @staticmethod
    def dfs_recursion(n, m, dct):
        assert len(dct) == m

        def hungarian(i):
            for j in dct[i]:
                if not visit[j]:
                    visit[j] = True
                    if match[j] == -1 or hungarian(match[j]):
                        match[j] = i
                        return True
            return False

        # left group size is n
        match = [-1] * n
        ans = 0
        for x in range(m):
            # right group size is m
            visit = [False] * n
            if hungarian(x):
                ans += 1
        return ans

    @staticmethod
    def bfs_iteration(n, m, dct):

        assert len(dct) == m

        match = [-1] * n
        ans = 0
        for i in range(m):
            hungarian = [0] * m
            visit = [0] * n
            stack = [[i, 0]]
            while stack:
                x, ind = stack[-1]
                if ind == len(dct[x]) or hungarian[x]:
                    stack.pop()
                    continue
                y = dct[x][ind]
                if not visit[y]:
                    visit[y] = 1
                    if match[y] == -1:
                        match[y] = x
                        hungarian[x] = 1
                    else:
                        stack.append([match[y], 0])
                else:
                    if hungarian[match[y]]:
                        match[y] = x
                        hungarian[x] = 1
                    stack[-1][1] += 1
            if hungarian[i]:
                ans += 1
        return ans


class EK:

    def __init__(self, n, m, s, t):
        self.flow = [0] * (n + 10)
        self.pre = [0] * (n + 10)
        self.used = set()
        self.g = defaultdict(list)
        self.edges_val = defaultdict(int)
        self.m = m
        self.s = s
        self.t = t
        self.res = 0

    def add_edge(self, from_node, to, flow):
        self.edges_val[(from_node, to)] += flow
        self.edges_val[(to, from_node)] += 0
        self.g[from_node].append(to)
        self.g[to].append(from_node)

    def bfs(self) -> bool:
        self.used.clear()
        q = deque()
        q.append(self.s)
        self.used.add(self.s)
        self.flow[self.s] = inf
        while q:
            now = q.popleft()
            for nxt in self.g[now]:
                edge = (now, nxt)
                val = self.edges_val[edge]
                if nxt not in self.used and val:
                    self.used.add(nxt)
                    self.flow[nxt] = min(self.flow[now], val)
                    self.pre[nxt] = now
                    if nxt == self.t:
                        return True
                    q.append(nxt)
        return False

    def pipline(self) -> int:
        while self.bfs():
            self.res += self.flow[self.t]
            from_node = self.t
            to = self.pre[from_node]
            while True:
                edge = (from_node, to)
                reverse_edge = (to, from_node)
                self.edges_val[edge] += self.flow[self.t]
                self.edges_val[reverse_edge] -= self.flow[self.t]
                if to == self.s:
                    break
                from_node = to
                to = self.pre[from_node]
        return self.res


class KM:
    def __init__(self):
        self.matrix = None
        self.max_weight = 0
        self.row, self.col = 0, 0
        self.size = 0
        self.lx = None
        self.ly = None
        self.match = None
        self.slack = None
        self.visit_x = None
        self.visit_y = None

        # 调整数据

    def pad_matrix(self, min):
        if min:
            max = self.matrix.max() + 1
            self.matrix = max - self.matrix

        if self.row > self.col:
            self.matrix = np.c_[self.matrix, np.array([[0] * (self.row - self.col)] * self.row)]
        elif self.col > self.row:
            self.matrix = np.r_[self.matrix, np.array([[0] * self.col] * (self.col - self.row))]

    def reset_slack(self):
        self.slack.fill(self.max_weight + 1)

    def reset_vis(self):
        self.visit_x.fill(False)
        self.visit_y.fill(False)

    def find_path(self, x):
        self.visit_x[x] = True
        for y in range(self.size):
            if self.visit_y[y]:
                continue
            tmp_delta = self.lx[x] + self.ly[y] - self.matrix[x][y]
            if tmp_delta == 0:
                self.visit_y[y] = True
                if self.match[y] == -1 or self.find_path(self.match[y]):
                    self.match[y] = x
                    return True
            elif self.slack[y] > tmp_delta:
                self.slack[y] = tmp_delta

        return False

    def km_cal(self):
        for x in range(self.size):
            self.reset_slack()
            while True:
                self.reset_vis()
                if self.find_path(x):
                    break
                else:  # update slack
                    delta = self.slack[~self.visit_y].min()
                    self.lx[self.visit_x] -= delta
                    self.ly[self.visit_y] += delta
                    self.slack[~self.visit_y] -= delta

    def compute(self, datas, min=False):
        """default compute maximum match"""
        self.matrix = np.array(datas) if not isinstance(datas, np.ndarray) else datas
        self.max_weight = self.matrix.sum()
        self.row, self.col = self.matrix.shape
        self.size = max(self.row, self.col)
        self.pad_matrix(min)
        self.lx = self.matrix.max(1)
        self.ly = np.array([0] * self.size, dtype=int)
        self.match = np.array([-1] * self.size, dtype=int)
        self.slack = np.array([0] * self.size, dtype=int)
        self.visit_x = np.array([False] * self.size, dtype=bool)
        self.visit_y = np.array([False] * self.size, dtype=bool)

        self.km_cal()

        match = [i[0] for i in sorted(enumerate(self.match), key=lambda x: x[1])]
        result = []
        for i in range(self.row):
            result.append((i, match[i] if match[i] < self.col else -1))
        return result
