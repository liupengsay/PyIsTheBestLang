import heapq
import math
from collections import deque


class DinicMaxflowMinCut:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_capacity = [0] * 2
        self.edge_to = [0] * 2
        self.edge_next = [0] * 2
        self.depth = [-1] * (self.n + 1)
        self.max_flow = 0
        self.min_cost = 0
        self.edge_id = 2
        self.cur = [0] * (self.n + 1)

    def _add_single_edge(self, u, v, cap):
        self.edge_capacity.append(cap)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_edge(self, u, v, cap):
        assert 1 <= u <= self.n
        assert 1 <= v <= self.n
        self._add_single_edge(u, v, cap)
        self._add_single_edge(v, u, 0)
        return

    def _bfs(self, s, t):
        for i in range(1, self.n + 1):
            self.depth[i] = -1
        self.depth[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            i = self.point_head[u]
            while i:
                v = self.edge_to[i]
                if self.edge_capacity[i] > 0 and self.depth[v] == -1:
                    self.depth[v] = self.depth[u] + 1
                    q.append(v)
                i = self.edge_next[i]
        return self.depth[t] != -1

    def _dfs(self, s, t, ff=math.inf):
        stack = [(s, ff, 0)]
        ind = 1
        max_flow = [0]
        sub = [-1]
        while stack:
            u, f, j = stack[-1]
            if u == t:
                max_flow[j] = f
                stack.pop()
                continue
            flag = 0
            while self.cur[u]:
                i = self.cur[u]
                v, cap, rev = self.edge_to[i], self.edge_capacity[i], i ^ 1
                if cap > 0 and self.depth[v] == self.depth[u] + 1:
                    x, y = f - max_flow[j], cap
                    x = x if x < y else y
                    if sub[j] == -1:
                        stack.append((v, x, ind))
                        max_flow.append(0)
                        sub[j] = ind
                        sub.append(-1)
                        ind += 1
                        flag = 1
                        break
                    else:
                        df = max_flow[sub[j]]
                        if df > 0:
                            self.edge_capacity[i] -= df
                            self.edge_capacity[i ^ 1] += df
                            max_flow[j] += df
                            if max_flow[j] == f:
                                break
                        sub[j] = -1

                self.cur[u] = self.edge_next[i]
            if flag:
                continue
            stack.pop()
        return max_flow[0]

    def max_flow_min_cut(self, s, t):
        total_flow = 0
        while self._bfs(s, t):
            for i in range(1, self.n + 1):
                self.cur[i] = self.point_head[i]
            flow = self._dfs(s, t)
            while flow > 0:
                total_flow += flow
                flow = self._dfs(s, t)
        return total_flow


class UndirectedGraph:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_w = [0] * 2
        self.edge_p = [0] * 2
        self.edge_to = [0] * 2
        self.edge_next = [0] * 2
        self.edge_id = 2

    def _add_single_edge(self, u, v, w, p):
        assert 1 <= u <= self.n
        assert 1 <= v <= self.n
        self.edge_w.append(w)
        self.edge_p.append(p)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_edge(self, u, v, w, p):
        assert 1 <= u <= self.n
        assert 1 <= v <= self.n
        self._add_single_edge(v, u, w, p)
        self._add_single_edge(u, v, w, p)
        return


class DirectedGraph:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_w = [0] * 2
        self.edge_to = [0] * 2
        self.edge_next = [0] * 2
        self.edge_id = 2

    def add_single_edge(self, u, v, w):
        assert 1 <= u <= self.n
        assert 1 <= v <= self.n
        self.edge_w.append(w)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def get_edge_ids(self, u):
        assert 1 <= u <= self.n
        i = self.point_head[u]
        ans = []
        while i:
            ans.append(i)
            i = self.edge_next[i]
        return


class DinicMaxflowMinCost:
    def __init__(self, n):
        self.n = n
        self.vis = [0] * (self.n + 1)
        self.point_head = [0] * (self.n + 1)
        self.edge_capacity = [0] * 2
        self.edge_cost = [0] * 2
        self.edge_to = [0] * 2
        self.edge_next = [0] * 2
        self.h = [math.inf] * (self.n + 1)
        self.dis = [math.inf] * (self.n + 1)
        self.max_flow = 0
        self.min_cost = 0
        self.edge_id = 2
        self.pre_edge = [0] * (self.n + 1)
        self.pre_point = [0] * (self.n + 1)

    def _add_single_edge(self, u, v, cap, c):
        self.edge_capacity.append(cap)
        self.edge_cost.append(c)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_edge(self, u, v, cap, c):
        assert 1 <= u <= self.n
        assert 1 <= v <= self.n
        self._add_single_edge(u, v, cap, c)
        self._add_single_edge(v, u, 0, -c)
        return

    def _spfa(self, s):
        self.h[s] = 0
        q = deque([s])
        self.vis[s] = 1
        while q:
            u = q.popleft()
            self.vis[u] = 0
            i = self.point_head[u]
            while i:
                v = self.edge_to[i]
                if self.edge_capacity[i] > 0 and self.h[v] > self.h[u] + self.edge_cost[i]:
                    self.h[v] = self.h[u] + self.edge_cost[i]
                    if not self.vis[v]:
                        q.append(v)
                        self.vis[v] = 1
                i = self.edge_next[i]
        return

    def _dijkstra(self, s, t):
        for i in range(1, self.n + 1):
            self.dis[i] = math.inf
            self.vis[i] = 0
        self.dis[s] = 0
        q = [(0, s)]
        while q:
            d, u = heapq.heappop(q)
            if self.vis[u]:
                continue
            self.vis[u] = 1
            i = self.point_head[u]
            while i:
                v = self.edge_to[i]
                nc = self.h[u] - self.h[v] + self.edge_cost[i]
                if self.edge_capacity[i] > 0 and self.dis[v] > self.dis[u] + nc:
                    self.dis[v] = self.dis[u] + nc
                    self.pre_edge[v] = i
                    self.pre_point[v] = u
                    if not self.vis[v]:
                        heapq.heappush(q, (self.dis[v], v))
                i = self.edge_next[i]
        return self.dis[t] < math.inf

    def max_flow_min_cost(self, s, t):
        self._spfa(s)
        while self._dijkstra(s, t):
            for i in range(1, self.n + 1):
                self.h[i] += self.dis[i]

            cur_flow = math.inf
            v = t
            while v != s:
                i = self.pre_edge[v]
                c = self.edge_capacity[i]
                cur_flow = cur_flow if cur_flow < c else c
                v = self.pre_point[v]

            v = t
            while v != s:
                i = self.pre_edge[v]
                self.edge_capacity[i] -= cur_flow
                self.edge_capacity[i ^ 1] += cur_flow
                v = self.pre_point[v]

            self.max_flow += cur_flow
            self.min_cost += cur_flow * self.h[t]

        return self.max_flow, self.min_cost
