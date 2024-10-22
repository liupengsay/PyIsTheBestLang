import math
from collections import defaultdict, deque
from heapq import heappush, heappop

from src.data_structure.sorted_list.template import SortedList


class WeightedGraphForShortestPathMST:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_weight = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.dis = [math.inf]
        self.edge_id = 1
        return

    def add_directed_edge(self, u, v, w):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.edge_weight.append(w)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_undirected_edge(self, u, v, w):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.add_directed_edge(u, v, w)
        self.add_directed_edge(v, u, w)
        return

    def get_edge_ids(self, u):
        assert 0 <= u < self.n
        i = self.point_head[u]
        ans = []
        while i:
            ans.append(i)
            i = self.edge_next[i]
        return

    def dijkstra(self, src=0, initial=0):
        self.dis = [math.inf] * (self.n + 1)
        stack = [initial * self.n + src]
        self.dis[src] = initial
        while stack:
            val = heappop(stack)
            d, u = val // self.n, val % self.n
            if self.dis[u] < d:
                continue
            i = self.point_head[u]
            while i:
                w = self.edge_weight[i]
                j = self.edge_to[i]
                dj = d + w
                if dj < self.dis[j]:
                    self.dis[j] = dj
                    heappush(stack, dj * self.n + j)
                i = self.edge_next[i]
        return

    def shortest_path_mst(self, root=0, ceil=0):
        dis = [math.inf] * self.n
        stack = [root]
        dis[root] = 0
        edge_ids = [-1] * self.n
        weights = [math.inf] * self.n
        while stack:
            val = heappop(stack)
            d, u = val // self.n, val % self.n
            if dis[u] < d:
                continue
            ind = self.point_head[u]
            while ind:
                w = self.edge_weight[ind]
                j = self.edge_to[ind]
                dj = w + d
                if dj < dis[j] or (dj == dis[j] and w < weights[j]):
                    dis[j] = dj
                    edge_ids[j] = (ind + 1) // 2
                    weights[j] = w
                    heappush(stack, dj * self.n + j)
                ind = self.edge_next[ind]
        dis = [dis[i] * self.n + i for i in range(self.n)]
        dis.sort()
        return [edge_ids[x % self.n] for x in dis[1:ceil + 1]]

class LimitedWeightedGraph:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_weight1 = [0]
        self.edge_weight2 = [0]
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.time = [math.inf]
        self.edge_id = 1
        return

    def add_directed_edge(self, u, v, t, c):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.edge_weight1.append(t)
        self.edge_weight2.append(c)
        self.edge_from.append(u)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_undirected_edge(self, u, v, t, c):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.add_directed_edge(u, v, t, c)
        self.add_directed_edge(v, u, t, c)
        return

    def get_edge_ids(self, u):
        assert 0 <= u < self.n
        i = self.point_head[u]
        ans = []
        while i:
            ans.append(i)
            i = self.edge_next[i]
        return

    def limited_dijkstra(self, src=0, des=0, ceil=math.inf, initial=0):
        self.time = [ceil] * (self.n + 1)
        stack = [initial * ceil * self.n + 0 * self.n + src]
        self.time[src] = initial  # min(cost) when time<ceil
        while stack:
            val = heappop(stack)  # cost time point
            cost, tm, u = val // (ceil * self.n), (val % (ceil * self.n)) // self.n, (val % (ceil * self.n)) % self.n
            if u == des:
                return cost
            i = self.point_head[u]
            while i:
                t = self.edge_weight1[i]
                c = self.edge_weight2[i]
                j = self.edge_to[i]
                dj = tm + t
                if dj < self.time[j]:
                    self.time[j] = dj
                    heappush(stack, (cost + c) * ceil * self.n + dj * self.n + j)
                i = self.edge_next[i]
        return -1

    def limited_dijkstra_tuple(self, src=0, des=0, ceil=math.inf, initial=0):
        self.time = [ceil] * (self.n + 1)
        stack = [(initial, 0, src)]
        self.time[src] = initial  # min(cost) when time<ceil
        while stack:
            cost, tm, u = heappop(stack)  # cost time point
            if u == des:
                return cost
            i = self.point_head[u]
            while i:
                t = self.edge_weight1[i]
                c = self.edge_weight2[i]
                j = self.edge_to[i]
                dj = tm + t
                if dj < self.time[j]:
                    self.time[j] = dj
                    heappush(stack, (cost + c, dj, j))
                i = self.edge_next[i]
        return -1




class WeightedGraphForDijkstra:
    def __init__(self, n, inf=math.inf):
        self.n = n
        self.point_head = [0] * self.n
        self.edge_weight = [0]
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.edge_id = 1
        self.inf = inf
        return

    def add_directed_edge(self, i, j, w):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        self.edge_weight.append(w)
        self.edge_from.append(i)
        self.edge_to.append(j)
        self.edge_next.append(self.point_head[i])
        self.point_head[i] = self.edge_id
        self.edge_id += 1
        return

    def add_undirected_edge(self, i, j, w):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        self.add_directed_edge(i, j, w)
        self.add_directed_edge(j, i, w)
        return

    def get_edge_ids(self, u):
        assert 0 <= u < self.n
        ind = self.point_head[u]
        edge_ids = []
        while ind:
            edge_ids.append(ind)
            ind = self.edge_next[ind]
        return edge_ids

    def get_to_nodes(self, u):
        assert 0 <= u < self.n
        ind = self.point_head[u]
        to_nodes = []
        while ind:
            to_nodes.append(self.edge_to[ind])
            ind = self.edge_next[ind]
        return to_nodes

    def dijkstra_for_shortest_path(self, src=0, initial=0):
        dis = [self.inf] * self.n
        stack = [initial * self.n + src]
        dis[src] = initial
        while stack:
            val = heappop(stack)
            d, i = val // self.n, val % self.n
            if dis[i] < d:
                continue
            ind = self.point_head[i]
            while ind:
                w = self.edge_weight[ind]
                j = self.edge_to[ind]
                dj = d + w
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, dj * self.n + j)
                ind = self.edge_next[ind]
        return dis

    def dijkstra_for_shortest_path_float(self, src=0, initial=0):
        dis = [self.inf] * self.n
        stack = [(initial, src)]
        dis[src] = initial
        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            ind = self.point_head[i]
            while ind:
                w = self.edge_weight[ind]
                j = self.edge_to[ind]
                dj = d + w
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, (dj, j))
                ind = self.edge_next[ind]
        return dis

    def dijkstra_for_longest_path(self, src=0, initial=0):
        dis = [-self.inf] * self.n
        stack = [-initial * self.n - src]
        dis[src] = initial
        while stack:
            val = -heappop(stack)
            d, i = val // self.n, val % self.n
            if dis[i] > d:
                continue
            ind = self.point_head[i]
            while ind:
                w = self.edge_weight[ind]
                j = self.edge_to[ind]
                dj = d + w
                if dj > dis[j]:
                    dis[j] = dj
                    heappush(stack, -dj * self.n - j)
                ind = self.edge_next[ind]
        return dis

    def dijkstra_for_cnt_of_shortest_path(self, src=0, initial=0, mod=-1):
        """number of shortest path"""
        assert 0 <= src < self.n
        dis = [self.inf] * self.n
        cnt = [0] * self.n
        dis[src] = initial
        cnt[src] = 1
        stack = [initial * self.n + src]
        while stack:
            val = heappop(stack)
            d, i = val // self.n, val % self.n
            if dis[i] < d:
                continue
            ind = self.point_head[i]
            while ind:
                w = self.edge_weight[ind]
                j = self.edge_to[ind]
                dj = d + w
                if dj < dis[j]:
                    dis[j] = dj
                    cnt[j] = cnt[i]
                    heappush(stack, dj * self.n + j)
                elif dj == dis[j]:
                    cnt[j] += cnt[i]
                    if mod != -1:
                        cnt[j] %= mod
                ind = self.edge_next[ind]
        return dis, cnt

    def dijkstra_for_strictly_second_shortest_path(self, src=0, initial=0):
        dis = [self.inf] * self.n * 2
        stack = [initial * self.n + src]
        dis[src * 2] = initial
        while stack:
            val = heappop(stack)
            d, i = val // self.n, val % self.n
            if dis[i * 2 + 1] < d:
                continue
            ind = self.point_head[i]
            while ind:
                w = self.edge_weight[ind]
                j = self.edge_to[ind]
                dj = d + w
                if dj < dis[j * 2]:
                    dis[j * 2 + 1] = dis[j * 2]
                    dis[j * 2] = dj
                    heappush(stack, dj * self.n + j)
                elif dis[j * 2] < dj < dis[j * 2 + 1]:
                    dis[j * 2 + 1] = dj
                    heappush(stack, dj * self.n + j)
                ind = self.edge_next[ind]
        return dis

    def dijkstra_for_cnt_of_strictly_second_shortest_path(self, src=0, initial=0, mod=-1):
        dis = [self.inf] * self.n * 2
        dis[src * 2] = initial
        cnt = [0] * self.n * 2
        cnt[src * 2] = 1
        stack = [initial * 2 * self.n + src * 2]
        while stack:
            val = heappop(stack)
            d, i = val // self.n // 2, val % (2 * self.n)
            i, state = i // 2, i % 2
            val = i * 2 + state
            if dis[val] < d:
                continue
            ind = self.point_head[i]
            while ind:
                w = self.edge_weight[ind]
                j = self.edge_to[ind]
                dj = d + w
                if dj < dis[j * 2]:
                    dis[j * 2 + 1] = dis[j * 2]
                    cnt[j * 2 + 1] = cnt[j * 2]
                    dis[j * 2] = dj
                    cnt[j * 2] = cnt[val]
                    heappush(stack, dj * 2 * self.n + j * 2)
                    heappush(stack, dis[j * 2 + 1] * 2 * self.n + j * 2 + 1)
                elif dj == dis[j * 2]:
                    cnt[j * 2] += cnt[val]
                    if mod != -1:
                        cnt[j * 2] %= mod
                elif dj < dis[j * 2 + 1]:
                    dis[j * 2 + 1] = dj
                    cnt[j * 2 + 1] = cnt[val]
                    heappush(stack, dj * 2 * self.n + j * 2 + 1)
                elif dj == dis[j * 2 + 1]:
                    cnt[j * 2 + 1] += cnt[val]
                    if mod != -1:
                        cnt[j * 2 + 1] %= mod
                ind = self.edge_next[ind]
        return dis, cnt

    def dijkstra_for_shortest_path_from_src_to_dst(self, src, dst):
        assert 0 <= src < self.n
        assert 0 <= dst < self.n
        dis = [self.inf] * self.n
        stack = [0 * self.n + src]
        dis[src] = 0
        parent = [-1] * self.n
        while stack:
            val = heappop(stack)
            d, i = val // self.n, val % self.n
            if dis[i] < d:
                continue
            if dst == i:
                break
            ind = self.point_head[i]
            while ind:
                w = self.edge_weight[ind]
                j = self.edge_to[ind]
                dj = d + w
                if dj < dis[j]:
                    dis[j] = dj
                    parent[j] = i
                    heappush(stack, dj * self.n + j)
                ind = self.edge_next[ind]
        path = [dst]
        while parent[path[-1]] != -1:
            path.append(parent[path[-1]])
        path.reverse()
        return path, dis[dst]

    def bfs_for_shortest_path_from_src_to_dst(self, src, dst):
        dis = [self.inf] * self.n
        dis[src] = 0
        stack = [src]
        parent = [-1] * self.n
        while stack:
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if self.edge_weight[ind] == 1:
                        dj = dis[i] + 1
                        if dj < dis[j]:
                            dis[j] = dj
                            parent[j] = i
                            nex.append(j)
                    ind = self.edge_next[ind]
            stack = nex
        path = [dst]
        while parent[path[-1]] != -1:
            path.append(parent[path[-1]])
        path.reverse()
        return path, dis[dst]

    def bfs_for_shortest_path(self, src=0, initial=0):
        dis = [self.inf] * self.n
        dis[src] = initial
        stack = [src]
        while stack:
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if self.edge_weight[ind] == 1:
                        dj = dis[i] + 1
                        if dj < dis[j]:
                            dis[j] = dj
                            nex.append(j)
                    ind = self.edge_next[ind]
            stack = nex
        return dis

    def bfs_for_shortest_path_with_odd_and_even(self, src=0, initial=0):
        dis = [self.inf] * self.n * 2
        dis[src * 2] = initial
        stack = [src * 2]
        while stack:
            nex = []
            for val in stack:
                i = val // 2
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if self.edge_weight[ind] == 1:
                        dj = dis[val] + 1
                        sj = j * 2 + dj % 2
                        if dj < dis[sj]:
                            dis[sj] = dj
                            nex.append(sj)
                    ind = self.edge_next[ind]
            stack = nex
        return dis

    def bfs_for_cnt_of_shortest_path(self, src=0, initial=0, mod=-1):
        dis = [self.inf] * self.n
        dis[src] = initial
        stack = [src]
        cnt = [0] * self.n
        cnt[src] = 1
        while stack:
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if self.edge_weight[ind] == 1:
                        dj = dis[i] + 1
                        if dj < dis[j]:
                            dis[j] = dj
                            nex.append(j)
                            cnt[j] = cnt[i]
                        elif dj == dis[j]:
                            cnt[j] += cnt[i]
                            if mod != -1:
                                cnt[j] %= mod
                    ind = self.edge_next[ind]
            stack = nex
        return dis, cnt

    def bfs_for_cnt_of_strictly_second_shortest_path(self, src=0, initial=0, mod=-1):
        dis = [self.inf] * self.n * 2
        dis[src * 2] = initial
        cnt = [0] * self.n * 2
        cnt[src * 2] = 1
        stack = [src * 2]
        while stack:
            nex = []
            for val in stack:
                i = val // 2
                ind = self.point_head[i]
                d = dis[val]
                while ind:
                    w = self.edge_weight[ind]
                    if w == 1:
                        j = self.edge_to[ind]
                        dj = d + 1
                        if dj < dis[j * 2]:
                            dis[j * 2 + 1] = dis[j * 2]
                            cnt[j * 2 + 1] = cnt[j * 2]
                            nex.append(j * 2)
                            dis[j * 2] = dj
                            cnt[j * 2] = cnt[val]
                        elif dj == dis[j * 2]:
                            cnt[j * 2] += cnt[val]
                            if mod != -1:
                                cnt[j * 2] %= mod
                        elif dj < dis[j * 2 + 1]:
                            dis[j * 2 + 1] = dj
                            cnt[j * 2 + 1] = cnt[val]
                            nex.append(j * 2 + 1)
                        elif dj == dis[j * 2 + 1]:
                            cnt[j * 2 + 1] += cnt[val]
                            if mod != -1:
                                cnt[j * 2 + 1] %= mod
                    ind = self.edge_next[ind]
            stack = nex
        return dis, cnt


class UnDirectedShortestCycle:
    def __init__(self):
        return

    @staticmethod
    def find_shortest_cycle_with_node(n: int, dct) -> int:
        # brute force by point
        ans = math.inf
        for i in range(n):
            dist = [math.inf] * n
            par = [-1] * n
            dist[i] = 0
            stack = [(0, i)]
            while stack:
                _, x = heappop(stack)
                for child in dct[x]:
                    if dist[x] > ans:
                        break
                    if dist[child] > dct[x][child] + dist[x]:
                        dist[child] = dct[x][child] + dist[x]
                        par[child] = x
                        heappush(stack, (dist[child], child))
                    elif par[x] != child and par[child] != x:
                        cur = dist[x] + dist[child] + dct[x][child]
                        ans = ans if ans < cur else cur
        return ans if ans != math.inf else -1

    @staticmethod
    def find_shortest_cycle_with_edge(n: int, dct, edges) -> int:
        # brute force by edge

        ans = math.inf
        for x, y, w in edges:
            dct[x].pop(y)
            dct[y].pop(x)

            dis = [math.inf] * n
            stack = [(0, x)]
            dis[x] = 0

            while stack:
                d, i = heappop(stack)
                if dis[i] < d:
                    continue
                if i == y:
                    break
                for j in dct[i]:
                    dj = dct[i][j] + d
                    if dj < dis[j]:
                        dis[j] = dj
                        heappush(stack, (dj, j))

            ans = ans if ans < dis[y] + w else dis[y] + w
            dct[x][y] = w
            dct[y][x] = w
        return ans if ans < math.inf else -1
