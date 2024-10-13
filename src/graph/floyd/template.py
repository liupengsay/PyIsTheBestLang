import math


class WeightedGraphForFloyd:
    def __init__(self, n, inf=math.inf):
        self.n = n
        self.inf = inf
        self.dis = [self.inf] * self.n * self.n
        for i in range(self.n):
            self.dis[i * self.n + i] = 0
        self.cnt = []
        return

    def add_undirected_edge_initial(self, i, j, w):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        self.dis[i * self.n + j] = self.dis[j * self.n + i] = min(self.dis[i * self.n + j], w)
        return

    def add_directed_edge_initial(self, i, j, w):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        self.dis[i * self.n + j] = min(self.dis[i * self.n + j], w)
        return

    def initialize_undirected(self):
        for k in range(self.n):
            self.update_point_undirected(k)
        return

    def initialize_directed(self):
        for k in range(self.n):
            self.update_point_directed(k)
        return

    def add_undirected_edge(self, i, j, w):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        for x in range(self.n):
            if self.dis[x * self.n + i] == self.inf:
                continue
            for y in range(x + 1, self.n):
                cur = min(self.dis[x * self.n + i] + w + self.dis[y * self.n + j],
                          self.dis[y * self.n + i] + w + self.dis[x * self.n + j])
                self.dis[x * self.n + y] = self.dis[y * self.n + x] = min(cur, self.dis[x * self.n + y])
        return

    def add_directed_edge(self, i, j, w):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        for x in range(self.n):
            if self.dis[x * self.n + i] == self.inf:
                continue
            for y in range(self.n):
                cur = self.dis[x * self.n + i] + w + self.dis[j * self.n + y]
                self.dis[x * self.n + y] = min(cur, self.dis[x * self.n + y])
        return

    def get_cnt_of_shortest_path_undirected(self, mod=-1):
        self.cnt = [0] * self.n * self.n
        for i in range(self.n):
            self.cnt[i * self.n + i] = 1
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.dis[i * self.n + j] < self.inf:
                    self.cnt[i * self.n + j] = self.cnt[j * self.n + i] = 1
        for k in range(self.n):
            for i in range(self.n):
                if self.dis[i * self.n + k] == self.inf:
                    continue
                for j in range(i + 1, self.n):
                    if self.dis[i * self.n + k] + self.dis[k * self.n + j] < self.dis[j * self.n + i]:
                        self.dis[i * self.n + j] = self.dis[j * self.n + i] = self.dis[i * self.n + k] + self.dis[
                            k * self.n + j]
                        self.cnt[i * self.n + j] = self.cnt[j * self.n + i] = self.cnt[i * self.n + k] * self.cnt[
                            k * self.n + j]

                    elif self.dis[i * self.n + k] + self.dis[k * self.n + j] == self.dis[j * self.n + i]:
                        self.cnt[i * self.n + j] += self.cnt[i * self.n + k] * self.cnt[k * self.n + j]
                        self.cnt[j * self.n + i] += self.cnt[i * self.n + k] * self.cnt[k * self.n + j]
                        if mod != -1:
                            self.cnt[i * self.n + j] %= mod
                            self.cnt[j * self.n + i] %= mod
        return

    def get_cnt_of_shortest_path_directed(self, mod=-1):
        self.cnt = [0] * self.n * self.n
        for i in range(self.n):
            self.cnt[i * self.n + i] = 1
        for i in range(self.n):
            for j in range(self.n):
                if self.dis[i * self.n + j] < self.inf:
                    self.cnt[i * self.n + j] = self.cnt[j * self.n + i] = 1
        for k in range(self.n):
            for i in range(self.n):
                if self.dis[i * self.n + k] == self.inf:
                    continue
                for j in range(self.n):
                    if self.dis[i * self.n + k] + self.dis[k * self.n + j] < self.dis[j * self.n + i]:
                        self.dis[i * self.n + j] = self.dis[i * self.n + k] + self.dis[
                            k * self.n + j]
                        self.cnt[i * self.n + j] = self.cnt[i * self.n + k] * self.cnt[
                            k * self.n + j]

                    elif self.dis[i * self.n + k] + self.dis[k * self.n + j] == self.dis[j * self.n + i]:
                        self.cnt[i * self.n + j] += self.cnt[i * self.n + k] * self.cnt[k * self.n + j]
                        if mod != -1:
                            self.cnt[i * self.n + j] %= mod
        return

    def update_point_undirected(self, k):
        for i in range(self.n):
            if self.dis[i * self.n + k] == self.inf:
                continue
            for j in range(i + 1, self.n):
                cur = self.dis[i * self.n + k] + self.dis[k * self.n + j]
                self.dis[i * self.n + j] = self.dis[j * self.n + i] = min(self.dis[i * self.n + j], cur)
        return

    def update_point_directed(self, k):
        for i in range(self.n):
            if self.dis[i * self.n + k] == self.inf:
                continue
            for j in range(self.n):
                cur = self.dis[i * self.n + k] + self.dis[k * self.n + j]
                self.dis[i * self.n + j] = min(self.dis[i * self.n + j], cur)
        return

    def get_nodes_between_src_and_dst(self, i, j):
        path = [x for x in range(self.n) if
                self.dis[i * self.n + x] + self.dis[x * self.n + j] == self.dis[i * self.n + j]]
        return path
