import math
from collections import deque
from heapq import heappop, heappush

from src.data_structure.tree_array.template import PointDescendPreMin
from src.graph.union_find.template import UnionFind
from src.utils.fast_io import inf


class ManhattanMST:
    def __init__(self):
        return

    @staticmethod
    def build(points):
        n = len(points)
        edges = list()

        def build():
            pos.sort()
            tree.initialize()
            mid = dict()
            for xx, yy, i in pos:
                val = tree.pre_min(dct[yy - xx] + 1)
                if val < inf:
                    edges.append((val + yy + xx, i, mid[val]))
                tree.point_descend(dct[yy - xx] + 1, -yy - xx)
                mid[-yy - xx] = i
            return

        nodes = set()
        for x, y in points:
            nodes.add(y - x)
            nodes.add(x - y)
            nodes.add(x + y)
            nodes.add(-y - x)
        nodes = sorted(nodes)
        dct = {num: i for i, num in enumerate(nodes)}
        m = len(dct)
        tree = PointDescendPreMin(m)
        pos = [(x, y, i) for i, (x, y) in enumerate(points)]
        build()
        pos = [(y, x, i) for i, (x, y) in enumerate(points)]
        build()
        pos = [(-y, x, i) for i, (x, y) in enumerate(points)]
        build()
        pos = [(x, -y, i) for i, (x, y) in enumerate(points)]
        build()

        uf = UnionFind(n)
        edges.sort()
        select = []
        ans = 0
        for w, u, v in edges:
            if uf.union(u, v):
                ans += w
                select.append((u, v))
                if uf.part == 1:
                    break
        return ans, select


class KruskalMinimumSpanningTree:
    def __init__(self, edges, n, method="kruskal"):
        self.n = n
        self.edges = edges
        self.cost = 0
        self.cnt = 0
        self.gen_minimum_spanning_tree(method)
        return

    def gen_minimum_spanning_tree(self, method):

        if method == "kruskal":
            # Edge priority
            self.edges.sort(key=lambda item: item[2])
            # greedy selection of edges based on weight for connected merging
            uf = UnionFind(self.n)
            for x, y, z in self.edges:
                if uf.union(x, y):
                    self.cost += z
            if uf.part != 1:
                self.cost = -1
        else:  # prim
            # Point priority with Dijkstra
            dct = [dict() for _ in range(self.n)]
            for i, j, w in self.edges:
                c = dct[i].get(j, inf)
                c = c if c < w else w
                dct[i][j] = dct[j][i] = c
            dis = [inf] * self.n
            dis[0] = 0
            visit = [0] * self.n
            stack = [(0, 0)]
            while stack:
                d, i = heappop(stack)
                if visit[i]:
                    continue
                visit[i] = 1
                # cost of mst
                self.cost += d
                # number of connected node
                self.cnt += 1
                for j in dct[i]:
                    w = dct[i][j]
                    if w < dis[j]:
                        dis[j] = w
                        heappush(stack, (w, j))
        return


class PrimMinimumSpanningTree:
    def __init__(self, function):
        self.dis = function
        return

    def build(self, nums):
        n = len(nums)
        ans = nex = 0
        rest = set(list(range(1, n)))
        visit = [inf] * n
        visit[nex] = 0
        while rest:
            i = nex
            rest.discard(i)
            d = visit[i]
            ans += d
            nex = -1
            x1, y1 = nums[i]
            for j in rest:
                x2, y2 = nums[j]
                dj = self.dis(x1, y1, x2, y2)
                if dj < visit[j]:
                    visit[j] = dj
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
        return ans if ans < inf else -1


class SecondMinimumSpanningTree:
    """"get some info of strictly second minimum spanning tree"""

    def __init__(self, dct, strictly=False):
        # default node 0 as root
        self.n = len(dct)  # [dict() for _ in range(n)]
        self.strictly = strictly
        self.parent = [-1] * self.n
        self.depth = [-1] * self.n
        stack = deque([0])
        self.depth[0] = 0
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                if self.depth[j] == -1:
                    self.depth[j] = self.depth[i] + 1
                    self.parent[j] = i
                    stack.append(j)

        # set the number of layers based on the node size
        self.cols = max(2, math.ceil(math.log2(self.n)))
        self.dp = [-1] * self.cols * self.n
        # the maximum and second maximum values of edge weights which is not strictly second
        self.maximum_weight = [-1] * self.cols * self.n
        self.second_maximum_weight = [-1] * self.cols * self.n  # not strictly
        for i in range(self.n):
            self.dp[i * self.cols] = self.parent[i]
            if self.parent[i] != -1:
                self.maximum_weight[i * self.cols] = dct[self.parent[i]][i]

        # dp[i*self.cols+j] as the 2**j th ancestor of node i
        for j in range(1, self.cols):
            for i in range(self.n):
                father = self.dp[i * self.cols + j - 1]
                a, b = self.maximum_weight[i * self.cols + j], self.second_maximum_weight[i * self.cols + j]
                c, d = self.maximum_weight[i * self.cols + j - 1], self.second_maximum_weight[i * self.cols + j - 1]
                self.maximum_weight[i * self.cols + j], self.second_maximum_weight[i * self.cols + j] = self.update(a,
                                                                                                                    b,
                                                                                                                    c,
                                                                                                                    d)
                if father != -1:
                    self.dp[i * self.cols + j] = self.dp[father * self.cols + j - 1]
                    a, b = self.maximum_weight[i * self.cols + j], self.second_maximum_weight[i * self.cols + j]
                    c, d = self.maximum_weight[father * self.cols + j - 1], self.second_maximum_weight[
                        father * self.cols + j - 1]
                    self.maximum_weight[i * self.cols + j], self.second_maximum_weight[i * self.cols + j] = self.update(
                        a, b, c, d)
        return

    def update(self, a, b, c, d):
        if not self.strictly:
            for x in [c, d]:
                if x >= a:
                    a, b = x, a
                elif x >= b:  # this is not strictly
                    b = x
        else:
            for x in [c, d]:
                if x > a:
                    a, b = x, a
                elif a > x > b:  # this is strictly
                    b = x
        return a, b

    def get_dist_weight_max_second(self, x: int, y: int):
        # calculate the maximum and second maximum weights on the shortest path of any point
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        ans_a = ans_b = -1
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            ans_c, ans_d = self.maximum_weight[x * self.cols + int(math.log2(d))], self.second_maximum_weight[
                x * self.cols + int(math.log2(d))]
            ans_a, ans_b = self.update(ans_a, ans_b, ans_c, ans_d)
            x = self.dp[x * self.cols + int(math.log2(d))]
        if x == y:
            return ans_a, ans_b

        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.dp[x * self.cols + k] != self.dp[y * self.cols + k]:
                ans_c, ans_d = self.maximum_weight[x * self.cols + k], self.second_maximum_weight[x * self.cols + k]
                ans_a, ans_b = self.update(ans_a, ans_b, ans_c, ans_d)

                ans_c, ans_d = self.maximum_weight[y * self.cols + k], self.second_maximum_weight[y * self.cols + k]
                ans_a, ans_b = self.update(ans_a, ans_b, ans_c, ans_d)

                x = self.dp[x * self.cols + k]
                y = self.dp[y * self.cols + k]

        ans_c, ans_d = self.maximum_weight[x * self.cols], self.second_maximum_weight[x * self.cols]
        ans_a, ans_b = self.update(ans_a, ans_b, ans_c, ans_d)

        ans_c, ans_d = self.maximum_weight[y * self.cols], self.second_maximum_weight[y * self.cols]
        ans_a, ans_b = self.update(ans_a, ans_b, ans_c, ans_d)
        return ans_a, ans_b


class SecondMinimumSpanningTreeLight:
    """"get some info of strictly second minimum spanning tree"""

    def __init__(self, dct):
        # default node 0 as root
        self.n = len(dct)  # [dict() for _ in range(n)]
        self.parent = [-1] * self.n
        self.depth = [-1] * self.n
        stack = deque([0])
        self.depth[0] = 0
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                if self.depth[j] == -1:
                    self.depth[j] = self.depth[i] + 1
                    self.parent[j] = i
                    stack.append(j)

        # set the number of layers based on the node size
        self.cols = max(2, math.ceil(math.log2(self.n)))
        self.dp = [-1] * self.cols * self.n
        # the maximum and second maximum values of edge weights which is not strictly second
        self.maximum_weight = [-1] * self.cols * self.n
        for i in range(self.n):
            self.dp[i * self.cols] = self.parent[i]
            if self.parent[i] != -1:
                self.maximum_weight[i * self.cols] = dct[self.parent[i]][i]

        # dp[i*self.cols+j] as the 2**j th ancestor of node i
        for j in range(1, self.cols):
            for i in range(self.n):
                father = self.dp[i * self.cols + j - 1]
                a, b = self.maximum_weight[i * self.cols + j], self.maximum_weight[i * self.cols + j - 1]
                self.maximum_weight[i * self.cols + j] = a if a > b else b
                if father != -1:
                    self.dp[i * self.cols + j] = self.dp[father * self.cols + j - 1]
                    a, b = self.maximum_weight[i * self.cols + j], self.maximum_weight[father * self.cols + j - 1]

                    self.maximum_weight[i * self.cols + j] = a if a > b else b
        return

    def get_dist_weight_max_second(self, x: int, y: int):
        # calculate the maximum and second maximum weights on the shortest path of any point
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        a = -1
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            b = self.maximum_weight[x * self.cols + int(math.log2(d))]
            a = a if a > b else b
            x = self.dp[x * self.cols + int(math.log2(d))]
        if x == y:
            return a

        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.dp[x * self.cols + k] != self.dp[y * self.cols + k]:
                b = self.maximum_weight[x * self.cols + k]
                a = a if a > b else b

                b = self.maximum_weight[y * self.cols + k]
                a = a if a > b else b

                x = self.dp[x * self.cols + k]
                y = self.dp[y * self.cols + k]

        b = self.maximum_weight[x * self.cols]
        a = a if a > b else b

        b = self.maximum_weight[y * self.cols]
        a = a if a > b else b
        return a
