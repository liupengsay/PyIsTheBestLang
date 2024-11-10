import math
from heapq import heappop, heappush

from src.graph.union_find.template import UnionFind
from src.struct.tree_array.template import PointDescendPreMin


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
                if val < math.inf:
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
        weight = []
        for w, u, v in edges:
            if uf.union(u, v):
                ans += w
                select.append((u, v))
                weight.append(w)
                if uf.part == 1:
                    break
        return ans, select, weight


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
        else:
            # Point priority with Dijkstra
            dct = [dict() for _ in range(self.n)]
            for i, j, w in self.edges:
                c = dct[i].get(j, math.inf)
                c = c if c < w else w
                dct[i][j] = dct[j][i] = c
            dis = [math.inf] * self.n
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
        visit = [math.inf] * n
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
        return ans if ans < math.inf else -1


class TreeAncestorMinIds:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.edge_id = 1
        self.parent = [-1]
        self.order = 0
        self.start = [-1]
        self.end = [-1]
        self.parent = [-1]
        self.depth = [0]
        self.order_to_node = [-1]
        self.cols = max(2, math.ceil(math.log2(self.n)))
        self.min_ids = [self.n + 1] * self.n * self.cols * 10
        self.father = [-1] * self.n * self.cols
        self.ids = []
        return

    def add_directed_edge(self, u, v):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.edge_from.append(u)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_undirected_edge(self, u, v):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.add_directed_edge(u, v)
        self.add_directed_edge(v, u)
        return

    def build_multiplication(self, ids):
        self.ids = ids
        self.parent = [-1] * self.n
        self.depth = [0] * self.n
        stack = [0]
        while stack:
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    # the self.order of son nodes can be assigned for lexicographical self.order
                    if j != self.parent[i]:
                        self.parent[j] = i
                        self.depth[j] = self.depth[i] + 1
                        nex.append(j)
                    ind = self.edge_next[ind]
            stack = nex

        for i in range(self.n):
            self.father[i * self.cols] = self.parent[i]
            cur = ids[i * 10:i * 10 + 10]
            if self.parent[i] != -1:
                cur = self.update(cur, ids[self.parent[i] * 10:self.parent[i] * 10 + 10])
            self.min_ids[(i * self.cols) * 10:(i * self.cols) * 10 + 10] = cur[:]
        for j in range(1, self.cols):
            for i in range(self.n):
                father = self.father[i * self.cols + j - 1]
                if father != -1:
                    self.min_ids[(i * self.cols + j) * 10: (i * self.cols + j) * 10 + 10] = self.update(
                        self.min_ids[(i * self.cols + j - 1) * 10: (i * self.cols + j - 1) * 10 + 10],
                        self.min_ids[(father * self.cols + j - 1) * 10: (father * self.cols + j - 1) * 10 + 10])
                    self.father[i * self.cols + j] = self.father[father * self.cols + j - 1]
        return

    def update(self, lst1, lst2):
        lst = []

        m, n = len(lst1), len(lst2)
        i = j = 0
        while i < m and j < n and len(lst) < 10:
            if lst1[i] < lst2[j]:
                if not lst or lst[-1] < lst1[i]:
                    lst.append(lst1[i])
                i += 1
            else:
                if not lst or lst[-1] < lst2[j]:
                    lst.append(lst2[j])
                j += 1
        while i < m and len(lst) < 10:
            if not lst or lst[-1] < lst1[i]:
                lst.append(lst1[i])
            i += 1
        while j < n and len(lst) < 10:
            if not lst or lst[-1] < lst2[j]:
                lst.append(lst2[j])
            j += 1
        while len(lst) < 10:
            lst.append(self.n + 1)
        return lst[:10]

    def get_min_ids_between_nodes(self, x: int, y: int):
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        ans = self.update(self.ids[x * 10:x * 10 + 10], self.ids[y * 10:y * 10 + 10])
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            ans = self.update(ans, self.min_ids[(x * self.cols + int(math.log2(d))) * 10:(x * self.cols + int(
                math.log2(d))) * 10 + 10])
            x = self.father[x * self.cols + int(math.log2(d))]

        if x == y:
            return ans

        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.father[x * self.cols + k] != self.father[y * self.cols + k]:
                ans = self.update(ans, self.min_ids[(x * self.cols + k) * 10:(x * self.cols + k) * 10 + 10])
                ans = self.update(ans, self.min_ids[(y * self.cols + k) * 10:(y * self.cols + k) * 10 + 10])
                x = self.father[x * self.cols + k]
                y = self.father[y * self.cols + k]

        ans = self.update(ans, self.min_ids[(x * self.cols) * 10:(x * self.cols) * 10 + 10])
        ans = self.update(ans, self.min_ids[(y * self.cols) * 10:(y * self.cols) * 10 + 10])
        return ans


class TreeMultiplicationMaxSecondWeights:
    def __init__(self, n, strictly=True):
        # strictly_second_minimum_spanning_tree
        self.n = n
        self.strictly = strictly
        self.point_head = [0] * (self.n + 1)
        self.edge_weight = [0]
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.edge_id = 1
        self.depth = [0]
        self.cols = max(2, math.ceil(math.log2(self.n)))
        self.weights = [-1]
        self.father = [-1]
        return

    def add_directed_edge(self, u, v, w):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.edge_weight.append(w)
        self.edge_from.append(u)
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

    def build_multiplication(self):
        self.weights = [-1] * self.n * self.cols * 2
        self.father = [-1] * self.n * self.cols
        self.depth = [0] * self.n
        stack = [0]
        while stack:
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if j != self.father[i * self.cols]:
                        self.father[j * self.cols] = i
                        self.weights[j * self.cols * 2: j * self.cols * 2 + 2] = [self.edge_weight[ind], -1]
                        self.depth[j] = self.depth[i] + 1
                        nex.append(j)
                    ind = self.edge_next[ind]
            stack = nex

        for j in range(1, self.cols):
            for i in range(self.n):
                father = self.father[i * self.cols + j - 1]
                if father != -1:
                    self.weights[(i * self.cols + j) * 2:(i * self.cols + j) * 2 + 2] = self.update(
                        self.weights[(i * self.cols + j - 1) * 2:(i * self.cols + j - 1) * 2 + 2],
                        self.weights[(father * self.cols + j - 1) * 2:(father * self.cols + j - 1) * 2 + 2])
                    self.father[i * self.cols + j] = self.father[father * self.cols + j - 1]
        return

    def update(self, lst1, lst2):
        a, b = lst1
        if not self.strictly:
            for x in lst2:
                if x >= a:
                    a, b = x, a
                elif x >= b:  # this is not strictly
                    b = x
        else:
            for x in lst2:
                if x > a:
                    a, b = x, a
                elif a > x > b:  # this is strictly
                    b = x
        return [a, b]

    def get_max_weights_between_nodes(self, x: int, y: int):
        assert 0 <= x < self.n
        assert 0 <= y < self.n
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        ans = [-1, -1]
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            ans = self.update(ans, self.weights[
                                   (x * self.cols + int(math.log2(d))) * 2:(x * self.cols + int(math.log2(d))) * 2 + 2])
            x = self.father[x * self.cols + int(math.log2(d))]
        if x == y:
            return ans
        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.father[x * self.cols + k] != self.father[y * self.cols + k]:
                ans = self.update(ans, self.weights[(x * self.cols + k) * 2:(x * self.cols + k) * 2 + 2])
                ans = self.update(ans, self.weights[(y * self.cols + k) * 2:(y * self.cols + k) * 2 + 2])
                x = self.father[x * self.cols + k]
                y = self.father[y * self.cols + k]
        ans = self.update(ans, self.weights[(x * self.cols) * 2:(x * self.cols) * 2 + 2])
        ans = self.update(ans, self.weights[(y * self.cols) * 2:(y * self.cols) * 2 + 2])
        return ans


class TreeMultiplicationMaxWeights:
    def __init__(self, n):
        # second_mst|strictly_second_minimum_spanning_tree
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_weight = [0]
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.edge_id = 1
        self.depth = [0]
        self.cols = max(2, math.ceil(math.log2(self.n)))
        self.weights = [-1]
        self.father = [-1]
        return

    def add_directed_edge(self, u, v, w):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.edge_weight.append(w)
        self.edge_from.append(u)
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

    def build_multiplication(self):
        self.weights = [0] * self.n * self.cols
        self.father = [-1] * self.n * self.cols
        self.depth = [0] * self.n
        stack = [0]
        while stack:
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if j != self.father[i * self.cols]:
                        self.father[j * self.cols] = i
                        self.weights[j * self.cols] = self.edge_weight[ind]
                        self.depth[j] = self.depth[i] + 1
                        nex.append(j)
                    ind = self.edge_next[ind]
            stack = nex

        for j in range(1, self.cols):
            for i in range(self.n):
                father = self.father[i * self.cols + j - 1]
                if father != -1:
                    self.weights[i * self.cols + j] = max(self.weights[i * self.cols + j - 1],
                                                          self.weights[father * self.cols + j - 1])
                    self.father[i * self.cols + j] = self.father[father * self.cols + j - 1]
        return

    def get_max_weights_between_nodes(self, x: int, y: int):
        assert 0 <= x < self.n
        assert 0 <= y < self.n
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        ans = 0
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            ans = max(ans, self.weights[x * self.cols + int(math.log2(d))])
            x = self.father[x * self.cols + int(math.log2(d))]
        if x == y:
            return ans
        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.father[x * self.cols + k] != self.father[y * self.cols + k]:
                ans = max(ans, self.weights[x * self.cols + k])
                ans = max(ans, self.weights[y * self.cols + k])
                x = self.father[x * self.cols + k]
                y = self.father[y * self.cols + k]
        ans = max(ans, self.weights[x * self.cols])
        ans = max(ans, self.weights[y * self.cols])
        return ans
