from src.utils.fast_io import FastIO


class WeightedTree:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_weight = [0]
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

    def get_edge_ids(self, u):
        assert 0 <= u < self.n
        i = self.point_head[u]
        ans = []
        while i:
            ans.append(i)
            i = self.edge_next[i]
        return

    def dfs_order(self, root=0):

        self.order = 0
        # index is original node value is dfs self.order
        self.start = [-1] * self.n
        # index is original node value is the maximum subtree dfs self.order
        self.end = [-1] * self.n
        # index is original node and value is its self.parent
        self.parent = [-1] * self.n
        stack = [root]
        # self.depth of every original node
        self.depth = [0] * self.n
        # index is dfs self.order and value is original node
        self.order_to_node = [-1] * self.n
        while stack:
            i = stack.pop()
            if i >= 0:
                self.start[i] = self.order
                self.order_to_node[self.order] = i
                self.end[i] = self.order
                self.order += 1
                stack.append(~i)
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    # the self.order of son nodes can be assigned for lexicographical self.order
                    if j != self.parent[i]:
                        self.parent[j] = i
                        self.depth[j] = self.depth[i] + 1
                        stack.append(j)
                    ind = self.edge_next[ind]
            else:
                i = ~i
                if self.parent[i] != -1:
                    self.end[self.parent[i]] = self.end[i]

        return

    def heuristic_merge(self):
        ans = [0] * self.n
        sub = [None for _ in range(self.n)]
        index = list(range(self.n))
        self.parent = [-1] * self.n
        self.depth = [0] * self.n
        stack = [0]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if j != self.parent[i]:
                        self.parent[j] = i
                        self.depth[j] = self.depth[i] + 1
                        stack.append(j)
                    ind = self.edge_next[ind]
            else:
                i = ~i
                sub[index[i]] = {self.depth[i]: 1}
                ans[i] = self.depth[i]
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if j != self.parent[i]:
                        a, b = index[i], index[j]
                        if len(sub[a]) > len(sub[b]):
                            res = ans[i]
                            a, b = b, a
                        else:
                            res = ans[j]

                        for x in sub[a]:
                            sub[b][x] = sub[b].get(x, 0) + sub[a][x]
                            if (sub[b][x] > sub[b][res]) or (sub[b][x] == sub[b][res] and x < res):
                                res = x
                        sub[a] = None
                        ans[i] = res
                        index[i] = b
                    ind = self.edge_next[ind]

        return [ans[i] - self.depth[i] for i in range(self.n)]

    # class Graph(WeightedTree):
    def tree_dp(self, nums):
        ans = [0] * self.n
        parent = [-1] * self.n
        stack = [0]
        res = max(nums)
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if j != parent[i]:
                        parent[j] = i
                        stack.append(j)
                    ind = self.edge_next[ind]
            else:
                i = ~i
                ind = self.point_head[i]
                a = b = 0
                while ind:
                    j = self.edge_to[ind]
                    if j != parent[i]:
                        cur = ans[j] - self.edge_weight[ind]
                        if cur > a:
                            a, b = cur, a
                        elif cur > b:
                            b = cur
                    ind = self.edge_next[ind]
                res = max(res, a + b + nums[i])
                ans[i] = a + nums[i]
        return res


class ReadGraph:
    def __init__(self):
        return

    @staticmethod
    def read(ac=FastIO()):
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        stack = [0]
        sub = [0] * n
        while stack:
            val = stack.pop()
            if val >= 0:
                x, fa = val // n, val % n
                stack.append(~val)
                for y in dct[x]:
                    if y != fa:
                        stack.append(y * n + x)
            else:
                val = ~val
                x, fa = val // n, val % n
                for y in dct[x]:
                    if y != fa:
                        sub[x] += sub[y]
                sub[x] += 1
        return sub


class ReRootDP:
    def __init__(self):
        return

    @staticmethod
    def get_tree_distance_weight(dct, weight):
        # Calculate the total distance from each node of the tree to all other nodes
        # each node has weight

        n = len(dct)
        sub = weight[:]
        s = sum(weight)  # default equal to [1]*n
        ans = [0] * n  # distance to all other nodes

        # first bfs to get ans[0] and subtree weight from bottom to top
        stack = [[0, -1, 1]]
        while stack:
            i, fa, state = stack.pop()
            if state:
                stack.append([i, fa, 0])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i, 1])
            else:
                for j in dct[i]:
                    if j != fa:
                        sub[i] += sub[j]
                        ans[i] += ans[j] + sub[j]

        # second bfs to get all ans[i]
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            for j in dct[i]:
                if j != fa:
                    # sub[j] equal j up to i
                    # s - sub[j] equal to i down to j
                    # change = -sub[j] + s - sub[j]
                    ans[j] = ans[i] - sub[j] + s - sub[j]
                    stack.append([j, i])
        return ans

    @staticmethod
    def get_tree_centroid(dct) -> int:
        # the smallest centroid of tree
        # equal the node with minimum of maximum subtree node cnt
        # equivalent to the node which has the shortest distance from all other nodes
        n = len(dct)
        sub = [1] * n  # subtree size of i-th node rooted by 0
        ma = [0] * n  # maximum subtree node cnt or i-rooted
        ma[0] = n
        center = 0
        stack = [[0, -1, 1]]
        while stack:
            i, fa, state = stack.pop()
            if state:
                stack.append([i, fa, 0])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i, 1])
            else:
                for j in dct[i]:
                    if j != fa:
                        sub[i] += sub[j]
                        ma[i] = ma[i] if ma[i] > sub[j] else sub[j]
                # like re-rooted dp to check the maximum subtree size
                ma[i] = ma[i] if ma[i] > n - sub[i] else n - sub[i]
                if ma[i] < ma[center] or (ma[i] == ma[center] and i < center):
                    center = i
        return center

    @staticmethod
    def get_tree_distance(dct):
        # Calculate the total distance from each node of the tree to all other nodes

        n = len(dct)
        sub = [1] * n  # Number of subtree nodes
        ans = [0] * n  # The sum of distances to all other nodes

        # first bfs to get ans[0] and subtree weight from bottom to top
        stack = [(0, -1, 1)]
        while stack:
            i, fa, state = stack.pop()
            if state:
                stack.append((i, fa, 0))
                for j in dct[i]:
                    if j != fa:
                        stack.append((j, i, 1))
            else:
                for j in dct[i]:
                    if j != fa:
                        sub[i] += sub[j]
                        ans[i] += ans[j] + sub[j]

        # second bfs to get all ans[i]
        stack = [(0, -1)]
        while stack:
            i, fa = stack.pop()
            for j in dct[i]:
                if j != fa:
                    # sub[j] equal j up to i
                    # s - sub[j] equal to i down to j
                    # change = -sub[j] + s - sub[j]
                    ans[j] = ans[i] - sub[j] + n - sub[j]
                    stack.append((j, i))
        return ans

    @staticmethod
    def get_tree_distance_max(dct):
        # Calculate the maximum distance from each node of the tree to all other nodes
        # point bfs on diameter can also be used

        n = len(dct)
        sub = [[0, 0] for _ in range(n)]

        # first bfs compute the largest distance and second large distance from bottom to up
        stack = [[0, -1, 1]]
        while stack:
            i, fa, state = stack.pop()
            if state:
                stack.append([i, fa, 0])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i, 1])
            else:
                a, b = sub[i]
                for j in dct[i]:
                    if j != fa:
                        x = sub[j][0] + 1
                        if x >= a:
                            a, b = x, a
                        elif x >= b:
                            b = x
                sub[i] = [a, b]

        # second bfs compute large distance from up to bottom
        stack = [(0, -1, 0)]
        ans = [s[0] for s in sub]
        while stack:
            i, fa, d = stack.pop()
            ans[i] = ans[i] if ans[i] > d else d
            for j in dct[i]:
                if j != fa:
                    nex = d
                    x = sub[j][0] + 1
                    a, b = sub[i]
                    # distance from current child nodes excluded
                    if x == a:
                        nex = nex if nex > b else b
                    else:
                        nex = nex if nex > a else a
                    stack.append((j, i, nex + 1))
        return ans

    @staticmethod
    def get_tree_distance_max_weighted(dct, weights):
        # Calculate the maximum distance from each node of the tree to all other nodes
        # point bfs on diameter can also be used

        n = len(dct)
        sub = [[0, 0] for _ in range(n)]

        # first bfs compute the largest distance and second large distance from bottom to up
        stack = [(0, -1)]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append((~i, fa))
                for j in dct[i]:
                    if j != fa:
                        stack.append((j, i))
            else:
                i = ~i
                a, b = sub[i]
                for j in dct[i]:
                    if j != fa:
                        x = sub[j][0] + weights[j]
                        if x >= a:
                            a, b = x, a
                        elif x >= b:
                            b = x
                sub[i] = [a, b]

        # second bfs compute large distance from up to bottom
        stack = [(0, -1, 0)]
        ans = [s[0] for s in sub]
        while stack:
            i, fa, d = stack.pop()
            ans[i] = ans[i] if ans[i] > d else d
            for j in dct[i]:
                if j != fa:
                    nex = d
                    x = sub[j][0] + weights[j]
                    a, b = sub[i]
                    # distance from current child nodes excluded
                    if x == a:
                        nex = nex if nex > b else b
                    else:
                        nex = nex if nex > a else a
                    stack.append((j, i, nex + weights[i]))
        return ans