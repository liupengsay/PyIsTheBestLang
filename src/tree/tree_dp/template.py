import math


class WeightedTree:
    def __init__(self, n, root=0):
        self.n = n
        self.root = root
        self.cols = max(2, math.ceil(math.log2(self.n)))
        self.edge_id = 1
        self.point_head = [0] * (self.n + 1)
        self.edge_weight = [0]
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.parent = [-1]
        self.order = 0
        self.start = [-1]
        self.end = [-1]
        self.parent = [-1]
        self.depth = [0]
        self.dis = [0]
        self.order_to_node = [-1]
        self.ancestor = [-1]
        return

    def add_directed_edge(self, u, v, w=1):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.edge_weight.append(w)
        self.edge_from.append(u)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_undirected_edge(self, u, v, w=1):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.add_directed_edge(u, v, w)
        self.add_directed_edge(v, u, w)
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

    def get_to_nodes_weights(self, u):
        assert 0 <= u < self.n
        ind = self.point_head[u]
        to_nodes = []
        while ind:
            to_nodes.append((self.edge_to[ind], self.edge_weight[ind]))
            ind = self.edge_next[ind]
        return to_nodes

    def dfs_order(self):
        self.order = 0
        # index is original node value is dfs self.order
        self.start = [-1] * self.n
        # index is original node value is the maximum subtree dfs self.order
        self.end = [-1] * self.n
        # index is original node and value is its self.parent
        self.parent = [-1] * self.n
        stack = [self.root]
        # self.depth of every original node
        self.depth = [0] * self.n
        # index is dfs self.order and value is original node
        self.order_to_node = [-1] * self.n
        while stack:
            u = stack.pop()
            if u >= 0:
                self.start[u] = self.order
                self.order_to_node[self.order] = u
                self.end[u] = self.order
                self.order += 1
                stack.append(~u)
                for v in self.get_to_nodes(u):
                    # the self.order of son nodes can be assigned for lexicographical self.order
                    if v != self.parent[u]:
                        self.parent[v] = u
                        self.depth[v] = self.depth[u] + 1
                        stack.append(v)
            else:
                u = ~u
                if self.parent[u] != -1:
                    self.end[self.parent[u]] = self.end[u]
        return

    def lca_build_with_multiplication(self):
        self.parent = [-1] * self.n
        self.depth = [-1] * self.n
        stack = [self.root]
        self.depth[self.root] = 0
        while stack:
            u = stack.pop()
            for v in self.get_to_nodes(u):
                if self.depth[v] == -1:
                    self.depth[v] = self.depth[u] + 1
                    self.parent[v] = u
                    stack.append(v)

        self.ancestor = [-1] * self.cols * self.n
        for u in range(self.n):
            self.ancestor[u * self.cols] = self.parent[u]

        for v in range(1, self.cols):
            for u in range(self.n):
                father = self.ancestor[u * self.cols + v - 1]
                if father != -1:
                    self.ancestor[u * self.cols + v] = self.ancestor[father * self.cols + v - 1]
        return

    def lca_get_kth_ancestor(self, node: int, k: int) -> int:
        for i in range(self.cols - 1, -1, -1):
            if k & (1 << i):
                node = self.ancestor[node * self.cols + i]
                if node == -1:
                    break
        return node

    def lca_get_lca_between_nodes(self, u, v):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        while self.depth[u] > self.depth[v]:
            d = self.depth[u] - self.depth[v]
            u = self.ancestor[u * self.cols + int(math.log2(d))]
        if u == v:
            return u
        for k in range(int(math.log2(self.depth[u])), -1, -1):
            if self.ancestor[u * self.cols + k] != self.ancestor[v * self.cols + k]:
                u = self.ancestor[u * self.cols + k]
                v = self.ancestor[v * self.cols + k]
        return self.ancestor[u * self.cols]

    def lca_get_lca_and_dist_between_nodes(self, u, v):
        lca = self.lca_get_lca_between_nodes(u, v)
        depth_u = self.depth[u]
        depth_v = self.depth[v]
        depth_lca = self.depth[lca]
        return lca, depth_u + depth_v - 2 * depth_lca

    # class Graph(WeightedTree):
    def bfs_dis(self):
        stack = [self.root]
        self.dis = [0] * self.n
        while stack:
            u = stack.pop()
            for v, weight in self.get_to_nodes_weights(u):
                if v != self.parent[u]:
                    self.dis[v] = self.dis[u] + weight
                    stack.append(v)
        return

    # class Graph(WeightedTree):
    def heuristic_merge(self):
        dp = [0] * self.n
        sub = [None] * self.n
        nums = list(range(self.n))
        cnt = [0] * self.n
        self.parent = [-1] * self.n
        stack = [self.root]
        while stack:
            u = stack.pop()
            if u >= 0:
                stack.append(~u)
                for v in self.get_to_nodes(u):
                    if v != self.parent[u]:
                        self.parent[v] = u
                        stack.append(v)
            else:
                u = ~u
                pre = {nums[u]: 1}
                cnt[u] = 1
                dp[u] = nums[u]
                for v in self.get_to_nodes(u):
                    if v != self.parent[u]:
                        cur = sub[v]
                        if len(pre) < len(cur):
                            dp[u] = dp[v]
                            cnt[u] = cnt[v]
                            pre, cur = cur, pre
                        for color in cur:
                            pre[color] = pre.get(color, 0) + cur[color]
                            if pre[color] > cnt[u]:
                                cnt[u] = pre[color]
                                dp[u] = color
                            elif pre[color] == cnt[u]:
                                dp[u] += color
                        sub[v] = None
                sub[u] = pre
        return dp

    # class Graph(WeightedTree):
    def tree_dp(self):
        dp = [0] * self.n
        self.parent = [-1] * self.n
        stack = [self.root]
        res = 0
        while stack:
            u = stack.pop()
            if u >= 0:
                stack.append(~u)
                for v in self.get_to_nodes(u):
                    if v != self.parent[u]:
                        self.parent[v] = u
                        stack.append(v)
            else:
                u = ~u
                ind = self.point_head[u]
                a = b = 0
                for v in self.get_to_nodes(u):
                    if v != self.parent[u]:
                        cur = dp[v] - self.edge_weight[ind]
                        if cur > a:
                            a, b = cur, a
                        elif cur > b:
                            b = cur
                    ind = self.edge_next[ind]
                res = max(res, a + b)
                dp[u] = a
        return res

    # class Graph(WeightedTree):
    def reroot_dp(self):
        nums = [0] * self.n
        dp = [0] * self.n
        self.parent = [-1] * self.n
        stack = [self.root]
        while stack:
            u = stack.pop()
            if u >= 0:
                stack.append(~u)
                for v in self.get_to_nodes(u):
                    if v != self.parent[u]:
                        self.parent[v] = u
                        stack.append(v)
            else:
                u = ~u
                dp[u] = nums[u]
                for v in self.get_to_nodes(u):
                    if v != self.parent[u]:
                        dp[u] += max(0, dp[v])

        ndp = dp[:]
        stack = [0]
        while stack:
            u = stack.pop()
            for v in self.get_to_nodes(u):
                if v != self.parent[u]:
                    ndp[v] = max(0, ndp[u] - max(0, dp[v])) + dp[v]
                    stack.append(v)
        return ndp

    # class Graph(WeightedTree):
    def reroot_dp_for_tree_dis_with_node_weights(self, weights):
        dp = [0] * self.n
        sub = weights[:]
        s = sum(weights)
        self.parent = [-1] * self.n
        stack = [self.root]
        while stack:
            u = stack.pop()
            if u >= 0:
                stack.append(~u)
                for v in self.get_to_nodes(u):
                    if v != self.parent[u]:
                        self.parent[v] = u
                        stack.append(v)
            else:
                u = ~u
                for v in self.get_to_nodes(u):
                    if v != self.parent[u]:
                        sub[u] += sub[v]
                        dp[u] += dp[v] + sub[v]
        stack = [0]
        while stack:
            u = stack.pop()
            for v in self.get_to_nodes(u):
                if v != self.parent[u]:
                    dp[v] = dp[u] - sub[v] + s - sub[v]
                    stack.append(v)
        return dp

    # class Graph(WeightedTree):
    def reroot_dp_for_tree_dis_with_node_weights_maximum(self, weights):
        largest = [0] * self.n
        second = [0] * self.n
        self.parent = [-1] * self.n
        stack = [self.root]
        while stack:
            u = stack.pop()
            if u >= 0:
                stack.append(~u)
                for v in self.get_to_nodes(u):
                    if v != self.parent[u]:
                        self.parent[v] = u
                        stack.append(v)
            else:
                u = ~u
                a = b = 0
                for v in self.get_to_nodes(u):
                    if v != self.parent[u]:
                        x = largest[v] + weights[v]
                        if x >= a:
                            a, b = x, a
                        elif x >= b:
                            b = x
                largest[u] = a
                second[u] = b

        stack = [0]
        up = [0]
        ans = largest[:]
        while stack:
            u = stack.pop()
            d = up.pop()
            ans[u] = max(ans[u], d)
            for v in self.get_to_nodes(u):
                if v != self.parent[u]:
                    nex = d
                    x = largest[v] + weights[v]
                    a, b = largest[u], second[u]
                    # distance from current child nodes excluded
                    if x == a:
                        nex = nex if nex > b else b
                    else:
                        nex = nex if nex > a else a
                    up.append(nex + weights[u])
                    stack.append(v)
        return ans

    # class Graph(WeightedTree):
    def diff_array_with_edge(self):
        # differential summation from bottom to top
        stack = [self.root]
        diff = [0] * self.n
        while stack:
            u = stack.pop()
            if u >= 0:
                stack.append(~u)
                for v in self.get_to_nodes(u):
                    if v != self.parent[u]:
                        stack.append(v)
            else:
                u = ~u
                for v in self.get_to_nodes(u):
                    if v != self.parent[u]:
                        diff[u] += diff[v]
        return
