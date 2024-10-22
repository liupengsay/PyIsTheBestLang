class WeightedTreeForDP:
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

    # class Graph(WeightedTreeForDP):
    def heuristic_merge(self):
        dp = [0] * self.n
        sub = [None] * self.n
        nums = list(range(self.n))
        cnt = [0] * self.n
        self.parent = [-1] * self.n
        stack = [0]
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

    # class Graph(WeightedTreeForDP):
    def tree_dp(self):
        dp = [0] * self.n
        self.parent = [-1] * self.n
        stack = [0]
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

    # class Graph(WeightedTreeForDP):
    def reroot_dp(self):
        nums = [0] * self.n
        dp = [0] * self.n
        self.parent = [-1] * self.n
        stack = [0]
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

    # class Graph(WeightedTreeForDP):
    def reroot_dp_for_tree_dis_with_node_weights(self, weights):
        dp = [0] * self.n
        sub = weights[:]
        s = sum(weights)
        self.parent = [-1] * self.n
        stack = [0]
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

    # class Graph(WeightedTreeForDP):
    def reroot_dp_for_tree_dis_with_node_weights_maximum(self, weights):
        largest = [0] * self.n
        second = [0] * self.n
        self.parent = [-1] * self.n
        stack = [0]
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
