class UnWeightedTree:
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

class DFS:
    def __init__(self):
        return

    @staticmethod
    def gen_bfs_order_iteration(dct, root=0):
        """template of dfs order for rooted tree"""
        n = len(dct)
        for i in range(n):
            # visit from small to large according to the number of child nodes
            dct[i].sort(reverse=True)  # which is not necessary

        order = 0
        # index is original node value is dfs order
        start = [-1] * n
        # index is original node value is the maximum subtree dfs order
        end = [-1] * n
        # index is original node and value is its parent
        parent = [-1] * n
        stack = [root]
        # depth of every original node
        depth = [0] * n
        # index is dfs order and value is original node
        order_to_node = [-1] * n
        while stack:
            i = stack.pop()
            if i >= 0:
                start[i] = order
                order_to_node[order] = i
                end[i] = order
                order += 1
                stack.append(~i)
                for j in dct[i]:
                    # the order of son nodes can be assigned for lexicographical order
                    if j != parent[i]:
                        parent[j] = i
                        depth[j] = depth[i] + 1
                        stack.append(j)
            else:
                i = ~i
                if parent[i] != -1:
                    end[parent[i]] = end[i]

        return start, end


class DfsEulerOrder:
    def __init__(self, dct, root=0):
        """dfs and euler order of rooted tree which can be used for online point update and query subtree sum"""
        n = len(dct)
        for i in range(n):
            # visit from small to large according to the number of child nodes
            dct[i].sort(reverse=True)  # which is not necessary
        # index is original node value is dfs order
        self.start = [-1] * n
        # index is original node value is the maximum subtree dfs order
        self.end = [-1] * n
        # index is original node and value is its parent
        self.parent = [-1] * n
        # index is dfs order and value is original node
        self.order_to_node = [-1] * n
        # index is original node and value is its depth
        self.node_depth = [0] * n
        # index is dfs order and value is its depth
        self.order_depth = [0] * n
        # the order of original node visited in the total backtracking path
        self.euler_order = []
        # the pos of original node first appears in the euler order
        self.euler_in = [-1] * n
        # the pos of original node last appears in the euler order
        self.euler_out = [-1] * n  # 每个原始节点再欧拉序中最后出现的位置
        self.build(dct, root)
        return

    def build(self, dct, root):
        """build dfs order and euler order and relative info"""
        order = 0
        stack = [(root, -1)]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                self.euler_order.append(i)
                self.start[i] = order
                self.order_to_node[order] = i
                self.end[i] = order
                self.order_depth[order] = self.node_depth[i]
                order += 1
                stack.append((~i, fa))
                for j in dct[i]:
                    if j != fa:
                        # the order of son nodes can be assigned for lexicographical order
                        self.parent[j] = i
                        self.node_depth[j] = self.node_depth[i] + 1
                        stack.append((j, i))
            else:
                i = ~i
                if i != root:
                    self.euler_order.append(self.parent[i])
                if self.parent[i] != -1:
                    self.end[self.parent[i]] = self.end[i]
        for i, num in enumerate(self.euler_order):
            # pos of euler order for every original node
            self.euler_out[num] = i
            if self.euler_in[num] == -1:
                self.euler_in[num] = i
        return
