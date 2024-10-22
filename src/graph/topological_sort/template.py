import math


class WeightedGraphForTopologicalSort:
    def __init__(self, n, inf=math.inf):
        self.n = n
        self.inf = inf
        self.point_head = [0] * self.n
        self.degree = [0] * self.n
        self.edge_weight = [0]
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.edge_id = 1
        return

    def add_directed_edge(self, i, j, w):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        self.edge_weight.append(w)
        self.edge_from.append(i)
        self.edge_to.append(j)
        self.edge_next.append(self.point_head[i])
        self.point_head[i] = self.edge_id
        self.degree[j] += 1
        self.edge_id += 1
        return

    def add_undirected_edge(self, i, j, w):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        self.add_directed_edge(i, j, w)
        self.add_directed_edge(j, i, w)
        return

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

    # class Graph(WeightedGraphForTopologicalSort):
    def dag_dp(self):
        dp = [0] * self.n
        stack = [u for u in range(self.n) if not self.degree[u]]
        res = -10 ** 9
        post = dp[:]
        while stack:
            nex = []
            for u in stack:
                for v in self.get_to_nodes(u):
                    self.degree[v] -= 1
                    res = max(res, post[u] - dp[v])
                    post[v] = max(post[v], post[u])
                    if not self.degree[v]:
                        nex.append(v)
            stack = nex
        return res

    def topological_sort_for_dag_dp_with_edge_weight(self, weights):
        ans = [0] * self.n
        stack = [i for i in range(self.n) if not self.degree[i]]
        for i in stack:
            ans[i] = weights[i]
        while stack:
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    self.degree[j] -= 1
                    ans[j] = max(ans[j], ans[i] + weights[j] + self.edge_weight[ind])
                    if not self.degree[j]:
                        nex.append(j)
                    ind = self.edge_next[ind]
            stack = nex
        return ans

    def topological_order(self):
        ans = []
        stack = [i for i in range(self.n) if not self.degree[i]]
        while stack:
            ans.extend(stack)
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    self.degree[j] -= 1
                    if not self.degree[j]:
                        nex.append(j)
                    ind = self.edge_next[ind]
            stack = nex
        return ans


class TopologicalSort:
    def __init__(self):
        return

    @staticmethod
    def get_rank(n, edges):
        dct = [list() for _ in range(n)]
        degree = [0] * n
        for i, j in edges:
            degree[j] += 1
            dct[i].append(j)
        stack = [i for i in range(n) if not degree[i]]
        visit = [-1] * n
        step = 0
        while stack:
            for i in stack:
                visit[i] = step
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
            step += 1
        return visit

    @staticmethod
    def count_dag_path(n, edges):
        # Calculate the number of paths in a directed acyclic connected graph
        edge = [[] for _ in range(n)]
        degree = [0] * n
        for i, j in edges:
            edge[i].append(j)
            degree[j] += 1
        cnt = [0] * n
        stack = [i for i in range(n) if not degree[i]]
        for x in stack:
            cnt[x] = 1
        while stack:
            nex = []
            for i in stack:
                for j in edge[i]:
                    degree[j] -= 1
                    cnt[j] += cnt[i]
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return cnt

    @staticmethod
    def is_topology_unique(dct, degree, n):
        # Determine whether it is unique while ensuring the existence of topological sorting
        ans = []
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            ans.extend(stack)
            if len(stack) > 1:
                return False
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return True

    @staticmethod
    def is_topology_loop(edge, degree, n):
        # using Topological Sorting to Determine the Existence of Rings in a Directed Graph
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            nex = []
            for i in stack:
                for j in edge[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return all(x == 0 for x in degree)

    @staticmethod
    def bfs_topologic_order(n, dct, degree):
        # topological sorting determines whether there are rings in a directed graph
        # while recording the topological order of nodes
        order = [0] * n
        stack = [i for i in range(n) if degree[i] == 0]
        ind = 0
        while stack:
            nex = []
            for i in stack:
                order[i] = ind
                ind += 1
                for j in dct[i]:
                    degree[j] -= 1
                    if degree[j] == 0:
                        nex.append(j)
            stack = nex[:]
        if any(d > 0 for d in degree):
            return []
        return order


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
