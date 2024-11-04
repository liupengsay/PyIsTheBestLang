class PruferAndTree:
    def __init__(self):
        return

    @staticmethod
    def adj_to_parent(adj, root):

        def dfs(v):
            for u in adj[v]:
                if u != parent[v]:
                    parent[u] = v
                    dfs(u)

        n = len(adj)
        parent = [-1] * n
        dfs(root)
        return parent

    @staticmethod
    def parent_to_adj(parent):
        n = len(parent)
        adj = [[] for _ in range(n)]
        for i in range(n):
            if parent[i] != -1:  # 即 i!=root
                adj[i].append(parent[i])
                adj[parent[i]].append(i)
        return parent

    def tree_to_prufer(self, adj, root):
        parent = self.adj_to_parent(adj, root)
        n = len(adj)
        ptr = -1
        degree = [0] * n
        for i in range(0, n):
            degree[i] = len(adj[i])
            if degree[i] == 1 and ptr == -1:
                ptr = i

        code = [0] * (n - 2)
        leaf = ptr
        for i in range(0, n - 2):
            nex = parent[leaf]
            code[i] = nex
            degree[nex] -= 1
            if degree[nex] == 1 and nex < ptr:
                leaf = nex
            else:
                ptr = ptr + 1
                while degree[ptr] != 1:
                    ptr = ptr + 1
                leaf = ptr
        return code

    @staticmethod
    def prufer_to_tree(code, root):
        n = len(code) + 2

        degree = [1] * n
        for i in code:
            degree[i] += 1
        ptr = 0
        while degree[ptr] != 1:
            ptr += 1
        leaf = ptr

        adj = [[] for _ in range(n)]
        for v in code:
            adj[v].append(leaf)
            adj[leaf].append(v)
            degree[v] -= 1
            if degree[v] == 1 and v < ptr and v != root:
                leaf = v
            else:
                ptr += 1
                while degree[ptr] != 1:
                    ptr += 1
                leaf = ptr

        adj[leaf].append(root)
        adj[root].append(leaf)
        for i in range(n):
            adj[i].sort()
        return adj
