import math
from collections import deque
from math import inf
from typing import List


class UnionFindLCA:
    def __init__(self, n: int) -> None:
        self.root = [i for i in range(n)]
        self.order = [0] * n
        return

    def find(self, x: int) -> int:
        lst = []
        while x != self.root[x]:
            lst.append(x)
            x = self.root[x]
        for w in lst:
            self.root[w] = x
        return x

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        # union to the smaller dfs order
        if self.order[root_x] < self.order[root_y]:
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        return True


class OfflineLCA:
    def __init__(self):
        return

    @staticmethod
    def bfs_iteration(dct: List[List[int]], queries: List[List[int]], root=0) -> List[int]:
        """Offline query of LCA"""
        n = len(dct)
        ans = [dict() for _ in range(n)]
        for i, j in queries:  # Node pairs that need to be queried
            ans[i][j] = -1
            ans[j][i] = -1
        ind = 1
        stack = [root]
        # 0 is not visited
        # 1 is visited but not visit all its subtree nodes
        # 2 is visited included all its subtree nodes
        visit = [0] * n
        parent = [-1] * n
        uf = UnionFindLCA(n)
        depth = [0] * n
        while stack:
            i = stack.pop()
            if i >= 0:
                uf.order[i] = ind  # dfs order
                ind += 1
                visit[i] = 1
                stack.append(~i)
                for j in dct[i]:
                    if j != parent[i]:
                        parent[j] = i
                        depth[j] = depth[i] + 1
                        stack.append(j)
                for y in ans[i]:
                    if visit[y] == 1:
                        ans[y][i] = ans[i][y] = y
                    else:
                        ans[y][i] = ans[i][y] = uf.find(y)
            else:
                i = ~i
                visit[i] = 2
                uf.union(i, parent[i])

        return [ans[i][j] for i, j in queries]

    @staticmethod
    def dfs_recursion(dct: List[List[int]], queries: List[List[int]], root=0) -> List[int]:

        n = len(dct)
        ans = [dict() for _ in range(n)]
        for i, j in queries:
            ans[i][j] = -1
            ans[j][i] = -1

        def dfs(x, fa):
            nonlocal ind
            visit[x] = 1
            uf.order[x] = ind
            ind += 1

            for y in ans[x]:
                if visit[y] == 1:
                    ans[x][y] = ans[y][x] = y
                elif visit[y] == 2:
                    ans[x][y] = ans[y][x] = uf.find(y)
            for y in dct[x]:
                if y != fa:
                    dfs(y, x)
            visit[x] = 2
            uf.union(x, fa)
            return

        uf = UnionFindLCA(n)
        ind = 1
        visit = [0] * n
        dfs(root, -1)
        return [ans[i][j] for i, j in queries]


class TreeAncestorPool:

    def __init__(self, edges: List[List[int]], weight):
        # node 0 as root
        n = len(edges)
        self.n = n
        self.parent = [-1] * n
        self.depth = [-1] * n
        stack = deque([0])
        self.depth[0] = 0
        while stack:
            i = stack.popleft()
            for j in edges[i]:
                if self.depth[j] == -1:
                    self.depth[j] = self.depth[i] + 1
                    self.parent[j] = i
                    stack.append(j)

        # Set the number of layers based on the node size
        self.cols = max(2, math.ceil(math.log2(n)))
        self.dp = [[-1] * self.cols for _ in range(n)]
        self.weight = [[inf] * self.cols for _ in range(n)]
        for i in range(n):
            # the amount of water accumulated during weight maintenance
            self.dp[i][0] = self.parent[i]
            self.weight[i][0] = weight[i]

        for j in range(1, self.cols):
            for i in range(n):
                father = self.dp[i][j - 1]
                if father != -1:
                    self.dp[i][j] = self.dp[father][j - 1]
                    self.weight[i][j] = self.weight[father][j - 1] + self.weight[i][j - 1]
        return

    def get_final_ancestor(self, node: int, v: int) -> int:
        # query the final inflow of water into the pool with Multiplication method
        for i in range(self.cols - 1, -1, -1):
            if v > self.weight[node][i]:
                v -= self.weight[node][i]
                node = self.dp[node][i]
        return node


class TreeAncestor:

    def __init__(self, edges: List[List[int]]):
        n = len(edges)
        self.parent = [-1] * n
        self.depth = [-1] * n
        stack = deque([0])
        self.depth[0] = 0
        while stack:
            i = stack.popleft()
            for j in edges[i]:
                if self.depth[j] == -1:
                    self.depth[j] = self.depth[i] + 1
                    self.parent[j] = i
                    stack.append(j)

        self.cols = max(2, math.ceil(math.log2(n)))
        self.dp = [[-1] * self.cols for _ in range(n)]
        for i in range(n):
            self.dp[i][0] = self.parent[i]

        for j in range(1, self.cols):
            for i in range(n):
                father = self.dp[i][j - 1]
                if father != -1:
                    self.dp[i][j] = self.dp[father][j - 1]
        return

    def get_kth_ancestor(self, node: int, k: int) -> int:
        for i in range(self.cols - 1, -1, -1):
            if k & (1 << i):
                node = self.dp[node][i]
                if node == -1:
                    break
        return node

    def get_lca(self, x: int, y: int) -> int:
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            x = self.dp[x][int(math.log2(d))]
        if x == y:
            return x
        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.dp[x][k] != self.dp[y][k]:
                x = self.dp[x][k]
                y = self.dp[y][k]
        return self.dp[x][0]

    def get_dist(self, u: int, v: int) -> int:
        lca = self.get_lca(u, v)
        depth_u = self.depth[u]
        depth_v = self.depth[v]
        depth_lca = self.depth[lca]
        return depth_u + depth_v - 2 * depth_lca


class TreeCentroid:
    def __init__(self):
        return

    @staticmethod
    def centroid_finder(to, root=0):
        # recursive centroid partitioning of rooted trees
        centroids = []
        pre_cent = []
        subtree_size = []
        n = len(to)
        roots = [(root, -1, 1)]
        size = [1] * n
        is_removed = [0] * n
        parent = [-1] * n
        while roots:
            root, pc, update = roots.pop()
            parent[root] = -1
            if update:
                stack = [root]
                dfs_order = []
                while stack:
                    u = stack.pop()
                    size[u] = 1
                    dfs_order.append(u)
                    for v in to[u]:
                        if v == parent[u] or is_removed[v]:
                            continue
                        parent[v] = u
                        stack.append(v)
                for u in dfs_order[::-1]:
                    if u == root:
                        break
                    size[parent[u]] += size[u]
            c = root
            while 1:
                mx, u = size[root] // 2, -1
                for v in to[c]:
                    if v == parent[c] or is_removed[v]:
                        continue
                    if size[v] > mx:
                        mx, u = size[v], v
                if u == -1:
                    break
                c = u
            centroids.append(c)
            pre_cent.append(pc)
            subtree_size.append(size[root])
            is_removed[c] = 1
            for v in to[c]:
                if is_removed[v]:
                    continue
                roots.append((v, c, v == parent[c]))
        # the centroid array of the tree
        # the parent node corresponding to the centroid
        # the size of the subtree with the centroid as the root
        return centroids, pre_cent, subtree_size


class HeavyChain:
    def __init__(self, dct: List[List[int]], root=0) -> None:
        self.n = len(dct)
        self.dct = dct
        # father of node
        self.parent = [-1] * self.n
        # number of subtree nodes
        self.cnt_son = [1] * self.n
        # heavy son
        self.weight_son = [-1] * self.n
        # chain forward star
        self.top = [-1] * self.n
        # index is original node and value is its dfs order
        self.dfn = [0] * self.n
        # index is dfs order and value is its original node
        self.rev_dfn = [0] * self.n
        self.depth = [0] * self.n
        self.build_weight(root)
        self.build_dfs(root)
        return

    def build_weight(self, root) -> None:
        # get the info of heavy children of the tree
        stack = [root]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in self.dct[i]:
                    if j != self.parent[i]:
                        stack.append(j)
                        self.parent[j] = i
                        self.depth[j] = self.depth[i] + 1
            else:
                i = ~i
                for j in self.dct[i]:
                    if j != self.parent[i]:
                        self.cnt_son[i] += self.cnt_son[j]
                        if self.weight_son[i] == -1 or self.cnt_son[j] > self.cnt_son[self.weight_son[i]]:
                            self.weight_son[i] = j
        return

    def build_dfs(self, root) -> None:
        # get the info of dfs order
        stack = [[root, root]]
        order = 0
        while stack:
            i, tp = stack.pop()
            self.dfn[i] = order
            self.rev_dfn[order] = i
            self.top[i] = tp
            order += 1
            # visit heavy children first, then visit light children
            w = self.weight_son[i]
            for j in self.dct[i]:
                if j != self.parent[i] and j != w:
                    stack.append([j, j])
            if w != -1:
                stack.append([w, tp])
        return

    def query_chain(self, x, y):
        # Query the shortest path from x to y that passes through the chain segment
        lst = []
        while self.top[x] != self.top[y]:
            if self.depth[self.top[x]] < self.depth[self.top[y]]:
                x, y = y, x
            lst.append([self.dfn[self.top[x]], self.dfn[x]])
            x = self.parent[self.top[x]]
        a, b = self.dfn[x], self.dfn[y]
        if a > b:
            a, b = b, a
        # The returned path is the dfs order of the chain, also known as dfn
        lst.append([a, b])
        return lst

    def query_lca(self, x, y):
        # Query the LCA nearest common ancestor of x and y
        while self.top[x] != self.top[y]:
            if self.depth[self.top[x]] < self.depth[self.top[y]]:
                x, y = y, x
            x = self.parent[self.top[x]]
        # What is returned is the actual number of the node, not the dfn
        return x if self.depth[x] < self.depth[y] else y
