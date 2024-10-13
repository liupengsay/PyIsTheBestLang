import math
from collections import deque
from typing import List




class UnionFindGetLCA:
    def __init__(self, parent, root=0):
        n = len(parent)
        self.root_or_size = [-1] * n
        self.edge = [0] * n
        self.order = [0] * n
        assert parent[root] == root

        out_degree = [0] * n
        que = deque()
        for i in range(n):
            out_degree[parent[i]] += 1
        for i in range(n):
            if out_degree[i] == 0:
                que.append(i)

        for i in range(n - 1):
            v = que.popleft()
            fa = parent[v]
            x, y = self.union(v, fa)
            self.edge[y] = fa
            self.order[y] = i
            out_degree[fa] -= 1
            if out_degree[fa] == 0:
                que.append(fa)

        self.order[self.find(root)] = n
        return

    def union(self, v, fa):
        x, y = self.find(v), self.find(fa)
        if self.root_or_size[x] > self.root_or_size[y]:
            x, y = y, x
        self.root_or_size[x] += self.root_or_size[y]
        self.root_or_size[y] = x
        return x, y

    def find(self, v):
        while self.root_or_size[v] >= 0:
            v = self.root_or_size[v]
        return v

    def get_lca(self, u, v):
        lca = v
        while u != v:
            if self.order[u] < self.order[v]:
                u, v = v, u
            lca = self.edge[v]
            v = self.root_or_size[v]
        return lca


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
    def bfs_iteration(dct, queries, root=0):
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


class TreeAncestorPool:

    def __init__(self, edges, weight):
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
        self.weight = [[math.inf] * self.cols for _ in range(n)]
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

    def __init__(self, edges, root=0):
        n = len(edges)
        self.parent = [-1] * n
        self.depth = [-1] * n
        stack = [root]
        self.depth[root] = 0
        while stack:
            i = stack.pop()
            for j in edges[i]:
                if self.depth[j] == -1:
                    self.depth[j] = self.depth[i] + 1  # can change to be weighted
                    self.parent[j] = i
                    stack.append(j)

        self.cols = max(2, math.ceil(math.log2(n)))
        self.dp = [-1] * self.cols * n
        for i in range(n):
            self.dp[i * self.cols] = self.parent[i]

        for j in range(1, self.cols):
            for i in range(n):
                father = self.dp[i * self.cols + j - 1]
                if father != -1:
                    self.dp[i * self.cols + j] = self.dp[father * self.cols + j - 1]
        return

    def get_kth_ancestor(self, node: int, k: int) -> int:
        for i in range(self.cols - 1, -1, -1):
            if k & (1 << i):
                node = self.dp[node * self.cols + i]
                if node == -1:
                    break
        return node

    def get_lca(self, x: int, y: int) -> int:
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            x = self.dp[x * self.cols + int(math.log2(d))]
        if x == y:
            return x
        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.dp[x * self.cols + k] != self.dp[y * self.cols + k]:
                x = self.dp[x * self.cols + k]
                y = self.dp[y * self.cols + k]
        return self.dp[x * self.cols]

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



class TreeAncestorMaxSubNode:
    def __init__(self, x):
        self.val = self.pref = self.suf = max(x, 0)
        self.sm = x
        pass



class TreeAncestorMaxSub:
    def __init__(self, edges: List[List[int]], values):
        n = len(edges)
        self.values = values
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
        self.dp = [-1] * self.cols * n
        self.weight = [TreeAncestorMaxSubNode(0)] * self.cols * n
        for i in range(n):
            self.dp[i * self.cols] = self.parent[i]
            if i:
                self.weight[i * self.cols] = TreeAncestorMaxSubNode(values[self.parent[i]])
            else:
                self.weight[i * self.cols] = TreeAncestorMaxSubNode(0)

        for j in range(1, self.cols):
            for i in range(n):
                father = self.dp[i * self.cols + j - 1]
                pre = self.weight[i * self.cols + j - 1]
                if father != -1:
                    self.dp[i * self.cols + j] = self.dp[father * self.cols + j - 1]
                    self.weight[i * self.cols + j] = self.merge(pre, self.weight[father * self.cols + j - 1])

        return

    @staticmethod
    def merge(a, b):
        ret = TreeAncestorMaxSubNode(0)
        ret.pref = max(a.pref, a.sm + b.pref)
        ret.suf = max(b.suf, b.sm + a.suf)
        ret.sm = a.sm + b.sm
        ret.val = max(a.val, b.val)
        ret.val = max(ret.val, a.suf + b.pref)
        return ret

    @staticmethod
    def reverse(a):
        a.suf, a.pref = a.pref, a.suf
        return a

    def get_ancestor_node_max(self, x: int, y: int) -> TreeAncestorMaxSubNode:
        ans = TreeAncestorMaxSubNode(self.values[x])
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            ans = self.merge(ans, self.weight[x * self.cols + int(math.log2(d))])
            x = self.dp[x * self.cols + int(math.log2(d))]
        return ans

    def get_kth_ancestor(self, node: int, k: int) -> int:
        for i in range(self.cols - 1, -1, -1):
            if k & (1 << i):
                node = self.dp[node * self.cols + i]
                if node == -1:
                    break
        return node

    def get_lca(self, x: int, y: int) -> int:
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            x = self.dp[x * self.cols + int(math.log2(d))]
        if x == y:
            return x
        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.dp[x * self.cols + k] != self.dp[y * self.cols + k]:
                x = self.dp[x * self.cols + k]
                y = self.dp[y * self.cols + k]
        return self.dp[x * self.cols]

    def get_max_con_sum(self, x: int, y: int) -> int:
        if x == y:
            return TreeAncestorMaxSubNode(self.values[x]).val
        z = self.get_lca(x, y)
        if z == x:
            ans = self.get_ancestor_node_max(y, x)
            return ans.val
        if z == y:
            ans = self.get_ancestor_node_max(x, y)
            return ans.val

        ax = self.get_kth_ancestor(x, self.depth[x] - self.depth[z])
        by = self.get_kth_ancestor(y, self.depth[y] - self.depth[z] - 1)
        a = self.get_ancestor_node_max(x, ax)
        b = self.get_ancestor_node_max(y, by)
        ans = self.merge(a, self.reverse(b))
        return ans.val

class HeavyChain:
    def __init__(self, dct, root=0) -> None:
        self.n = len(dct)
        self.dct = dct
        self.parent = [-1] * self.n  # father of node
        self.cnt_son = [1] * self.n  # number of subtree nodes
        self.weight_son = [-1] * self.n  # heavy son
        self.top = [-1] * self.n  # chain forward star
        self.dfn = [0] * self.n  # index is original node and value is its dfs order
        self.rev_dfn = [0] * self.n  # index is dfs order and value is its original node
        self.depth = [0] * self.n  # depth of node
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
        stack = [(root, root)]
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
                    stack.append((j, j))
            if w != -1:
                stack.append((w, tp))
        return

    def query_chain(self, x, y):
        # query the shortest path from x to y that passes through the chain segment
        pre = []
        post = []
        while self.top[x] != self.top[y]:
            if self.depth[self.top[x]] > self.depth[self.top[y]]:
                pre.append([self.dfn[x], self.dfn[self.top[x]]])
                x = self.parent[self.top[x]]
            else:
                post.append([self.dfn[self.top[y]], self.dfn[y]])
                y = self.parent[self.top[y]]

        a, b = self.dfn[x], self.dfn[y]
        pre.append([a, b])
        lca = a if a < b else b
        pre += post[::-1]
        return pre, lca

    def query_lca(self, x, y):
        # query the LCA nearest common ancestor of x and y
        while self.top[x] != self.top[y]:
            if self.depth[self.top[x]] < self.depth[self.top[y]]:
                x, y = y, x
            x = self.parent[self.top[x]]
        # returned value is the actual number of the node and not the dfn!!!
        return x if self.depth[x] < self.depth[y] else y