import math
import unittest
from collections import deque
from typing import List

from src.data_structure.tree_array import RangeAddRangeSum
from utils.fast_io import FastIO, inf




class UnionFindLCA:
    def __init__(self, n: int) -> None:
        self.root = [i for i in range(n)]
        self.order = [0] * n
        return

    def find(self, x: int) -> int:
        lst = []
        while x != self.root[x]:
            lst.append(x)
            # 在查询的时候合并到顺带直接根节点
            x = self.root[x]
        for w in lst:
            self.root[w] = x
        return x

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.order[root_x] < self.order[root_y]:
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        return True


class OfflineLCA:
    def __init__(self):
        return

    @staticmethod
    def bfs_iteration(dct: List[List[int]], queries: List[List[int]], root=0) -> List[int]:

        # 模板：离线查询LCA
        n = len(dct)
        ans = [dict() for _ in range(n)]
        for i, j in queries:  # 需要查询的节点对
            ans[i][j] = -1
            ans[j][i] = -1
        ind = 1
        stack = [root]  # 使用栈记录访问过程
        visit = [0] * n  # 访问状态数组 0 为未访问 1 为访问但没有遍历完子树 2 为访问且遍历完子树
        parent = [-1] * n  # 父节点
        uf = UnionFindLCA(n)
        depth = [0] * n
        while stack:
            i = stack.pop()
            if i >= 0:
                uf.order[i] = ind  # dfs序
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

        # 模板：离线查询LCA
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


class TreeDiffArray:

    # 模板：树上差分、点差分、边差分
    def __init__(self):
        return

    @staticmethod
    def bfs_iteration(dct: List[List[int]], queries: List[List[int]], root=0) -> List[int]:
        n = len(dct)
        stack = [root]
        parent = [-1] * n
        while stack:
            i = stack.pop()
            for j in dct[i]:
                if j != parent[i]:
                    stack.append(j)
                    parent[j] = i

        # 进行点差分计数
        diff = [0] * n
        for u, v, ancestor in queries:
            # 将 u 与 c 到 ancestor 的路径经过的节点进行差分修改
            diff[u] += 1
            diff[v] += 1
            diff[ancestor] -= 1
            if parent[ancestor] != -1:
                diff[parent[ancestor]] -= 1

        # 自底向上进行差分加和
        stack = [root]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in dct[i]:
                    if j != parent[i]:
                        stack.append(j)
            else:
                i = ~i
                for j in dct[i]:
                    if j != parent[i]:
                        diff[i] += diff[j]
        return diff

    @staticmethod
    def bfs_iteration_edge(dct: List[List[int]], queries: List[List[int]], root=0) -> List[int]:
        # 模板：树上边差分计算，将边的计数下放到对应的子节点
        n = len(dct)
        stack = [root]
        parent = [-1] * n
        while stack:
            i = stack.pop()
            for j in dct[i]:
                if j != parent[i]:
                    stack.append(j)
                    parent[j] = i

        # 进行边差分计数
        diff = [0] * n
        for u, v, ancestor in queries:
            # 将 u 与 v 到 ancestor 的路径的边下放到节点
            diff[u] += 1
            diff[v] += 1
            diff[ancestor] -= 2

        # 自底向上进行边差分加和
        stack = [[root, 1]]
        while stack:
            i, state = stack.pop()
            if state:
                stack.append([i, 0])
                for j in dct[i]:
                    if j != parent[i]:
                        stack.append([j, 1])
            else:
                for j in dct[i]:
                    if j != parent[i]:
                        diff[i] += diff[j]
        return diff

    @staticmethod
    def dfs_recursion(dct: List[List[int]], queries: List[List[int]], root=0) -> List[int]:
        n = len(dct)

        stack = [root]
        parent = [-1] * n
        while stack:
            i = stack.pop()
            for j in dct[i]:
                if j != parent[i]:
                    stack.append(j)
                    parent[j] = i

        # 进行差分计数
        diff = [0] * n
        for u, v, ancestor in queries:
            diff[u] += 1
            diff[v] += 1
            diff[ancestor] -= 1
            if parent[ancestor] != -1:
                diff[parent[ancestor]] -= 1

        def dfs(x, fa):
            for y in dct[x]:
                if y != fa:
                    diff[x] += dfs(y, x)
            return diff[x]

        dfs(0, -1)
        return diff


class TreeAncestorPool:

    def __init__(self, edges: List[List[int]], weight):
        # 以 0 为根节点
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

        # 根据节点规模设置层数
        self.cols = max(2, math.ceil(math.log2(n)))
        self.dp = [[-1] * self.cols for _ in range(n)]
        self.weight = [[inf] * self.cols for _ in range(n)]
        for i in range(n):
            # weight维护向上积累的水量和
            self.dp[i][0] = self.parent[i]
            self.weight[i][0] = weight[i]

        # 动态规划设置祖先初始化, dp[node][j] 表示 node 往前推第 2^j 个祖先
        for j in range(1, self.cols):
            for i in range(n):
                father = self.dp[i][j - 1]
                if father != -1:
                    self.dp[i][j] = self.dp[father][j - 1]
                    self.weight[i][j] = self.weight[father][j - 1] + self.weight[i][j - 1]
        return

    def get_final_ancestor(self, node: int, v: int) -> int:
        # 倍增查询水最后流入的水池
        for i in range(self.cols - 1, -1, -1):
            if v > self.weight[node][i]:
                v -= self.weight[node][i]
                node = self.dp[node][i]
        return node


class TreeAncestor:

    def __init__(self, edges: List[List[int]]):
        # 默认以 0 为根节点
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

        # 根据节点规模设置层数
        self.cols = max(2, math.ceil(math.log2(n)))
        self.dp = [[-1] * self.cols for _ in range(n)]
        for i in range(n):
            self.dp[i][0] = self.parent[i]
        # 动态规划设置祖先初始化, dp[node][j] 表示 node 往前推第 2^j 个祖先
        for j in range(1, self.cols):
            for i in range(n):
                father = self.dp[i][j-1]
                if father != -1:
                    self.dp[i][j] = self.dp[father][j-1]
        return

    def get_kth_ancestor(self, node: int, k: int) -> int:
        # 查询节点的第 k 个祖先
        for i in range(self.cols - 1, -1, -1):
            if k & (1 << i):
                node = self.dp[node][i]
                if node == -1:
                    break
        return node

    def get_lca(self, x: int, y: int) -> int:
        # 计算任意两点的最近公共祖先 LCA
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        while self.depth[x] > self.depth[y]:
            d = self.depth[x]-self.depth[y]
            x = self.dp[x][int(math.log2(d))]
        if x == y:
            return x
        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.dp[x][k] != self.dp[y][k]:
                x = self.dp[x][k]
                y = self.dp[y][k]
        return self.dp[x][0]

    def get_dist(self, u: int, v: int) -> int:
        # 计算任意点的最短路距离
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
        # 模板：将有根数进行质心递归划分
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
                        if v == parent[u] or is_removed[v]: continue
                        parent[v] = u
                        stack.append(v)
                for u in dfs_order[::-1]:
                    if u == root: break
                    size[parent[u]] += size[u]
            c = root
            while 1:
                mx, u = size[root] // 2, -1
                for v in to[c]:
                    if v == parent[c] or is_removed[v]:
                        continue
                    if size[v] > mx: mx, u = size[v], v
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
        # 树的质心数组，质心对应的父节点，以及质心作为根的子树规模
        return centroids, pre_cent, subtree_size


class HeavyChain:
    def __init__(self, dct: List[List[int]], root=0) -> None:
        # 模板：对树进行重链剖分
        self.n = len(dct)
        self.dct = dct
        self.parent = [-1]*self.n  # 父节点
        self.cnt_son = [1]*self.n  # 子树节点数
        self.weight_son = [-1]*self.n  # 重孩子
        self.top = [-1]*self.n  # 链式前向星
        self.dfn = [0]*self.n  # 节点对应的深搜序
        self.rev_dfn = [0]*self.n  # 深搜序对应的节点
        self.depth = [0]*self.n  # 节点的深度信息
        # 初始化
        self.build_weight(root)
        self.build_dfs(root)
        return

    def build_weight(self, root) -> None:
        # 生成树的重孩子信息
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
        # 生成树的深搜序信息
        stack = [[root, root]]
        order = 0
        while stack:
            i, tp = stack.pop()
            self.dfn[i] = order
            self.rev_dfn[order] = i
            self.top[i] = tp
            order += 1
            # 先访问重孩子再访问轻孩子
            w = self.weight_son[i]
            for j in self.dct[i]:
                if j != self.parent[i] and j != w:
                    stack.append([j, j])
            if w != -1:
                stack.append([w, tp])
        return

    def query_chain(self, x, y):
        # 查询 x 到 y 的最短路径经过的链路段
        lst = []
        while self.top[x] != self.top[y]:
            if self.depth[self.top[x]] < self.depth[self.top[y]]:
                x, y = y, x
            lst.append([self.dfn[self.top[x]], self.dfn[x]])
            x = self.parent[self.top[x]]
        a, b = self.dfn[x], self.dfn[y]
        if a > b:
            a, b = b, a
        lst.append([a, b])  # 返回的路径是链条的 dfs 序即 dfn
        return lst

    def query_lca(self, x, y):
        # 查询 x 与 y 的 LCA 最近公共祖先
        while self.top[x] != self.top[y]:
            if self.depth[self.top[x]] < self.depth[self.top[y]]:
                x, y = y, x
            x = self.parent[self.top[x]]
        # 返回的是节点真实的编号而不是 dfs 序即 dfn
        return x if self.depth[x] < self.depth[y] else y


