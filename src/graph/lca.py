import math
import unittest
from collections import deque
from typing import List

from src.data_structure.tree_array import TreeArrayRangeSum
from src.fast_io import FastIO, inf

"""

算法：LCA、倍增算法、树链剖分、树的质心、树的重心、离线LCA与树上差分
功能：来求一棵树的最近公共祖先（LCA）也可以使用
题目：

===================================力扣===================================
1483. 树节点的第 K 个祖先（https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/）动态规划与二进制跳转维护祖先信息，类似ST表的思想与树状数组的思想，经典LCA应用题
2646. 最小化旅行的价格总和（https://leetcode.cn/problems/minimize-the-total-price-of-the-trips/）离线LCA与树上差分计数，再使用树形DP计算

===================================洛谷===================================
P3379 【模板】最近公共祖先（LCA）（https://www.luogu.com.cn/problem/P3379）最近公共祖先模板题
P7128 「RdOI R1」序列(sequence)（https://www.luogu.com.cn/problem/P7128）完全二叉树进行LCA路径模拟交换，使得数组有序
P3128 [USACO15DEC]Max Flow P（https://www.luogu.com.cn/problem/P3128）离线LCA与树上差分
P7167 [eJOI2020 Day1] Fountain（https://www.luogu.com.cn/problem/P7167）单调栈建树倍增在线LCA查询
P3384 【模板】重链剖分/树链剖分（https://www.luogu.com.cn/problem/P3384）树链剖分与树状数组模拟
P2912 [USACO08OCT]Pasture Walking G（https://www.luogu.com.cn/problem/P2912）离线LCA查询与任意点对之间距离计算
P3019 [USACO11MAR]Meeting Place S（https://www.luogu.com.cn/problem/P3019）离线查询 LCA 最近公共祖先
P3258 [JLOI2014]松鼠的新家（https://www.luogu.com.cn/problem/P3258）离线LCA加树上差分加树形DP
P6869 [COCI2019-2020#5] Putovanje（https://www.luogu.com.cn/problem/P6869）离线 LCA 查询与树上边差分计算

==================================LibreOJ==================================
#10135. 「一本通 4.4 练习 2」祖孙询问（https://loj.ac/p/10135）lca查询与判断

================================CodeForces================================
E. Tree Queries（https://codeforces.com/problemset/problem/1328/E）利用 LCA 判定节点组是否符合条件，也可以使用 dfs 序
C. Ciel the Commander（https://codeforces.com/problemset/problem/321/C）使用树的质心递归，依次切割形成平衡树赋值
E. A and B and Lecture Rooms（https://codeforces.com/problemset/problem/519/E）LCA经典运用题目，查询距离与第k个祖先节点，与子树节点计数

================================AcWing================================
4202. 穿过圆（https://www.acwing.com/problem/content/4205/）使用位运算进行计算，也可使用包含关系建树，查询LCA计算距离


参考：
CSDN（https://blog.csdn.net/weixin_42001089/article/details/83590686）

"""


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


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p7167(ac=FastIO()):

        # 模板：单调栈加倍增LCA计算
        n, q = ac.read_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        # 使用单调栈建树
        parent = [n] * n
        edge = [[] for _ in range(n + 1)]
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]][0] < nums[i][0]:
                parent[stack.pop()] = i
            stack.append(i)
        for i in range(n):
            edge[n - parent[i]].append(n - i)

        # LCA预处理
        weight = [x for _, x in nums] + [inf]
        tree = TreeAncestorPool(edge, weight[::-1])

        # 查询
        for _ in range(q):
            r, v = ac.read_ints()
            ans = tree.get_final_ancestor(n - r + 1, v)
            ac.st(0 if ans == 0 else n - ans + 1)
        return

    @staticmethod
    def cf_519e(ac=FastIO()):
        # 模板：使用LCA计算第k个祖先与节点之间的距离
        n = ac.read_int()
        edges = [[] for _ in range(n)]
        for _ in range(n-1):
            x, y = ac.read_ints_minus_one()
            edges[x].append(y)
            edges[y].append(x)

        lca = TreeAncestor(edges)
        sub = [0]*n

        @ac.bootstrap
        def dfs(i, fa):
            nonlocal sub
            cur = 1
            for j in edges[i]:
                if j != fa:
                    yield dfs(j, i)
                    cur += sub[j]
            sub[i] = cur
            yield
        dfs(0, -1)

        for _ in range(ac.read_int()):
            x, y = ac.read_ints_minus_one()
            if x == y:
                ac.st(n)
                continue

            dis = lca.get_dist(x, y)
            if dis % 2 == 1:
                ac.st(0)
            else:
                z = lca.get_lca(x, y)
                dis1 = lca.get_dist(x, z)
                dis2 = lca.get_dist(y, z)
                if dis1 == dis2:
                    up = n - sub[z]
                    down = sub[z] - sub[lca.get_kth_ancestor(x, dis1-1)]-sub[lca.get_kth_ancestor(y, dis2-1)]
                    ac.st(up+down)
                elif dis1 > dis2:
                    w = lca.get_kth_ancestor(x, (dis1+dis2)//2)
                    ac.st(sub[w] - sub[lca.get_kth_ancestor(x, (dis1+dis2)//2-1)])
                else:
                    w = lca.get_kth_ancestor(y, (dis1 + dis2) // 2)
                    ac.st(sub[w] - sub[lca.get_kth_ancestor(y, (dis1 + dis2) // 2 - 1)])
        return

    @staticmethod
    def cf_1328e(ac=FastIO()):
        # 模板：利用 LCA 的方式查询是否为一条链上距离不超过 1 的点
        n, m = ac.read_ints()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_ints_minus_one()
            edge[u].append(v)
            edge[v].append(u)

        tree = TreeAncestor(edge)
        for _ in range(m):
            nums = ac.read_list_ints_minus_one()[1:]
            deep = nums[0]
            for num in nums:
                if tree.depth[num] > tree.depth[deep]:
                    deep = num
            ans = True
            for num in nums:
                fa = tree.get_lca(num, deep)
                if fa == num or tree.parent[num] == fa:
                    continue
                else:
                    ans = False
                    break
            ac.st("YES" if ans else "NO")
        return

    @staticmethod
    def lc_1483(parent, node, k):
        # 模板：查询任意节点的第 k 个祖先
        n = len(parent)
        edges = [[] for _ in range(n)]
        for i in range(n):
            if parent[i] != -1:
                edges[i].append(parent[i])
                edges[parent[i]].append(i)
        tree = TreeAncestor(edges)
        return tree.get_kth_ancestor(node, k)

    @staticmethod
    def lg_p3379_1(ac=FastIO()):
        # 模板：使用倍增查询任意两个节点的 LCA
        n, m, s = ac.read_ints()
        s -= 1
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_ints_minus_one()
            edge[x].append(y)
            edge[y].append(x)
        # 需要改 s 为默认根
        tree = TreeAncestor(edge)
        for _ in range(m):
            x, y = ac.read_ints_minus_one()
            ac.st(tree.get_lca(x, y) + 1)
        return

    @staticmethod
    def cf_321c(ac=FastIO()):
        # 模板：使用质心算法进行树的递归切割
        n = ac.read_int()
        to = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            to[u].append(v)
            to[v].append(u)

        cc, pp, ss = TreeCentroid().centroid_finder(to)
        ans = [64] * n
        for c, p in zip(cc, pp):
            ans[c] = ans[p] + 1
        ac.lst([chr(x) for x in ans])
        return
    
    @staticmethod
    def lc_6738(n: int, edges: List[List[int]], price: List[int], trips: List[List[int]]) -> int:

        # 模板：离线LCA加树上差分加树形DP
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        # 离线LCA
        res = OfflineLCA().bfs_iteration(dct, trips)
        # res = OfflineLCA().dfs_recursion(dct, trips)   # 也可以使用递归

        # 树上差分
        m = len(trips)
        queries = [trips[i] + [res[i]] for i in range(m)]
        cnt = TreeDiffArray().bfs_iteration(dct, queries)
        # cnt = TreeDiffArray().dfs_recursion(dct, queries)  # 也可以使用递归

        # 迭代版的树形DP
        stack = [0]
        sub = [[] for _ in range(n)]
        parent = [-1] * n
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in dct[i]:
                    if j != parent[i]:
                        parent[j] = i
                        stack.append(j)
            else:
                i = ~i
                res = [cnt[i] * price[i], cnt[i] * price[i] // 2]
                for j in dct[i]:
                    if j != parent[i]:
                        a, b = sub[j]
                        res[0] += a if a < b else b
                        res[1] += a
                sub[i] = res

        return min(sub[0])

    @staticmethod
    def lg_p3128(ac=FastIO()):
        # 模板：离线LCA加树上差分
        n, k = ac.read_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n-1):
            i, j = ac.read_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        queries = [ac.read_list_ints_minus_one() for _ in range(k)]
        res = OfflineLCA().bfs_iteration(dct, queries)
        # res = OfflineLCA().dfs_recursion(dct, trips)  # 也可以使用递归
        queries = [queries[i] + [res[i]] for i in range(k)]
        cnt = TreeDiffArray().bfs_iteration(dct, queries)
        # cnt = TreeDiffArray().dfs_recursion(dct, queries)  # 也可以使用递归
        ac.st(max(cnt))
        return

    @staticmethod
    def lg_p3384(ac=FastIO()):
        # 模板：使用树链剖分和深搜序进行节点值修改与区间和查询
        n, m, r, p = ac.read_ints()
        r -= 1
        tree = TreeArrayRangeSum(n)
        nums = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n-1):
            i, j = ac.read_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        heavy = HeavyChain(dct, r)
        tree.build([nums[i] for i in heavy.rev_dfn])

        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, y, z = lst[1:]
                for a, b in heavy.query_chain(x-1, y-1):
                    tree.update_range(a+1, b+1, z)
            elif lst[0] == 2:
                x, y = lst[1:]
                ans = 0
                for a, b in heavy.query_chain(x - 1, y - 1):
                    ans += tree.get_sum_range(a + 1, b + 1)
                    ans %= p
                ac.st(ans)
            elif lst[0] == 3:
                x, z = lst[1:]
                x -= 1
                a, b = heavy.dfn[x], heavy.cnt_son[x]
                tree.update_range(a+1, a+b, z)
            else:
                x = lst[1] - 1
                a, b = heavy.dfn[x], heavy.cnt_son[x]
                ans = tree.get_sum_range(a + 1, a + b) % p
                ac.st(ans)
        return

    @staticmethod
    def lg_p3379_2(ac=FastIO()):
        # 模板：使用树链剖分求 LCA
        n, m, r = ac.read_ints()
        r -= 1
        dct = [[] for _ in range(n)]
        for _ in range(n-1):
            i, j = ac.read_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        heavy = HeavyChain(dct, r)
        for _ in range(m):
            x, y = ac.read_ints_minus_one()
            ac.st(heavy.query_lca(x, y) + 1)
        return

    @staticmethod
    def lg_p2912(ac=FastIO()):
        # 模板：离线LCA查询与任意点对之间距离计算
        n, q = ac.read_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(n - 1):
            i, j, w = ac.read_ints()
            i -= 1
            j -= 1
            dct[i][j] = dct[j][i] = w

        # 根节点bfs计算距离
        dis = [inf] * n
        dis[0] = 0
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            for j in dct[i]:
                if j != fa:
                    dis[j] = dis[i] + dct[i][j]
                    stack.append([j, i])

        # 查询公共祖先
        queries = [ac.read_list_ints_minus_one() for _ in range(q)]
        dct = [list(d.keys()) for d in dct]
        ancestor = OfflineLCA().bfs_iteration(dct, queries, 0)
        for x in range(q):
            i, j = queries[x]
            w = ancestor[x]
            ans = dis[i] + dis[j] - 2 * dis[w]
            ac.st(ans)
        return

    @staticmethod
    def lg_p3019(ac=FastIO()):
        # 模板：离线查询 LCA 最近公共祖先
        n, m = ac.read_ints()
        dct = [[] for _ in range(n)]
        for i in range(n - 1):
            dct[ac.read_int() - 1].append(i + 1)
        queries = [ac.read_list_ints_minus_one() for _ in range(m)]
        ans = OfflineLCA().bfs_iteration(dct, queries)
        for a in ans:
            ac.st(a + 1)
        return

    @staticmethod
    def lg_p3258(ac=FastIO()):
        # 模板：离线LCA加树上差分加树形DP
        n = ac.read_int()
        nums = ac.read_list_ints_minus_one()
        root = nums[0]
        dct = [[] for _ in range(n)]
        for _ in range(n-1):
            i, j = ac.read_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)

        n = len(dct)
        stack = [root]
        parent = [-1] * n
        while stack:
            i = stack.pop()
            for j in dct[i]:
                if j != parent[i]:
                    stack.append(j)
                    parent[j] = i

        trips = []
        for i in range(1, n):
            trips.append([nums[i-1], nums[i]])

        # 离线LCA
        res = OfflineLCA().bfs_iteration(dct, trips, root)
        # 树上差分
        diff = [0] * n
        for i in range(n-1):
            u, v, ancestor = trips[i] + [res[i]]
            # 将 u 与 v 到 ancestor 的路径经过的节点进行差分修改（不包含u）
            if u != ancestor:
                u = parent[u]
                diff[u] += 1
                diff[v] += 1
                diff[ancestor] -= 1
                if parent[ancestor] != -1:
                    diff[parent[ancestor]] -= 1
            else:
                diff[v] += 1
                diff[u] -= 1

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
        diff[nums[0]] += 1
        diff[nums[-1]] -= 1
        for a in diff:
            ac.st(a)
        return

    @staticmethod
    def lc_2646(n: int, edges: List[List[int]], price: List[int], trips: List[List[int]]) -> int:
        # 模板：离线LCA与树上差分计数，再使用树形 DP 计算
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)
        res = OfflineLCA().bfs_iteration(dct, trips)
        m = len(trips)
        queries = [trips[i] + [res[i]] for i in range(m)]
        cnt = TreeDiffArray().bfs_iteration(dct, queries)

        stack = [[0, 1]]
        sub = [[] for _ in range(n)]
        parent = [-1] * n
        while stack:
            i, state = stack.pop()
            if state:
                stack.append([i, 0])
                for j in dct[i]:
                    if j != parent[i]:
                        parent[j] = i
                        stack.append([j, 1])
            else:
                res = [cnt[i] * price[i], cnt[i] * price[i] // 2]
                for j in dct[i]:
                    if j != parent[i]:
                        a, b = sub[j]
                        res[0] += a if a < b else b
                        res[1] += a
                sub[i] = res

        return min(sub[0])

    @staticmethod
    def lg_p6969(ac=FastIO()):
        # 模板：离线 LCA 查询与树上边差分计算
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        cost = [dict() for _ in range(n)]
        for _ in range(n - 1):
            a, b, c1, c2 = ac.read_ints()
            a -= 1
            b -= 1
            cost[a][b] = cost[b][a] = [c1, c2]
            dct[a].append(b)
            dct[b].append(a)

        query = [[i, i + 1] for i in range(n - 1)]
        res = OfflineLCA().bfs_iteration(dct, query)
        for i in range(n - 1):
            query[i].append(res[i])
        diff = TreeDiffArray().bfs_iteration_edge(dct, query, 0)

        # 将变形的边差分还原
        ans = 0
        stack = [0]
        parent = [-1] * n
        while stack:
            i = stack.pop()
            for j in dct[i]:
                if j != parent[i]:
                    stack.append(j)
                    parent[j] = i
                    # 边计数下放到节点上
                    cnt = diff[j]
                    c1, c2 = cost[i][j]
                    ans += ac.min(cnt * c1, c2)
        ac.st(ans)
        return


class TestGeneral(unittest.TestCase):

    def test_tree_ancestor(self):
        parent = [-1, 0, 0, 1, 2]
        n = len(parent)
        edges = [[] for _ in range(n)]
        for i in range(n):
            if parent[i] != -1:
                edges[i].append(parent[i])
                edges[parent[i]].append(i)
        tree = TreeAncestor(edges)
        assert tree.get_kth_ancestor(4, 3) == -1
        assert tree.get_kth_ancestor(4, 2) == 0
        assert tree.get_kth_ancestor(4, 1) == 2
        assert tree.get_kth_ancestor(4, 0) == 4
        assert tree.get_lca(3, 4) == 0
        assert tree.get_lca(2, 4) == 2
        assert tree.get_lca(3, 1) == 1
        assert tree.get_lca(3, 2) == 0
        assert tree.get_dist(0, 0) == 0
        assert tree.get_dist(0, 4) == 2
        assert tree.get_dist(3, 4) == 4
        assert tree.get_dist(1, 0) == 1
        assert tree.get_dist(2, 3) == 3
        return


if __name__ == '__main__':
    unittest.main()
