import unittest

from typing import List
import math
import unittest
from collections import deque
from typing import List

from algorithm.src.fast_io import FastIO
from algorithm.src.graph.union_find import UnionFind

"""

算法：LCA、倍增算法、树链剖分、树的质心
功能：来求一棵树的最近公共祖先（LCA）也可以使用
题目：

===================================力扣===================================
1483. 树节点的第 K 个祖先（https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/）动态规划与二进制跳转维护祖先信息，类似ST表的思想与树状数组的思想

===================================洛谷===================================
P3379 【模板】最近公共祖先（LCA）（https://www.luogu.com.cn/problem/P3379）最近公共祖先模板题
P7128 「RdOI R1」序列(sequence)（https://www.luogu.com.cn/problem/P7128）完全二叉树进行LCA路径模拟交换，使得数组有序

==================================LibreOJ==================================
#10135. 「一本通 4.4 练习 2」祖孙询问（https://loj.ac/p/10135）lca查询与判断

================================CodeForces================================
E. Tree Queries（https://codeforces.com/problemset/problem/1328/E）利用 LCA 判定节点组是否符合条件，也可以使用 dfs 序
C. Ciel the Commander（https://codeforces.com/problemset/problem/321/C）使用树的质心递归，依次切割形成平衡树赋值
E. Minimum spanning tree for each edge（https://codeforces.com/problemset/problem/609/E）使用LCA的思想维护树中任意两点的路径边权最大值，并贪心替换获得边作为最小生成树时的最小权值和，有点类似于关键边与非关键边，但二者并不相同

参考：
CSDN（https://blog.csdn.net/weixin_42001089/article/details/83590686）

"""


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


class TreeAncestorWeight:

    def __init__(self, edges: List[List[int]], dct):
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
        self.cols = FastIO().max(2, math.ceil(math.log2(n)))
        self.dp = [[-1] * self.cols for _ in range(n)]
        self.weight = [[0] * self.cols for _ in range(n)]
        for i in range(n):
            self.dp[i][0] = self.parent[i]
            self.weight[i][0] = dct[(self.parent[i], i)]
        # 动态规划设置祖先初始化, dp[node][j] 表示 node 往前推第 2^j 个祖先
        for j in range(1, self.cols):
            for i in range(n):
                father = self.dp[i][j - 1]
                pre = self.weight[i][j - 1]
                if father != -1:
                    self.dp[i][j] = self.dp[father][j - 1]
                    self.weight[i][j] = FastIO().max(self.weight[father][j - 1], pre)

        return

    def get_dist_weight_max(self, x: int, y: int) -> int:
        # 计算任意点的最短路上的权重最大值
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        ans = 0
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            ans = FastIO().max(ans, self.weight[x][int(math.log2(d))])
            x = self.dp[x][int(math.log2(d))]
        if x == y:
            return ans

        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.dp[x][k] != self.dp[y][k]:
                ans = FastIO().max(ans, self.weight[x][k])
                ans = FastIO().max(ans, self.weight[y][k])
                x = self.dp[x][k]
                y = self.dp[y][k]
        ans = FastIO().max(ans, self.weight[x][0])
        return FastIO().max(ans, self.weight[y][0])


class Solution:
    def __init__(self):
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
    def lg_p3379(ac=FastIO()):
        # 模板：查询任意两个节点的 LCA
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
    def cf_609e(ac=FastIO()):
        # 模板：计算最小生成树有指定边参与时的最小权值和
        n, m = ac.read_ints()
        edge = [[] for _ in range(n)]
        lst = []
        dct = dict()
        for _ in range(m):
            a, b, c = ac.read_ints()
            dct[(a - 1, b - 1)] = dct[(b - 1, a - 1)] = c
            lst.append([a - 1, b - 1, c])
        dct[(-1, 0)] = 0
        dct[(0, -1)] = 0

        uf = UnionFind(n)
        cost = 0

        for a, b, c in sorted(lst, key=lambda it: it[-1]):
            if not uf.is_connected(a, b):
                edge[a].append(b)
                edge[b].append(a)
                uf.union(a, b)
                cost += c
        lca = TreeAncestorWeight(edge, dct)
        for a, b, c in lst:
            ac.st(cost + c - lca.get_dist_weight_max(a, b))
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
