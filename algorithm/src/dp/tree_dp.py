
import unittest
from collections import deque
from functools import lru_cache
from heapq import nlargest
from typing import List
from algorithm.src.fast_io import FastIO, inf
from algorithm.src.graph.union_find import UnionFind

"""
算法：树形DP、树的直径
功能：在树形或者图结构上进行DP，有换根DP，自顶向下和自底向上DP
题目：

===================================力扣===================================
2458 移除子树后的二叉树高度（https://leetcode.cn/problems/height-of-binary-tree-after-subtree-removal-queries/）跑两边DFS进行自顶向下和自底向上DP结合
2440 创建价值相同的连通块（https://leetcode.cn/problems/create-components-with-same-value/）利用总和的因子和树形递归判断连通块是否可行
1569 将子数组重新排序得到同一个二叉查找树的方案数（https://leetcode.cn/problems/number-of-ways-to-reorder-array-to-get-same-bst/solution/by-liupengsay-yi3h/）
968. 监控二叉树（https://leetcode.cn/problems/binary-tree-cameras/）树形DP监控每个节点
6294. 最大价值和与最小价值和的差值（https://leetcode.cn/problems/difference-between-maximum-and-minimum-price-sum/）树形换根DP，求去掉其中一个叶子节点的最大直径
124. 二叉树中的最大路径和（https://leetcode.cn/problems/binary-tree-maximum-path-sum/）树形DP
1617. 统计子树中城市之间最大距离（https://leetcode.cn/problems/count-subtrees-with-max-distance-between-cities/）二进制枚举加树的直径计算
2378. 选择边来最大化树的得分（https://leetcode.cn/problems/choose-edges-to-maximize-score-in-a-tree/）树形DP
2445. 值为 1 的节点数（https://leetcode.cn/problems/number-of-nodes-with-value-one/）自上而下DP模拟

===================================洛谷===================================

P1395 会议（https://leetcode.cn/problems/create-components-with-same-value/）树的总距离，单个节点距离其他所有节点的最大距离
P1352 没有上司的舞会（https://www.luogu.com.cn/problem/P1352）树形DP，隔层进行动态规划转移
P1922 女仆咖啡厅桌游吧（https://www.luogu.com.cn/problem/P1922）树形DP，贪心进行子树与叶子节点的分配
P2016 战略游戏（https://www.luogu.com.cn/problem/P2016）树形DP瞭望每条边
P1122 最大子树和（https://www.luogu.com.cn/problem/P1122）计算最大的连通块和
P2932 [USACO09JAN]Earthquake Damage G（https://www.luogu.com.cn/problem/P2932）树形DP统计子树个数与贪心安排最小损坏个数
P2996 [USACO10NOV]Visiting Cows G（https://www.luogu.com.cn/problem/P2996）树形DP
P3074 [USACO13FEB]Milk Scheduling S（https://www.luogu.com.cn/problem/P3074）树的最长路径（广搜DP记录最长时间也可以）
P3884 [JLOI2009]二叉树问题（https://www.luogu.com.cn/problem/P3884）基础树形DP计算两点间路径变种长度
P3915 树的分解（https://www.luogu.com.cn/problem/P3915）递归拆解生成等大小的连通块
P4615 [COCI2017-2018#5] Birokracija（https://www.luogu.com.cn/problem/P4615）树形DP
P5002 专心OI - 找祖先（https://www.luogu.com.cn/problem/P5002）使用树形DP与容斥原理进行计数
P5651 基础最短路练习题（https://www.luogu.com.cn/problem/P5651）脑筋急转弯使用并查集去环，转换为树形DP里面任意两点路径的异或和
P6591 [YsOI2020]植树（https://www.luogu.com.cn/problem/P6591）换根DP，即无根树递归判断每个节点作为根节点的情况
P7159 「dWoi R1」Sweet Fruit Chocolate（https://www.luogu.com.cn/problem/P7159）树形DP枚举计数与快速幂计算

==================================AtCoder=================================
F - Expensive Expense （https://atcoder.jp/contests/abc222/tasks/abc222_f）换根DP
D. Distance in Tree（https://codeforces.com/problemset/problem/161/D）树形DP计数，记录距离为k的点对数

================================CodeForces================================
C. Uncle Bogdan and Country Happiness（https://codeforces.com/problemset/problem/1388/C）树形DP模拟计算，递归获取子树信息，逆向从上往下还原
F. Maximum White Subtree（https://codeforces.com/problemset/problem/1324/F）经典换根DP题，两遍dfs搜索更新计算
D. Book of Evil（https://codeforces.com/problemset/problem/337/D）经典换根DP题，两遍dfs搜索更新计算

参考：OI WiKi（xx）
"""


class TreeDP:
    def __init__(self):
        return

    @staticmethod
    def change_root_dp(n: int, edges: List[List[int]], price: List[int]):
        # 模板:  换根DP
        edge = [[] for _ in range(n)]
        for u, v in edges:
            edge[u].append(v)
            edge[v].append(u)

        @lru_cache(None)
        def dfs(i, fa):
            # 注意在星状图的复杂度是O(n^2)（还有一种特殊的树结构是树链）
            # 也是求以此为根的最大路径
            ans = 0
            for j in edge[i]:
                if j != fa:
                    cur = dfs(j, i)
                    ans = ans if ans > cur else cur
            return ans + price[i]

        return max(dfs(i, -1) - price[i] for i in range(n))

    @staticmethod
    def sum_of_distances_in_tree(n: int, edges):
        # 计算节点到所有其他节点的总距离
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        tree = [[] for _ in range(n)]
        stack = [0]
        visit = {0}
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    if j not in visit:
                        visit.add(j)
                        nex.append(j)
                        tree[i].append(j)
            stack = nex[:]

        def dfs(x):
            res = 1
            for y in tree[x]:
                res += dfs(y)
            son_count[x] = res
            return res

        son_count = [0] * n
        dfs(0)

        def dfs(x):
            res = son_count[x] - 1
            for y in tree[x]:
                res += dfs(y)
            son_dis[x] = res
            return res

        son_dis = [0] * n
        dfs(0)

        def dfs(x):
            for y in tree[x]:
                father_dis[y] = (
                    son_dis[x] - son_dis[y] - son_count[y]) + father_dis[x] + n - son_count[y]
                dfs(y)
            return

        father_dis = [0] * n
        dfs(0)
        return [father_dis[i] + son_dis[i] for i in range(n)]

    @staticmethod
    def longest_path_through_node(dct):

        # 模板: 换根DP，两遍DFS获取从下往上与从上往下的DP信息
        n = len(dct)

        # 两遍DFS获取从下往上与从上往下的节点最远距离
        def dfs(x, fa):
            res = [0, 0]
            for y in dct[x]:
                if y != fa:
                    dfs(y, x)
                    res.append(max(down_to_up[y]) + 1)
            down_to_up[x] = nlargest(2, res)
            return

        # 默认以 0 为根
        down_to_up = [[] for _ in range(n)]
        dfs(0, -1)

        def dfs(x, pre, fa):
            up_to_down[x] = pre
            son = [0, 0]
            for y in dct[x]:
                if y != fa:
                    son.append(max(down_to_up[y]))
            son = nlargest(2, son)

            for y in dct[x]:
                if y != fa:
                    father = pre + 1
                    tmp = son[:]
                    if max(down_to_up[y]) in tmp:
                        tmp.remove(max(down_to_up[y]))
                    if tmp[0]:
                        father = father if father > tmp[0] + 2 else tmp[0] + 2
                    dfs(y, father, x)
            return

        up_to_down = [0] * n
        # 默认以 0 为根
        dfs(0, 0, -1)
        # 树的直径、核心可通过这两个数组计算得到，其余类似的递归可参照这种方式
        return up_to_down, down_to_up


class TreeDiameter:
    def __init__(self):
        return

    @staticmethod
    def get_diameter_bfs(edge):

        def bfs(node):
            # 模板：使用BFS计算获取树的直径端点以及直径长度
            d = 0
            q = deque([(node, -1, d)])
            while q:
                node, pre, d = q.popleft()
                for nex in edge[node]:
                    if nex != pre:
                        q.append((nex, node, d + 1))
            return node, d

        n = len(edge)

        # 这个算法依赖于一个性质，对于树中的任一个点，距离它最远的点一定是树上一条直径的一个端点
        x, _ = bfs(0)
        # 任取树中的一个节点x，找出距离它最远的点y，那么点y就是这棵树中一条直径的一个端点。我们再从y出发，找出距离y最远的点就找到了一条直径
        y, dis = bfs(x)
        return dis

    @staticmethod
    def get_diameter_dfs(edge):

        def dfs(i, fa):
            nonlocal ans
            a = b = 0
            for j in edge[i]:
                if j != fa:
                    x = dfs(j, i)
                    if x >= a:
                        a, b = x, a
                    elif x >= b:
                        b = x
            ans = ans if ans > a + b else a + b
            return a + 1 if a > b else b + 1

        # 模板：使用DFS与动态规划计算直径
        ans = 0
        dfs(0, -1)
        return ans


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_2458(root, queries: List[int]) -> List[int]:
        # 模板：经典类似换根 DP 的思想跑两遍 DFS
        def dfs(node, d):
            if not node:
                return 0
            left = dfs(node.left, d + 1)
            right = dfs(node.right, d + 1)
            h = max(left, right)
            node_depth[node.val] = d
            node_height[node.val] = h
            depth_height[node.val] = d + h
            return h + 1

        # 节点深度
        node_depth = dict()
        # 子树高度
        node_height = dict()
        # 每层节点的子树深度集合
        depth_height = dict()
        dfs(root, 0)

        def get_ans(node, pre):
            if not node:
                return
            ans[node.val] = pre

            pre = max(pre, node_depth[node.val])
            if node.right:
                pre_right = max(pre, depth_height[node.right.val])
            else:
                pre_right = pre

            if node.left:
                pre_left = max(pre, depth_height[node.left.val])
            else:
                pre_left = pre

            get_ans(node.left, pre_right)
            get_ans(node.right, pre_left)
            return

        ans = dict()
        get_ans(root, 0)
        return [ans[q] for q in queries]

    @staticmethod
    def cf_1388c(ac):
        n, m = ac.read_ints()
        person = ac.read_list_ints()
        h = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_ints_minus_one()
            edge[x].append(y)
            edge[y].append(x)

        @ac.bootstrap
        def dfs(i, fa):
            nonlocal ans
            a = b = 0
            for j in edge[i]:
                if j != fa:
                    yield dfs(j, i)
                    a += pos[j]
                    b += neg[j]

            if (h[i] + person[i] + b + a) % 2:
                ans = False
                yield
            good = (h[i] + person[i] + b + a) // 2
            bad = person[i] + a + b - good
            if good < 0 or bad < 0 or bad > person[i] + b:
                ans = False
            pos[i] = good
            neg[i] = bad
            yield

        ans = True
        pos = [0] * n
        neg = [0] * n
        dfs(0, -1)
        return "YES" if ans else "NO"

    @staticmethod
    def cf_161d(n, k, pairs):
        # 模板：记录树中距离为 k 的节点对数量
        edge = [[] for _ in range(n)]
        for x, y in pairs:
            edge[x].append(y)
            edge[y].append(x)
        dp = [[0] * (k + 1) for _ in range(n)]

        def dfs(i, fa):
            nonlocal ans
            dp[i][0] = 1
            for j in edge[i]:
                if j != fa:
                    yield dfs(j, i)
                    for s in range(1, k + 1):
                        dp[i][s] += dp[j][s - 1]

            ans += dp[i][k]
            cur = 0
            for j in edge[i]:
                if j != fa:
                    for s in range(1, k):
                        cur += dp[j][s - 1] * (dp[i][k - s] - dp[j][k - s - 1])
            ans += cur // 2
            yield

        ans = 0
        dfs(0, -1)
        return ans

    @staticmethod
    def cf_1324f(ac=FastIO()):

        # 模板：换根DP，根据题意进行转换贪心计算结果
        n = ac.read_int()
        nums = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_ints_minus_one()
            edge[x].append(y)
            edge[y].append(x)

        @ac.bootstrap
        def dfs(i, fa):
            res = 0
            for j in edge[i]:
                if j != fa:
                    yield dfs(j, i)
                    cur = son[j] + 2 * nums[j] - 1
                    res += ac.max(cur, 0)
            son[i] = res
            yield

        # 第一遍获取从下往上的最优结果
        son = [0] * n
        dfs(0, -1)

        @ac.bootstrap
        def dfs2(i, fa, pre):
            ans[i] = son[i] + pre + 2 * nums[i] - 1

            lst = [son[j] + 2 * nums[j] - 1 for j in edge[i] if j != fa]
            s = sum([y for y in lst if y >= 0])
            for j in edge[i]:
                if j != fa:
                    tmp = son[j] + 2 * nums[j] - 1
                    cur = ac.max(0, pre + s - ac.max(0, tmp) + 2 * nums[i] - 1)
                    yield dfs2(j, i, cur)
            yield

        # 第二遍获取从下往上的最优结果并更新加和
        ans = [0] * n
        dfs2(0, -1, 0)
        ac.lst(ans)
        return

    @staticmethod
    def cf_337d(ac=FastIO()):
        n, m, d = ac.read_ints()
        evil = set(ac.read_list_ints_minus_one())
        edge = [[] for _ in range(n)]
        for _ in range(n-1):
            u, v = ac.read_ints_minus_one()
            edge[u].append(v)
            edge[v].append(u)

        @ac.bootstrap
        def dfs(i, fa):
            res = -inf
            for j in edge[i]:
                if j != fa:
                    yield dfs(j, i)
                    res = ac.max(res, son[j]+1)
            if i in evil:
                res = ac.max(res, 0)
            son[i] = res
            yield

        # 计算子节点最远的evil
        son = [-inf]*n
        dfs(0, -1)

        @ac.bootstrap
        def dfs2(i, fa, pre):
            father[i] = pre
            a = b = pre+1
            for j in edge[i]:
                if j != fa:
                    c = son[j] + 2
                    if c >= a:
                        a, b = c, a
                    elif c >= b:
                        a, b = a, c
            if i in evil:
                c = 1
                if c >= a:
                    a, b = c, a
                elif c >= b:
                    a, b = a, c
            for j in edge[i]:
                if j != fa:
                    c = son[j] + 2
                    if a == c:
                        yield dfs2(j, i, b)
                    else:
                        yield dfs2(j, i, a)
            yield

        # 计算父节点最远的evil
        father = [-inf] * n
        dfs2(0, -1, -inf)
        ans = sum(ac.max(father[i], son[i]) <= d for i in range(n))
        ac.st(ans)
        return

    @staticmethod
    def lc_1617(n: int, edges: List[List[int]]) -> List[int]:
        # 模板：枚举子集使用并查集判断连通性再计算树的直径
        ans = [0] * n
        for state in range(1, 1 << n):
            node = [i for i in range(n) if state & (1 << i)]
            ind = {num: i for i, num in enumerate(node)}
            m = len(node)
            dct = [[] for _ in range(m)]
            uf = UnionFind(m)
            for u, v in edges:
                u -= 1
                v -= 1
                if u in ind and v in ind:
                    dct[ind[u]].append(ind[v])
                    dct[ind[v]].append(ind[u])
                    uf.union(ind[u], ind[v])
            if uf.part != 1:
                continue
            # 计算直径或者是get_diameter_bfs都可以
            ans[TreeDiameter().get_diameter_dfs(dct)] += 1
        return ans[1:]


class TestGeneral(unittest.TestCase):

    def test_tree_dp(self):
        td = TreeDP()
        n = 5
        edges = [[0, 1], [0, 2], [2, 4], [1, 3]]
        assert td.sum_of_distances_in_tree(n, edges) == [6, 7, 7, 10, 10]

        dct = [[1, 2], [0, 3], [0, 4], [1], [2]]
        up_to_down, down_to_up = td.longest_path_through_node(dct)
        assert up_to_down == [0, 3, 3, 4, 4]
        assert down_to_up == [[2, 2], [1, 0], [1, 0], [0, 0], [0, 0]]
        return


if __name__ == '__main__':
    unittest.main()
