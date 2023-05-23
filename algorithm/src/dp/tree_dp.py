
import unittest
from collections import deque, Counter
from functools import lru_cache
from heapq import nlargest
from itertools import accumulate
from operator import add
from typing import List
from algorithm.src.fast_io import FastIO, inf
from algorithm.src.graph.union_find import UnionFind

"""
算法：树形DP、树的直径、树上差分、树的重心（以及树的每个节点到其余节点的总距离和）、树的最小偏心距
功能：在树形或者图结构上进行DP，有换根DP，自顶向下和自底向上DP
题目：

===================================力扣===================================
2458 移除子树后的二叉树高度（https://leetcode.cn/problems/height-of-binary-tree-after-subtree-removal-queries/）跑两边DFS进行自顶向下和自底向上DP结合
2440 创建价值相同的连通块（https://leetcode.cn/problems/create-components-with-same-value/）利用总和的因子和树形递归判断连通块是否可行
1569 将子数组重新排序得到同一个二叉查找树的方案数（https://leetcode.cn/problems/number-of-ways-to-reorder-array-to-get-same-bst/solution/by-liupengsay-yi3h/）
968. 监控二叉树（https://leetcode.cn/problems/binary-tree-cameras/）树形DP监控每个节点
2538. 最大价值和与最小价值和的差值（https://leetcode.cn/problems/difference-between-maximum-and-minimum-price-sum/）树形换根DP，求去掉其中一个叶子节点的最大直径
124. 二叉树中的最大路径和（https://leetcode.cn/problems/binary-tree-maximum-path-sum/）树形DP
1617. 统计子树中城市之间最大距离（https://leetcode.cn/problems/count-subtrees-with-max-distance-between-cities/）二进制枚举加树的直径计算
2378. 选择边来最大化树的得分（https://leetcode.cn/problems/choose-edges-to-maximize-score-in-a-tree/）树形DP
2445. 值为 1 的节点数（https://leetcode.cn/problems/number-of-nodes-with-value-one/）自上而下DP模拟
834. 树中距离之和（https://leetcode.cn/problems/sum-of-distances-in-tree/）树的总距离，求树的重心

===================================洛谷===================================

P1395 会议（https://www.luogu.com.cn/problem/P1395）树的总距离，求树的重心，单个节点距离其他所有节点的最大距离，换根DP可以做
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
P2015 二叉苹果树（https://www.luogu.com.cn/problem/P2015）树形DP，有点像树上背包
P2014 [CTSC1997] 选课（https://www.luogu.com.cn/problem/P2014）树形DP
P4316 绿豆蛙的归宿（https://www.luogu.com.cn/problem/P4316）逆向建图，拓扑排序DP
P1351 [NOIP2014 提高组] 联合权值（https://www.luogu.com.cn/problem/P1351#submit）树形DP
P3304 [SDOI2013]直径（https://www.luogu.com.cn/problem/P3304）经典计算带权无向图的直径以及直径的必经边
P3408 恋爱（https://www.luogu.com.cn/problem/P3408）树形DP
P3478 [POI2008] STA-Station（https://www.luogu.com.cn/problem/P3478）树的质心
P3931 SAC E#1 - 一道难题 Tree（https://www.luogu.com.cn/problem/P3931）典型树形DP
P4084 [USACO17DEC]Barn Painting G（https://www.luogu.com.cn/problem/P4084）典型树形DP

==================================AtCoder=================================
F - Expensive Expense （https://atcoder.jp/contests/abc222/tasks/abc222_f）换根DP
D. Distance in Tree（https://codeforces.com/problemset/problem/161/D）树形DP计数，记录距离为k的点对数

================================CodeForces================================
C. Uncle Bogdan and Country Happiness（https://codeforces.com/problemset/problem/1388/C）树形DP模拟计算，递归获取子树信息，逆向从上往下还原
F. Maximum White Subtree（https://codeforces.com/problemset/problem/1324/F）经典换根DP题，两遍dfs搜索更新计算
D. Book of Evil（https://codeforces.com/problemset/problem/337/D）经典换根DP题，两遍dfs搜索更新计算
E. Tree Painting（https://codeforces.com/problemset/problem/1187/E）经典换根DP题，两遍dfs搜索更新计算
E. Lomsat gelral（https://codeforces.com/problemset/problem/600/E）迭代方式写深搜序，按秩合并，由小到大
D. A Wide, Wide Graph（https://codeforces.com/problemset/problem/1805/D）树的直径计算，任意点到直径的某个端点的距离最长
G. White-Black Balanced Subtrees（https://codeforces.com/contest/1676/problem/G）使用迭代的方式进行树形DP计算
F. Gardening Friends（https://codeforces.com/contest/1822/problem/F）计算树中节点到其余节点的最大距离

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
        # 计算节点到所有其他节点的总距离即树的重心
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


class TreeDiameterWeighted:
    def __init__(self):
        return

    @staticmethod
    def bfs(dct, src):
        # 模板：使用 BFS 计算获取带权树的直径端点以及直径长度
        n = len(dct)
        res = [inf] * n
        stack = [src]
        res[src] = 0
        parent = [-1] * n
        while stack:
            node = stack.pop()
            for nex in dct[node]:
                if nex != parent[node]:
                    parent[nex] = node
                    res[nex] = res[node] + dct[node][nex]
                    stack.append(nex)
        far = res.index(max(res))
        diameter = [far]
        while diameter[-1] != src:
            diameter.append(parent[diameter[-1]])
        diameter.reverse()
        return far, diameter, res[far]


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


class TreeDiameterDis:
    # 任取树中的一个节点x，找出距离它最远的点y，那么点y就是这棵树中一条直径的一个端点。我们再从y出发，找出距离y最远的点就找到了一条直径。
    # 这个算法依赖于一个性质：对于树中的任一个点，距离它最远的点一定是树上一条直径的一个端点。
    def __init__(self, edge):
        self.edge = edge
        self.n = len(self.edge)
        return

    def get_furthest(self, node):
        q = deque([(node, -1)])
        while q:
            node, pre = q.popleft()
            for x in self.edge[node]:
                if x != pre:
                    q.append((x, node))
        return node

    def get_diameter_node(self):
        # 获取树的直径端点
        x = self.get_furthest(0)
        y = self.get_furthest(x)
        return x, y

    def get_bfs_dis(self, node):
        dis = [inf] * self.n
        stack = [node]
        dis[node] = 0
        while stack:
            nex = []
            for i in stack:
                for j in self.edge[i]:
                    if dis[j] == inf:
                        nex.append(j)
                        dis[j] = dis[i] + 1
            stack = nex[:]
        return dis


class TreeCentroid:
    def __init__(self):
        return

    @staticmethod
    def get_tree_centroid(dct: List[List[int]]) -> int:
        # 模板：获取树的编号最小的重心
        n = len(dct)
        sub = [1]*n  # 以 0 为根的有根树节点 i 的子树节点数
        ma = [0]*n  # 以 i 为根最大的子树节点数
        ma[0] = n
        center = 0
        stack = [[0, -1, 1]]
        while stack:
            i, fa, state = stack.pop()
            if state:
                stack.append([i, fa, 0])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i, 1])
            else:
                for j in dct[i]:
                    if j != fa:
                        sub[i] += sub[j]
                        ma[i] = ma[i] if ma[i] > sub[j] else sub[j]
                # 类似换根 DP 计算最大子树节点数
                ma[i] = ma[i] if ma[i] > n-sub[i] else n-sub[i]
                if ma[i] < ma[center] or (ma[i] == ma[center] and i < center):
                    center = i
        # 树的重心等同于最大子树最小的节点也等同于到其余所有节点距离之和最小的节点
        return center

    @staticmethod
    def get_tree_distance(dct: List[List[int]]) -> List[int]:
        # 模板：计算树的每个节点到其余所有的节点的总距离

        n = len(dct)
        sub = [1] * n  # 子树节点个数
        ans = [0] * n  # 到其余所有节点的距离之和

        # 第一遍 BFS 自下而上计算子树节点数与 ans[0]
        stack = [[0, -1, 1]]
        while stack:
            i, fa, state = stack.pop()
            if state:
                stack.append([i, fa, 0])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i, 1])
            else:
                for j in dct[i]:
                    if j != fa:
                        sub[i] += sub[j]
                        ans[i] += ans[j] + sub[j]

        # 第二遍 BFS 自上而下计算距离
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            for j in dct[i]:
                if j != fa:
                    ans[j] = ans[i] - sub[j] + n - sub[j]
                    stack.append([j, i])
        return ans

    @staticmethod
    def get_tree_distance_max(dct: List[List[int]]) -> List[int]:
        # 模板：计算树的每个节点到其余所有的节点的最大距离（也可以使用直径上的点BFS）

        n = len(dct)
        sub = [[0, 0] for _ in range(n)]

        # 第一遍 BFS 自下而上计算子树的最大距离与次大距离
        stack = [[0, -1, 1]]
        while stack:
            i, fa, state = stack.pop()
            if state:
                stack.append([i, fa, 0])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i, 1])
            else:
                a, b = sub[i]
                for j in dct[i]:
                    if j != fa:
                        x = sub[j][0]+1
                        if x >= a:
                            a, b = x, a
                        elif x>=b:
                            b = x
                sub[i] = [a, b]

        # 第二遍 BFS 自上而下更新最大距离
        stack = [[0, -1, 0]]
        ans = [s[0] for s in sub]
        while stack:
            i, fa, d = stack.pop()
            ans[i] = ans[i] if ans[i] > d else d
            for j in dct[i]:
                if j != fa:
                    nex = d
                    x = sub[j][0]+1
                    a, b = sub[i]
                    # 排除当前子节点的距离
                    if x == a:
                        nex = nex if nex > b else b
                    else:
                        nex = nex if nex > a else a
                    stack.append([j, i, nex+1])
        return ans


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1676g(ac=FastIO()):
        # 模板：使用迭代的方式计算树形DP
        for _ in range(ac.read_int()):
            n = ac.read_int()
            parent = ac.read_list_ints_minus_one()
            color = ac.read_str()
            dct = [[] for _ in range(n)]
            for i in range(n-1):
                dct[parent[i]].append(i+1)
            ans = 0
            sub = [0]*n
            stack = [[0, 1]]
            while stack:
                i, state = stack.pop()
                if state:
                    stack.append([i, 0])
                    for j in dct[i]:
                        stack.append([j, 1])
                else:
                    x = 0
                    for j in dct[i]:
                        x += sub[j]
                    x += 1 if color[i] == "B" else -1
                    sub[i] = x
                    ans += x == 0
            ac.st(ans)
        return

    @staticmethod
    def cf_1805d(ac=FastIO()):
        # 模板：使用树的直径与端点距离，计算节点对距离至少为k的连通块个数
        n = ac.read_int()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_ints_minus_one()
            edge[u].append(v)
            edge[v].append(u)
        tree = TreeDiameterDis(edge)
        u, v = tree.get_diameter_node()
        dis1 = tree.get_bfs_dis(u)
        dis2 = tree.get_bfs_dis(v)
        diff = [0] * n
        for i in range(n):
            diff[ac.max(dis1[i], dis2[i])] += 1
        diff[0] = 1
        diff = list(accumulate(diff, add))
        ac.lst([ac.min(x, n) for x in diff])
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

    @staticmethod
    def cf_1187e(ac=FastIO()):
        # 模板：经典换根DP题计算最佳结果
        n = ac.read_int()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            edge[u].append(v)
            edge[v].append(u)

        @ac.bootstrap
        def dfs(i, fa):
            for j in edge[i]:
                if j != fa:
                    yield dfs(j, i)
                    down[i] += down[j]
                    son[i] += son[j]
            son[i] += 1
            down[i] += son[i]
            yield

        down = [0] * n
        son = [0] * n
        dfs(0, -1)

        @ac.bootstrap
        def dfs2(i, fa, pre):
            up[i] = pre
            res = sum(down[j] for j in edge[i] if j != fa)
            for j in edge[i]:
                if j != fa:
                    yield dfs2(j, i, (n - son[j]) + pre + (res - down[j]))
            yield

        up = [0] * n
        dfs2(0, -1, 0)
        ac.st(max(up[i] + (down[i] - son[i]) + n for i in range(n)))
        return

    @staticmethod
    def cf_600e_bfs(ac=FastIO()):
        # 模板：自下而上递归的迭代写法，从小到大按秩合并
        n = ac.read_int()
        colors = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_ints_minus_one()
            edge[i].append(j)
            edge[j].append(i)
        # 深搜序自下而上以及父子信息获取
        stack = [[0, -1]]
        parent = [-1] * n
        down_to_up_order = []
        while stack:
            i, fa = stack.pop()
            down_to_up_order.append(i)
            for j in edge[i]:
                if j != fa:
                    stack.append([j, i])
                    parent[j] = i
        down_to_up_order.reverse()

        # 维护一个最大值的出现次数
        mx = [0] * n
        ans = [0] * n
        dp = [None] * n
        for i in down_to_up_order:
            dp[i] = Counter()
            dp[i][colors[i]] += 1
            mx[i] = 1
            ans[i] = colors[i]
            for j in edge[i]:
                if dp[j]:
                    if len(dp[j]) > len(dp[i]):
                        # 从小到大
                        dp[i], dp[j] = dp[j], dp[i]
                        mx[i] = mx[j]
                        ans[i] = ans[j]
                    for w in dp[j]:
                        # 按秩合并
                        dp[i][w] += dp[j][w]
                        if dp[i][w] == mx[i]:
                            ans[i] += w
                        elif dp[i][w] > mx[i]:
                            mx[i] = dp[i][w]
                            ans[i] = w
                    # 及时清空
                    dp[j] = None
        ac.lst(ans)
        return

    @staticmethod
    def cf_600e_dfs(ac=FastIO()):
        # 模板：自下而上递归的递归写法，从小到大按秩合并
        n = ac.read_int()
        nums = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_ints_minus_one()
            edge[u].append(v)
            edge[v].append(u)

        @ac.bootstrap
        def dfs(i, fa):
            dp[i] = Counter()
            dp[i][nums[i]] += 1
            ceil[i] = 1
            cur = nums[i]
            for j in edge[i]:
                if j != fa:
                    yield dfs(j, i)
                    if len(dp[j]) > len(dp[i]):
                        dp[i], dp[j] = dp[j], dp[i]
                        cur = ans[j]
                        ceil[i] = ceil[j]
                    for num in dp[j]:
                        dp[i][num] += dp[j][num]
                        if dp[i][num] > ceil[i]:
                            ceil[i] = dp[i][num]
                            cur = num
                        elif dp[i][num] == ceil[i]:
                            cur += num
                    dp[j] = None
            ans[i] = cur
            yield

        dp = [None] * n
        ans = [0] * n
        ceil = [0] * n
        dfs(0, -1)
        ac.lst(ans)
        return

    @staticmethod
    def lg_p1395_1(ac=FastIO()):
        # 模板：计算树的重心为最大子树节点数最小
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n-1):
            i, j = ac.read_ints()
            dct[i-1].append(j-1)
            dct[j-1].append(i-1)

        root = TreeCentroid().get_tree_centroid(dct)

        def bfs_diameter(src):
            ans = 0
            stack = [[src, 0]]
            parent = [-1] * n
            while stack:
                u, dis = stack.pop()
                ans += dis
                for v in dct[u]:
                    if v != parent[u]:
                        parent[v] = u
                        stack.append([v, dis + 1])
            return ans

        ac.lst([root + 1, bfs_diameter(root)])
        return

    @staticmethod
    def lg_p1395_2(ac=FastIO()):
        # 模板：计算树的重心为距离其余所有节点
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n-1):
            i, j = ac.read_ints()
            dct[i-1].append(j-1)
            dct[j-1].append(i-1)

        ans = TreeCentroid().get_tree_distance(dct)
        dis = min(ans)
        ac.lst([ans.index(dis)+1, dis])
        return

    @staticmethod
    def cf_1822f(ac=FastIO()):
        # 模板：换根 DP 计算树中节点其余节点最大的距离
        for _ in range(ac.read_int()):
            n, k, c = ac.read_ints()
            dct = [[] for _ in range(n)]
            for _ in range(n-1):
                i, j = ac.read_ints_minus_one()
                dct[i].append(j)
                dct[j].append(i)

            dis = TreeCentroid().get_tree_distance_max(dct)

            ans = -inf
            stack = [[0, 0, -1]]
            while stack:
                i, d, fa = stack.pop()
                cur = dis[i]*k - d
                ans = ac.max(ans, cur)
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, d+c, i])
            ac.st(ans)
        return

    @staticmethod
    def lg_p1352(ac=FastIO()):
        # 模板：使用树形DP的迭代写法进行计算
        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]

        dct = [[] for _ in range(n)]
        degree = [0]*n
        for _ in range(n-1):
            x, y = ac.read_ints_minus_one()
            dct[y].append(x)
            degree[x] += 1

        root = [i for i in range(n) if not degree[i]][0]
        dp = [[0, 0] for _ in range(n)]
        stack = [[root, -1]]
        while stack:
            # 为取反码后的负数则直接出栈
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                x = nums[i] if nums[i] > 0 else 0
                y = 0
                for j in dct[i]:
                    if j != fa:
                        a, b = dp[j]
                        x += a
                        y += b
                dp[i] = [y, ac.max(x, y)]
        ac.st(max(dp[root]))
        return

    @staticmethod
    def lg_p2015(ac=FastIO()):
        # 模板：树形DP
        n, q = ac.read_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(n-1):
            x, y, z = ac.read_ints()
            dct[x-1][y-1] = z
            dct[y-1][x-1] = z
        dp = [[0]*(q+1) for _ in range(n)]
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                if len(dct[i]) > 1:
                    a, b = [x for x in dct[i] if x != fa]
                    for j in range(1, q+1):
                        cur = ac.max(dp[a][j-1]+dct[i][a], dp[b][j-1]+dct[i][b])
                        for k in range(j-1):
                            cur = ac.max(cur, dp[a][k]+dp[b][j-k-2]+dct[i][a]+dct[i][b])
                        dp[i][j] = cur
        ac.st(dp[0][q])
        return

    @staticmethod
    def lg_p2014(ac=FastIO()):
        # 模板：树形DP加背包DP
        n, m = ac.read_ints()
        dct = [[] for _ in range(n + 1)]
        nums = [0]
        for i in range(n):
            k, s = ac.read_ints()
            nums.append(s)
            dct[k].append(i + 1)
        dp = [[0] * (m + 2) for _ in range(n + 1)]
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                dp[i][1] = nums[i]
                for j in dct[i]:
                    if j != fa:
                        cur = dp[i][:]
                        for x in range(1, m + 2):
                            for y in range(m + 2 - x):
                                cur[x + y] = ac.max(cur[x + y], dp[i][x] + dp[j][y])
                        dp[i] = cur[:]
        ac.st(dp[0][m + 1])
        return

    @staticmethod
    def lg_p4316(ac=FastIO()):
        # 模板：反向建图加拓扑排序树形概率DP
        n, m = ac.read_ints()
        dp = [0 for _ in range(n)]
        degree = [0]*n
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            a, b, w = ac.read_ints()
            a -= 1
            b -= 1
            dct[b][a] = w
            degree[a] += 1
        cnt = degree[:]

        stack = deque([n-1])
        while stack:
            i = stack.popleft()
            k = len(dct[i])
            a = dp[i]
            for j in dct[i]:
                dp[j] += a + dct[i][j]
                degree[j] -= 1
                if not degree[j]:
                    dp[j] /= cnt[j]
                    stack.append(j)
        ans = "%.2f" % (dp[0])
        ac.st(ans)
        return

    @staticmethod
    def lg_p1351(ac=FastIO()):
        # 模板：树形DP
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        nums = ac.read_list_ints()
        ceil = ans = 0
        mod = 10007
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            lst = []
            if fa != -1:
                lst.append(nums[fa])
            for j in dct[i]:
                if j != fa:
                    stack.append([j, i])
                    lst.append(nums[j])
            if lst:

                s = sum(lst)
                a = b = 0
                for num in lst:
                    ans += num * (s - num)
                    ans %= mod
                    if num > a:
                        a, b = num, a
                    elif num > b:
                        b = num
                ceil = ac.max(ceil, a * b)

        ac.lst([ceil, ans])
        return

    @staticmethod
    def lg_p3304(ac=FastIO()):
        # 模板：经典计算带权无向图的直径以及直径的必经边
        n = ac.read_int()
        dct = [dict() for _ in range(n)]
        for _ in range(n - 1):
            i, j, k = ac.read_ints()
            i -= 1
            j -= 1
            dct[i][j] = dct[j][i] = k
        # 首先计算直径
        tree = TreeDiameterWeighted()
        x, _, _ = tree.bfs(dct, 0)
        y, path, dia = tree.bfs(dct, x)
        ac.st(dia)
        # 确定直径上每个点的最远端距离
        nodes = set(path)
        dis = [0] * n
        for x in path:
            q = [[x, -1, 0]]
            while q:
                i, fa, d = q.pop()
                for j in dct[i]:
                    if j != fa and j not in nodes:
                        dis[x] = d + dct[i][j]
                        q.append([j, i, d + dct[i][j]])

        # 计算直径必经边的最右边端点
        m = len(path)
        pre = right = 0
        for j in range(1, m):
            pre += dct[path[j - 1]][path[j]]
            right = j
            if dis[path[j]] == dia - pre:  # 此时点下面有非当前直径的最远路径
                break

        # 计算直径必经边的最左边端点
        left = m - 1
        post = 0
        for j in range(m - 2, -1, -1):
            post += dct[path[j]][path[j + 1]]
            left = j
            if dis[path[j]] == dia - post:  # 此时点下面有非当前直径的最远路径
                break

        ans = ac.max(0, right - left)
        ac.st(ans)
        return

    @staticmethod
    def lg_3408(ac=FastIO()):

        # 模板：树形 DP 模拟
        n, t, c = ac.read_ints()
        dct = [[] for _ in range(n + 1)]
        nums = [c]
        for i in range(n):
            b, a = ac.read_ints()
            dct[b].append(i + 1)
            nums.append(a)

        stack = [0]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in dct[i]:
                    stack.append(j)
            else:
                i = ~i
                if not dct[i]:
                    continue
                else:
                    # 收到子树下属的最少花费
                    if nums[i] > t:
                        continue

                    # 需要最少的 x 个下属花费
                    x = math.ceil(len(dct[i]) * nums[i] / t)
                    lst = []
                    for j in dct[i]:
                        lst.append(nums[j])
                    lst.sort()
                    nums[i] = sum(lst[:x])
        ac.st(nums[0])
        return

    @staticmethod
    def lg_p3478(ac=FastIO()):
        # 模板：计算树的质心
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        dis = TreeCentroid().get_tree_distance(dct)
        ind = 0
        for i in range(1, n):
            if dis[i] > dis[ind]:
                ind = i
        ac.st(ind + 1)
        return

    @staticmethod
    def lg_p3931(ac=FastIO()):
        # 模板：树形 DP 模拟
        n, root = ac.read_ints()
        root -= 1
        dct = [dict() for _ in range(n)]
        for _ in range(n - 1):
            i, j, c = ac.read_ints_minus_one()
            c += 1
            dct[i][j] = dct[j][i] = c
        stack = [[root, -1]]
        sub = [inf] * n
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                if len(dct[i]) == 1 and i != root:
                    continue
                res = 0
                for j in dct[i]:
                    if j != fa:
                        res += ac.min(dct[i][j], sub[j])
                sub[i] = res
        ac.st(sub[root] if sub[root] < inf else 0)
        return


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
