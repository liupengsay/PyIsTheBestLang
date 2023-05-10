import heapq
import math
import unittest
from collections import deque
from typing import List

from algorithm.src.fast_io import FastIO, inf
from algorithm.src.graph.union_find import UnionFind

"""

算法：最小生成树（Kruskal算法和Prim算法两种）、严格次小生成树（使用LCA枚举替换边计算可得）
功能：计算无向图边权值和最小的生成树
Prim在稠密图中比Kruskal优，在稀疏图中比Kruskal劣。Prim是以更新过的节点的连边找最小值，Kruskal是直接将边排序。
两者其实都是运用贪心的思路，Kruskal相对比较常用

题目：

===================================力扣===================================
1489. 找到最小生成树里的关键边和伪关键边（https://leetcode.cn/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/）计算最小生成树的关键边与伪关键边

===================================洛谷===================================
P3366 最小生成树（https://www.luogu.com.cn/problem/P3366）最小生成树裸题
P2820 局域网（https://www.luogu.com.cn/problem/P2820）逆向思维，求最小生成树权值和
P1991 无线通讯网（https://www.luogu.com.cn/problem/P1991）计算保证k个连通块下最小的边权值

P1661 扩散（https://www.luogu.com.cn/problem/P1661）最小生成树的边最大权值
P1547 [USACO05MAR]Out of Hay S（https://www.luogu.com.cn/problem/P1547）最小生成树的边最大权值
P2121 拆地毯（https://www.luogu.com.cn/problem/P2121）保留 k 条边的最大生成树权值
P2126 Mzc家中的男家丁（https://www.luogu.com.cn/problem/P2126）转化为最小生成树求解
P2872 Building Roads S（https://www.luogu.com.cn/problem/P2872）使用prim计算最小生成树
P2330 [SCOI2005]繁忙的都市（https://www.luogu.com.cn/problem/P2330）最小生成树边数量与最大边权值
P2504 [HAOI2006]聪明的猴子（https://www.luogu.com.cn/problem/P2504）识别为最小生成树求解
P2700 逐个击破（https://www.luogu.com.cn/problem/P2700）逆向思维与最小生成树，选取最大权组合，修改并查集size

P1195 口袋的天空（https://www.luogu.com.cn/record/list?user=739032&status=12&page=13）最小生成树生成K个连通块
P1194 买礼物（https://www.luogu.com.cn/problem/P1194）最小生成树变种问题


P2916 [USACO08NOV]Cheering up the Cow G（https://www.luogu.com.cn/problem/P2916）需要自定义排序之后计算最小生成树的好题
P4955 [USACO14JAN]Cross Country Skiing S（https://www.luogu.com.cn/problem/P4955）最小生成树，自定义中止条件
P6705 [COCI2010-2011#7] POŠTAR（https://www.luogu.com.cn/problem/P6705）枚举最小值，使用最小生成树，与自定义权值进行计算
P7775 [COCI2009-2010#2] VUK（https://www.luogu.com.cn/problem/P7775）BFS加最小生成树思想，求解

P2658 汽车拉力比赛（https://www.luogu.com.cn/problem/P2658）典型最小生成树计算
P4180 [BJWC2010] 严格次小生成树（https://www.luogu.com.cn/problem/P4180）使用最小生成树与LCA倍增查询计算严格次小生成树

P1265 公路修建（https://www.luogu.com.cn/problem/P1265）使用prim求解最小生成树
P1340 兽径管理（https://www.luogu.com.cn/problem/P1340）逆序并查集，维护最小生成树的边
P1550 [USACO08OCT]Watering Hole G（https://www.luogu.com.cn/problem/P1550）经典题目，建立虚拟源点，转换为最小生成树问题
P2212 [USACO14MAR]Watering the Fields S（https://www.luogu.com.cn/problem/P2212）经典题目，使用prim计算稠密图最小生成树

================================CodeForces================================
D. Design Tutorial: Inverse the Problem（https://codeforces.com/problemset/problem/472/D）使用最小生成树判断构造给定的点对最短路距离是否存在，使用prim算法复杂度更优
E. Minimum spanning tree for each edge（https://codeforces.com/problemset/problem/609/E）使用LCA的思想维护树中任意两点的路径边权最大值，并贪心替换获得边作为最小生成树时的最小权值和，有点类似于关键边与非关键边，但二者并不相同，即为严格次小生成树


参考：OI WiKi（xx）
"""


class MinimumSpanningTree:
    def __init__(self, edges, n, method="kruskal"):
        # n个节点
        self.n = n
        # m条权值边edges
        self.edges = edges
        self.cost = 0
        self.cnt = 0
        self.gen_minimum_spanning_tree(method)
        return

    def gen_minimum_spanning_tree(self, method):

        if method == "kruskal":
            # 边优先
            self.edges.sort(key=lambda item: item[2])
            # 贪心按照权值选择边进行连通合并
            uf = UnionFind(self.n)
            for x, y, z in self.edges:
                if uf.union(x, y):
                    self.cost += z
            # 不能形成生成树
            if uf.part != 1:
                self.cost = -1
        else:
            # 点优先使用 Dijkstra求解
            dct = [dict() for _ in range(self.n)]
            for i, j, w in self.edges:
                c = dct[i].get(j, float("inf"))
                c = c if c < w else w
                dct[i][j] = dct[j][i] = c
            dis = [float("inf")]*self.n
            dis[0] = 0
            visit = [0]*self.n
            stack = [[0, 0]]
            while stack:
                d, i = heapq.heappop(stack)
                if visit[i]:
                    continue
                visit[i] = 1
                self.cost += d  # 连通花费的代价
                self.cnt += 1  # 连通的节点数
                for j in dct[i]:
                    w = dct[i][j]
                    if w < dis[j]:
                        dis[j] = w
                        heapq.heappush(stack, [w, j])
        return


class TreeAncestorWeightSecond:

    def __init__(self, dct):
        # 默认以 0 为根节点
        n = len(dct)
        self.parent = [-1] * n
        self.depth = [-1] * n
        stack = deque([0])
        self.depth[0] = 0
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                if self.depth[j] == -1:
                    self.depth[j] = self.depth[i] + 1
                    self.parent[j] = i
                    stack.append(j)

        # 根据节点规模设置层数
        self.cols = FastIO().max(2, math.ceil(math.log2(n)))
        self.dp = [[-1] * self.cols for _ in range(n)]
        self.weight = [[[-1, -1] for _ in range(self.cols)] for _ in range(n)]  # 边权的最大值与次大值
        for i in range(n):
            self.dp[i][0] = self.parent[i]
            if self.parent[i] != -1:
                self.weight[i][0] = [dct[self.parent[i]][i], -1]

        # 动态规划设置祖先初始化, dp[node][j] 表示 node 往前推第 2^j 个祖先
        for j in range(1, self.cols):
            for i in range(n):
                father = self.dp[i][j - 1]
                self.weight[i][j] = self.update(self.weight[i][j], self.weight[i][j-1])
                if father != -1:
                    self.dp[i][j] = self.dp[father][j - 1]
                    self.weight[i][j] = self.update(self.weight[i][j], self.weight[father][j-1])
        return

    @staticmethod
    def update(lst1, lst2):
        a, b = lst1
        c, d = lst2
        # 更新最大值与次大值
        for x in [c, d]:
            if x >= a:
                a, b = x, a
            elif x >= b:
                b = x
        return [a, b]

    def get_dist_weight_max_second(self, x: int, y: int) -> List[int]:
        # 计算任意点的最短路上的权重最大值与次大值
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        ans = [-1, -1]
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            ans = self.update(ans, self.weight[x][int(math.log2(d))])
            x = self.dp[x][int(math.log2(d))]
        if x == y:
            return ans

        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.dp[x][k] != self.dp[y][k]:
                ans = self.update(ans, self.weight[x][k])
                ans = self.update(ans, self.weight[y][k])
                x = self.dp[x][k]
                y = self.dp[y][k]

        ans = self.update(ans, self.weight[x][0])
        ans = self.update(ans, self.weight[y][0])
        return ans


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1991(ac=FastIO()):
        # 模板：利用最小生成树计算 k 个连通块所需的最大边权值
        k, n = ac.read_ints()
        pos = [ac.read_list_ints() for _ in range(n)]
        edge = []
        for i in range(n):
            for j in range(i + 1, n):
                a = pos[i][0] - pos[j][0]
                b = pos[i][1] - pos[j][1]
                edge.append([i, j, a * a + b * b])

        uf = UnionFind(n)
        edge.sort(key=lambda it: it[2])
        cost = 0
        for x, y, z in edge:
            if uf.part == k:
                break
            if uf.union(x, y):
                cost = z
        ans = cost**0.5
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_2820(ac=FastIO()):
        # 模板：求删除最大权值和使得存在回路的连通图变成最小生成树
        n, m = ac.read_ints()
        edge = [ac.read_list_ints() for _ in range(m)]
        uf = UnionFind(n)
        edge.sort(key=lambda it: it[2])
        cost = 0
        for x, y, z in edge:
            if not uf.union(x - 1, y - 1):
                cost += z
        ac.st(cost)
        return

    @staticmethod
    def lg_p3366_1(ac=FastIO()):
        # 模板：kruskal求最小生成树
        n, m = ac.read_ints()
        edges = []
        for _ in range(m):
            x, y, z = ac.read_list_ints()
            x -= 1
            y -= 1
            edges.append([x, y, z])
        mst = MinimumSpanningTree(edges, n, "kruskal")
        if mst.cost == -1:
            ac.st("orz")
        else:
            ac.st(mst.cost)
        return

    @staticmethod
    def lg_p3366_2(ac=FastIO()):
        # 模板：prim求最小生成树
        n, m = ac.read_ints()
        edges = []
        for _ in range(m):
            x, y, z = ac.read_list_ints()
            x -= 1
            y -= 1
            edges.append([x, y, z])
        mst = MinimumSpanningTree(edges, n, "prim")
        if mst.cnt < n:
            ac.st("orz")
        else:
            ac.st(mst.cost)
        return


    @staticmethod
    def lc_1489(n: int, edges: List[List[int]]) -> List[List[int]]:
        # 模板：求最小生成树的关键边与伪关键边
        m = len(edges)
        # 代价排序
        lst = list(range(m))
        lst.sort(key=lambda it: edges[it][2])

        # 计算最小生成树代价
        min_cost = 0
        uf = UnionFind(n)
        for i in lst:
            x, y, cost = edges[i]
            if uf.union(x, y):
                min_cost += cost

        # 枚举关键边
        key = set()
        for i in lst:
            cur_cost = 0
            uf = UnionFind(n)
            for j in lst:
                if j != i:
                    x, y, cost = edges[j]
                    if uf.union(x, y):
                        cur_cost += cost
            if cur_cost > min_cost or uf.part != 1:
                key.add(i)

        # 枚举伪关键边
        fake = set()
        for i in lst:
            if i not in key:
                cur_cost = edges[i][2]
                uf = UnionFind(n)
                # 先将当前边加入生成树
                uf.union(edges[i][0], edges[i][1])
                for j in lst:
                    x, y, cost = edges[j]
                    if uf.union(x, y):
                        cur_cost += cost
                # 若仍然是最小生成树就说明是伪关键边
                if cur_cost == min_cost and uf.part == 1:
                    fake.add(i)

        return [list(key), list(fake)]

    @staticmethod
    def lg_p2872(ac=FastIO()):

        # 模板：使用prim计算最小生成树，适合稠密图场景
        def dis(x1, y1, x2, y2):
            res = (x1 - x2) ** 2 + (y1 - y2) ** 2
            return res**0.5

        n, m = ac.read_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        dct = [dict() for _ in range(n)]
        for i in range(m):
            u, v = ac.read_ints_minus_one()
            dct[u][v] = dct[v][u] = 0
        # 初始化最短距离
        ans = nex = 0
        rest = set(list(range(1, n)))
        visit = [inf] * n
        visit[nex] = 0
        while rest:
            # 点优先选择距离当前集合最近的点合并
            i = nex
            rest.discard(i)
            d = visit[i]
            ans += d
            # 更新所有节点到当前节点的距离最小值并更新下一个节点
            x, y = nums[i]
            nex = -1
            for j in rest:
                dj = dis(nums[j][0], nums[j][1], x, y)
                visit[j] = ac.min(visit[j], dct[i].get(j, inf))
                if dj < visit[j]:
                    visit[j] = dj
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
        # 时间复杂度O(n^2)空间复杂度O(n)优于kruskal
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p1194(ac=FastIO()):
        # 模板：使用超级源点建图计算最小生成树
        a, b = ac.read_ints()
        grid = [ac.read_list_ints() for _ in range(b)]
        edge = [[0, i, a] for i in range(1, b + 1)]
        for i in range(b):
            for j in range(i + 1, b):
                if 0 < grid[i][j] < a:
                    edge.append([i + 1, j + 1, grid[i][j]])
        mst = MinimumSpanningTree(edge, b+1)
        ac.st(mst.cost)
        return

    @staticmethod
    def cf_472d(ac=FastIO()):
        # 模板：使用 prim 校验最小生成树是否存在
        n = ac.read_int()
        grid = [ac.read_list_ints() for _ in range(n)]
        for i in range(n):
            if grid[i][i]:
                ac.st("NO")
                return
            for j in range(i + 1, n):
                if grid[i][j] != grid[j][i] or not grid[i][j]:
                    ac.st("NO")
                    return

        # Prim 贪心按照权值选择边进行连通合并
        dis = [float("inf")] * n
        dis[0] = 0
        visit = [0] * n
        stack = [[0, 0, -1]]
        res = [[] for _ in range(n)]
        cnt = 0
        while stack:
            d, i, fa = heapq.heappop(stack)
            if visit[i]:
                continue
            visit[i] = 1
            if fa != -1:
                res[fa].append([i, d])
                res[i].append([fa, d])
            cnt += 1
            if cnt == n:
                break
            for j in range(n):
                w = grid[i][j]
                if w < dis[j]:
                    dis[j] = w
                    heapq.heappush(stack, [w, j, i])
        del stack

        # BFS 计算根节点到所有节点的距离
        for i in range(n):
            cur = [inf] * n
            stack = [i]
            cur[i] = 0
            while stack:
                x = stack.pop()
                for y, w in res[x]:
                    if cur[y] == inf:
                        cur[y] = cur[x] + w
                        stack.append(y)
            if cur != grid[i]:
                ac.st("NO")
                return
        ac.st("YES")
        return

    @staticmethod
    def lg_p4180(ac=FastIO()):
        # 模板：使用最小生成树与LCA倍增查询计算严格次小生成树
        n, m = ac.read_ints()
        edges = []
        for _ in range(m):
            i, j, w = ac.read_ints()
            if i != j:  # 去除自环
                edges.append([i-1, j-1, w])

        # 计算kruskal最小生成树
        edges.sort(key=lambda it: it[2])
        uf = UnionFind(n)
        dct = [dict() for _ in range(n)]
        cost = 0
        for i, j, w in edges:
            if uf.union(i, j):
                cost += w
                dct[i][j] = dct[j][i] = w
            if uf.part == 1:
                break

        # 枚举新增的边
        tree = TreeAncestorWeightSecond(dct)
        ans = inf
        for i, j, w in edges:
            for dis in tree.get_dist_weight_max_second(i, j):
                if dis != -1:
                    cur = cost - dis + w
                    if cost < cur < ans:
                        ans = cur
        ac.st(ans)
        return

    @staticmethod
    def cf_609e(ac=FastIO()):
        # 模板：计算最小生成树有指定边参与时的最小权值和，由此也可计算严格次小生成树
        n, m = ac.read_ints()
        edges = []
        for _ in range(m):
            i, j, w = ac.read_ints()
            if i != j:  # 去除自环
                edges.append([i - 1, j - 1, w])

        # 计算kruskal最小生成树
        uf = UnionFind(n)
        dct = [dict() for _ in range(n)]
        cost = 0
        for i, j, w in sorted(edges, key=lambda it: it[2]):
            if uf.union(i, j):
                cost += w
                dct[i][j] = dct[j][i] = w
            if uf.part == 1:
                break

        # 枚举新增的边
        tree = TreeAncestorWeightSecond(dct)
        for i, j, w in edges:
            if j in dct[i] and dct[i][j] == w:
                ac.st(cost)
            else:
                dis = tree.get_dist_weight_max_second(i, j)[0]
                ac.st(cost-dis+w)
        return

    @staticmethod
    def lg_p1265(ac=FastIO()):
        # 模板：使用prim计算最小生成树，适合稠密图场景

        def dis(x1, y1, x2, y2):
            return (x1-x2)**2+(y1-y2)**2

        # 初始化最短距离
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        ans = nex = 0
        rest = set(list(range(1, n)))
        visit = [inf]*n
        visit[nex] = 0
        while rest:
            # 点优先选择距离当前集合最近的点合并
            i = nex
            rest.discard(i)
            d = visit[i]
            ans += d ** 0.5
            nex = -1
            # 更新所有节点到当前节点的距离最小值并更新下一个节点
            x, y = nums[i]
            for j in rest:
                dj = dis(nums[j][0], nums[j][1], x, y)
                if dj < visit[j]:
                    visit[j] = dj
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
        # 时间复杂度O(n^2)空间复杂度O(n)优于kruskal
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p1340(ac=FastIO()):
        # 模板：逆序并查集，维护最小生成树的边
        n, w = ac.read_ints()
        edges = [ac.read_list_ints() for _ in range(w)]
        ind = list(range(w))
        ind.sort(key=lambda it: edges[it][-1])

        uf = UnionFind(n)
        ans = []
        select = set()
        cost = 0
        for i in range(w-1, -1, -1):
            if uf.part > 1:
                cost = 0
                select = set()
                for j in ind:
                    if j <= i:
                        x, y, ww = edges[j]
                        if uf.union(x-1, y-1):
                            cost += ww
                            select.add(j)
            if uf.part > 1:
                ans.append(-1)
                break
            ans.append(cost)
            if i in select:
                uf = UnionFind(n)
                select = set()
                cost = 0
        while len(ans) < w:
            ans.append(-1)

        for i in range(w-1, -1, -1):
            ls = ans[i]
            ac.st(ls)
        return

    @staticmethod
    def lg_p1550(ac=FastIO()):
        # 模板：建立虚拟源点，转换为最小生成树问题
        n = ac.read_int()
        edges = []
        for i in range(n):
            w = ac.read_int()
            edges.append([0, i+1, w])

        for i in range(n):
            grid = ac.read_list_ints()
            for j in range(i+1, n):
                edges.append([i+1, j+1, grid[j]])
        edges.sort(key=lambda it: it[2])
        cost = 0
        uf = UnionFind(n+1)
        for i, j, c in edges:
            if uf.union(i, j):
                cost += c
            if uf.part == 1:
                break
        ac.st(cost)
        return

    @staticmethod
    def lg_p2212(ac=FastIO()):

        # 模板：使用prim计算最小生成树，适合稠密图场景
        def dis(x1, y1, x2, y2):
            res = (x1 - x2) ** 2 + (y1 - y2) ** 2
            return res if res >= c else inf

        n, c = ac.read_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        # 初始化最短距离
        ans = nex = 0
        rest = set(list(range(1, n)))
        visit = [inf] * n
        visit[nex] = 0
        while rest:
            # 点优先选择距离当前集合最近的点合并
            i = nex
            rest.discard(i)
            d = visit[i]
            ans += d
            nex = -1
            # 更新所有节点到当前节点的距离最小值并更新下一个节点
            x, y = nums[i]
            for j in rest:
                dj = dis(nums[j][0], nums[j][1], x, y)
                if dj < visit[j]:
                    visit[j] = dj
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
        # 时间复杂度O(n^2)空间复杂度O(n)优于kruskal
        ac.st(ans if ans < inf else -1)
        return


class TestGeneral(unittest.TestCase):

    def test_minimum_spanning_tree(self):
        n = 3
        edges = [[0, 1, 2], [1, 2, 3], [2, 0, 4]]
        mst = MinimumSpanningTree(edges, n)
        assert mst.cost == 5

        n = 4
        edges = [[0, 1, 2], [1, 2, 3], [2, 0, 4]]
        mst = MinimumSpanningTree(edges, n)
        assert mst.cost == -1
        return


if __name__ == '__main__':
    unittest.main()
