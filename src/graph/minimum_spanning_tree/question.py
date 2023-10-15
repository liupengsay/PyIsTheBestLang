import heapq
import math
import unittest
from collections import deque, defaultdict
from heapq import heappop, heappush
from typing import List

from src.fast_io import FastIO, inf
from src.graph.union_find import UnionFind

"""

算法：最小生成树（Kruskal算法和Prim算法两种）、严格次小生成树（使用LCA枚举替换边计算可得）、最短路生成树
功能：计算无向图边权值和最小的生成树
Prim在稠密图中比Kruskal优，在稀疏图中比Kruskal劣。Prim是以更新过的节点的连边找最小值，Kruskal是直接将边排序。
两者其实都是运用贪心的思路，Kruskal相对比较常用

题目：

===================================力扣===================================
1489. 找到最小生成树里的关键边和伪关键边（https://leetcode.cn/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/）计算最小生成树的关键边与伪关键边
1584. 连接所有点的最小费用（https://leetcode.cn/problems/min-cost-to-connect-all-points/）稠密图使用 prim 生成最小生成树
1724. 检查边长度限制的路径是否存在 II（https://leetcode.cn/problems/checking-existence-of-edge-length-limited-paths-ii/）经典使用最小生成树与倍增求解任意点对之间简单路径的最大边权值

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
P2847 [USACO16DEC]Moocast G（https://www.luogu.com.cn/problem/P2847）使用prim计算最小生成树，适合稠密图场景
P3535 [POI2012]TOU-Tour de Byteotia（https://www.luogu.com.cn/problem/P3535）最小生成树思想与并查集判环
P4047 [JSOI2010]部落划分（https://www.luogu.com.cn/problem/P4047）使用最小生成树进行最优聚类距离计算
P6171 [USACO16FEB]Fenced In G（https://www.luogu.com.cn/problem/P6171）稀疏图使用 Kruskal 计算最小生成树
P1550 [USACO08OCT] Watering Hole G（https://www.luogu.com.cn/problem/P1550）经典最小生成树，增加虚拟源点

================================CodeForces================================
D. Design Tutorial: Inverse the Problem（https://codeforces.com/problemset/problem/472/D）使用最小生成树判断构造给定的点对最短路距离是否存在，使用prim算法复杂度更优
E. Minimum spanning tree for each edge（https://codeforces.com/problemset/problem/609/E）使用LCA的思想维护树中任意两点的路径边权最大值，并贪心替换获得边作为最小生成树时的最小权值和，有点类似于关键边与非关键边，但二者并不相同，即为严格次小生成树
F. MST Unification（https://codeforces.com/contest/1108/problem/F）使得最小生成树的边组合唯一时，需要增加权重的最少边数量

===================================AtCoder===================================
D - Built?（https://atcoder.jp/contests/abc065/tasks/arc076_b）最小生成树变形问题

================================AcWing================================
3728. 城市通电（https://www.acwing.com/problem/content/3731/）使用prim计算最小生成树，适合稠密图场景，并获取具体连边方案，也可直接使用Kruskal（超时）

================================LibraryChecker================================
Manhattan MST（https://judge.yosupo.jp/problem/manhattanmst）
Directed MST（https://judge.yosupo.jp/problem/directedmst）

参考：OI WiKi（xx）
"""


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
        ans = cost ** 0.5
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
    def cf_1108f(ac=FastIO()):
        # 模板：使得最小生成树的边组合唯一时，需要增加权重的最少边数量
        n, m = ac.read_list_ints()
        edges = []
        for _ in range(m):
            i, j, w = ac.read_list_ints()
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
        del uf
        # 枚举新增的边
        tree = TreeAncestorWeightSecond(dct)
        ans = 0
        # 使得最小生成树唯一等价于有某条边参与时依旧代价最小的该边数量
        for i, j, w in edges:
            if j in dct[i] and dct[i][j] == w:
                ans += 1
            else:
                dis = tree.get_dist_weight_max_second(i, j)[0]
                if dis == w:
                    ans += 1
        ac.st(ans - n + 1)
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
            return res ** 0.5

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
        mst = MinimumSpanningTree(edge, b + 1)
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
        dis = [inf] * n
        dis[0] = 0
        visit = [0] * n
        stack = [[0, 0, -1]]
        res = [[] for _ in range(n)]
        cnt = 0
        while stack:
            d, i, fa = heappop(stack)
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
                    heappush(stack, [w, j, i])
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
                edges.append([i - 1, j - 1, w])

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
                ac.st(cost - dis + w)
        return

    @staticmethod
    def lg_p1265(ac=FastIO()):
        # 模板：使用prim计算最小生成树，适合稠密图场景

        def dis(x1, y1, x2, y2):
            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        # 初始化最短距离
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        ans = nex = 0
        rest = set(list(range(1, n)))
        visit = [inf] * n
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

        # 离线查询处理，按照边权排序
        edges = [ac.read_list_ints() for _ in range(w)]
        ind = list(range(w))
        ind.sort(key=lambda it: edges[it][-1])

        uf = UnionFind(n)
        ans = []
        select = set()
        cost = 0
        for i in range(w - 1, -1, -1):
            if uf.part > 1:
                # 重新生成最小生成树
                cost = 0
                select = set()
                for j in ind:
                    if j <= i:
                        x, y, ww = edges[j]
                        if uf.union(x - 1, y - 1):
                            cost += ww
                            select.add(j)
            if uf.part > 1:
                # 无法连通直接终止
                ans.append(-1)
                break
            ans.append(cost)
            if i in select:  # 当前路径不可用，重置并查集
                uf = UnionFind(n)
                select = set()
                cost = 0
        while len(ans) < w:
            ans.append(-1)

        for i in range(w - 1, -1, -1):
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
            edges.append([0, i + 1, w])

        for i in range(n):
            grid = ac.read_list_ints()
            for j in range(i + 1, n):
                edges.append([i + 1, j + 1, grid[j]])
        edges.sort(key=lambda it: it[2])
        cost = 0
        uf = UnionFind(n + 1)
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

    @staticmethod
    def lg_p2658(ac=FastIO()):
        # 模板：典型最小生成树计算
        m, n = ac.read_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        uf = UnionFind(m * n)
        s = 0
        for i in range(m):
            lst = ac.read_list_ints()
            for j in range(n):
                if lst[j]:
                    uf.size[i * n + j] = 1
                    s += 1
        edge = []
        for i in range(m):
            for j in range(n):
                if i + 1 < m:
                    edge.append([i * n + j, i * n + j + n, abs(grid[i][j] - grid[i + 1][j])])
                if j + 1 < n:
                    edge.append([i * n + j, i * n + j + 1, abs(grid[i][j] - grid[i][j + 1])])
        edge.sort(key=lambda it: it[2])
        del grid

        if s == 1:
            ac.st(0)
            return
        for x, y, d in edge:
            uf.union(x, y)
            if uf.size[uf.find(x)] == s:
                ac.st(d)
                return
        return

    @staticmethod
    def lg_p2847(ac=FastIO()):

        # 模板：使用prim计算最小生成树，适合稠密图场景
        def dis(x1, y1, x2, y2):
            res = (x1 - x2) ** 2 + (y1 - y2) ** 2
            return res

        n = ac.read_int()
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
            ans = ac.max(ans, d)
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

    @staticmethod
    def lg_p3535(ac=FastIO()):
        # 模板：最小生成树思想与并查集判环
        n, m, k = ac.read_ints()
        edge = []
        uf = UnionFind(n)
        ans = 0
        for _ in range(m):
            # 先将大于等于 k 的连接起来
            i, j = ac.read_ints_minus_one()
            if i >= k and j >= k:
                uf.union(i, j)
            else:
                edge.append([i, j])
        # 再依次判断剩余的边
        for i, j in edge:
            if not uf.union(i, j):
                ans += 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p4047(ac=FastIO()):

        def dis():
            return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

        # 模板：使用最小生成树进行最优聚类距离计算
        n, k = ac.read_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        edge = []
        for i in range(n):
            x1, y1 = nums[i]
            for j in range(i + 1, n):
                x2, y2 = nums[j]
                edge.append([i, j, dis()])
        edge.sort(key=lambda it: it[2])

        # 当分成 k 个联通块时计算最短距离
        uf = UnionFind(n)
        ans = 0
        for i, j, d in edge:
            if uf.part == k:
                if not uf.is_connected(i, j):
                    ans = d
                    break
            else:
                uf.union(i, j)
        ans = ans ** 0.5
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p6171(ac=FastIO()):
        # 模板：稀疏图使用 Kruskal 计算最小生成树
        a, b, n, m = ac.read_ints()
        nums1 = [0, a] + [ac.read_int() for _ in range(n)]
        nums2 = [0, b] + [ac.read_int() for _ in range(m)]
        nums1.sort()
        nums2.sort()
        dct = defaultdict(list)
        for i in range(1, m + 2):
            for j in range(1, n + 2):
                # 建图是关键
                x = (i - 1) * (n + 1) + (j - 1)
                if i + 1 < m + 2:
                    y = i * (n + 1) + (j - 1)
                    dct[nums1[j] - nums1[j - 1]].append([x, y])
                if j + 1 < n + 2:
                    y = (i - 1) * (n + 1) + j
                    dct[nums2[i] - nums2[i - 1]].append([x, y])
        ans = 0
        uf = UnionFind((n + 1) * (m + 1))
        for c in sorted(dct):
            for i, j in dct[c]:
                if uf.union(i, j):
                    ans += c
        ac.st(ans)
        return

    @staticmethod
    def lc_1584_1(nums: List[List[int]]) -> int:

        # 模板：使用prim计算最小生成树，适合稠密图场景
        def dis(x1, y1, x2, y2):
            res = abs(x1 - x2) + abs(y1 - y2)
            return res

        n = len(nums)
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
        return ans

    @staticmethod
    def lc_1584_2(nums: List[List[int]]) -> int:

        # 模板：使用prim计算最小生成树，适合稠密图场景
        def dis(x1, y1, x2, y2):
            res = abs(x1 - x2) + abs(y1 - y2)
            return res

        n = len(nums)
        edges = []
        for i in range(n):
            x1, y1 = nums[i]
            for j in range(i + 1, n):
                x2, y2 = nums[j]
                edges.append([i, j, dis(x1, y1, x2, y2)])

        tree = MinimumSpanningTree(edges, n, "prim")
        return tree.cost

    @staticmethod
    def lg_p1556(ac=FastIO()):
        # 模板：经典最小生成树，增加虚拟源点
        n = ac.read_int()
        edges = []
        for i in range(n):
            w = ac.read_int()
            edges.append([0, i + 1, w])
            # 虚拟源点
        for i in range(n):
            grid = ac.read_list_ints()
            for j in range(i + 1, n):
                edges.append([i + 1, j + 1, grid[j]])
        # kruskal最小生成树
        edges.sort(key=lambda it: it[2])
        cost = 0
        uf = UnionFind(n + 1)
        for i, j, c in edges:
            if uf.union(i, j):
                cost += c
            if uf.part == 1:
                break
        ac.st(cost)
        return

    @staticmethod
    def abc_65d(ac=FastIO()):
        # 模板：最小生成树变形问题
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        ind = list(range(n))
        ind.sort(key=lambda it: nums[it][0])
        edges = []
        for i in range(1, n):
            x, y = ind[i - 1], ind[i]
            d = nums[y][0] - nums[x][0]
            edges.append([x, y, d])
        uf = UnionFind(n)
        ind.sort(key=lambda it: nums[it][1])
        for i in range(1, n):
            x, y = ind[i - 1], ind[i]
            d = nums[y][1] - nums[x][1]
            edges.append([x, y, d])
        edges.sort(key=lambda it: it[2])
        ans = 0
        for i, j, d in edges:
            if uf.union(i, j):
                ans += d
        ac.st(ans)
        return

    @staticmethod
    def ac_3728(ac=FastIO()):

        # 模板：使用prim计算最小生成树，适合稠密图场景，并获取具体连边方案，也可直接使用Kruskal（超时）

        def dis(aa, bb):
            if aa == 0:
                return cost[bb]
            if bb == 0:
                return cost[aa]

            return (k[aa] + k[bb]) * (abs(nums[aa][0] - nums[bb][0]) + abs(nums[aa][1] - nums[bb][1]))

        n = ac.read_int()
        nums = [[inf, inf]] + [ac.read_list_ints() for _ in range(n)]
        cost = [inf] + ac.read_list_ints()
        k = [inf] + ac.read_list_ints()

        # 初始化最短距离
        ans = nex = 0
        rest = set(list(range(n + 1)))
        visit = [inf] * (n + 1)
        visit[nex] = 0
        pre = [-1] * (n + 1)  # 记录最小生成树的父节点
        edge = []
        while rest:
            # 点优先选择距离当前集合最近的点合并
            i = nex
            rest.discard(i)
            d = visit[i]
            ans += d
            nex = -1
            # 更新所有节点到当前节点的距离最小值并更新下一个节点
            for j in rest:
                dj = dis(i, j)
                if dj < visit[j]:
                    visit[j] = dj
                    pre[j] = i
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
            if nex != -1:
                edge.append([pre[nex], nex])
        # 时间复杂度O(n^2)空间复杂度O(n)优于kruskal
        ac.st(ans if ans < inf else -1)
        lst = []
        for a, b in edge:
            if a == 0:
                lst.append(b)
            elif b == 0:
                lst.append(a)
        ac.st(len(lst))
        ac.lst(lst)
        ac.st(len(edge) - len(lst))
        for a, b in edge:
            if a and b:
                ac.lst([a, b])
        return

