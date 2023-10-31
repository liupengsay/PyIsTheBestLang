from heapq import heappop, heappush
from math import inf
from typing import List

from src.basis.binary_search.template import BinarySearch
from src.graph.dijkstra.template import Dijkstra
from src.utils.fast_io import FastIO

"""

算法：Floyd（多源最短路经算法）、可以处理有向图无向图以及正负权边、也可以检测负环
功能：计算点到有向或者无向图里面其他点的最短路，也可以计算最长路，以及所有最长路最短路上经过的点（关键节点）
方案： Floyd 就要记录 dp[i][j]对 应的 pre[i][j] = k; 而 Bellman-Ford 和 Dijkstra 一般记录 pre[v] = u
参考：OI WiKi（https://oi-wiki.org/graph/shortest-path/）
题目：

===================================力扣===================================
2642. 设计可以求最短路径的图类（https://leetcode.cn/problems/design-graph-with-shortest-path-calculator/）Floyd动态更新最短路
1462. 课程表 IV（https://leetcode.cn/problems/course-schedule-iv/）可考虑使用传递闭包Floyd求解

===================================洛谷===================================
P1119 灾后重建 （https://www.luogu.com.cn/problem/P1119）离线查询加Floyd动态更新经过中转站的起终点距离，修复增加维护的是点
P1476 休息中的小呆（https://www.luogu.com.cn/problem/P1476）Floyd 求索引从 1 到 n 的最长路并求所有在最长路上的点
P3906 Geodetic集合（https://www.luogu.com.cn/problem/P3906）Floyd算法计算最短路径上经过的点集合

P2009 跑步（https://www.luogu.com.cn/problem/P2009）Floyd求最短路
P2419 [USACO08JAN]Cow Contest S（https://www.luogu.com.cn/problem/P2419）看似拓扑排序其实是使用Floyd进行拓扑排序
P2910 [USACO08OPEN]Clear And Present Danger S（https://www.luogu.com.cn/problem/P2910）最短路计算之后进行查询，Floyd模板题
P6464 [传智杯 #2 决赛] 传送门（https://www.luogu.com.cn/problem/P6464）枚举边之后进行Floyd算法更新计算，经典理解Floyd的原理题，经典借助中间两点更新最短距离
P6175 无向图的最小环问题（https://www.luogu.com.cn/problem/P6175）经典使用Floyd枚举三个点之间的距离和，O(n^3)，也可以使用BFS或者Dijkstra计算
B3611 【模板】传递闭包（https://www.luogu.com.cn/problem/B3611）传递闭包模板题，使用FLoyd解法
P1613 跑路（https://www.luogu.com.cn/problem/P1613）经典Floyd动态规划，使用两遍最短路综合计算
P8312 [COCI2021-2022#4] Autobus（https://www.luogu.com.cn/problem/P8312）经典最多k条边的最短路跑k遍Floyd
P8794 [蓝桥杯 2022 国 A] 环境治理（https://www.luogu.com.cn/problem/P8794）经典二分加Floyd计算


================================CodeForces================================
D. Design Tutorial: Inverse the Problem（https://codeforces.com/problemset/problem/472/D）使用Floyd判断构造给定的点对最短路距离是否存在

================================AtCoder================================
D - Candidates of No Shortest Paths（https://atcoder.jp/contests/abc051/tasks/abc051_d）经典Floyd计算最短路的必经边
D - Restoring Road Network（https://atcoder.jp/contests/abc074/tasks/arc083_b）经典最短路生成图，使用Floyd维护最小生成图
E - Travel by Car（https://atcoder.jp/contests/abc143/tasks/abc143_e）Floyd建图最短路，两种最短路，建两次图

===================================AcWing===================================
4872. 最短路之和（https://www.acwing.com/problem/content/submission/4875/）经典Floyd逆序逆向思维更新最短路对

"""


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1613(ac=FastIO()):
        # 模板：经典Floyd动态规划，使用两遍最短路综合计算
        n, m = ac.read_list_ints()

        # dp[i][j][k] 表示 i 到 j 有无花费为 k 秒即距离为 2**k 的的路径
        dp = [[[0] * 32 for _ in range(n)] for _ in range(n)]
        for _ in range(m):
            u, v = ac.read_list_ints_minus_one()
            dp[u][v][0] = 1
        for x in range(1, 32):
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        if dp[i][k][x - 1] and dp[k][j][x - 1]:
                            dp[i][j][x] = 1

        # 建立距离二进制 1 的个数为 1 的有向图
        dis = [[inf] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for x in range(32):
                    if dp[i][j][x]:
                        dis[i][j] = 1
                        break

        # 第二遍 Floyd 求新距离意义上的最短路
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dis[i][j] = ac.min(dis[i][j], dis[i][k] + dis[k][j])
        ac.st(dis[0][n - 1])

        return

    @staticmethod
    def ac_4872(ac=FastIO()):
        # 模板：经典Floyd逆序逆向思维更新最短路对
        n = ac.read_int()
        dp = [ac.read_list_ints() for _ in range(n)]
        a = ac.read_list_ints_minus_one()
        node = []
        ans = []
        for ind in range(n-1, -1, -1):
            x = a[ind]
            node.append(x)
            cur = 0
            for i in node:
                for j in node:
                    dp[i][x] = ac.min(dp[i][x], dp[i][j]+dp[j][x])
                    dp[x][i] = ac.min(dp[x][i], dp[x][j]+dp[j][i])

            for i in node:
                for j in node:
                    dp[i][j] = ac.min(dp[i][j], dp[i][x]+dp[x][j])
                    cur += dp[i][j]
            ans.append(cur)

        ac.lst(ans[::-1])
        return

    @staticmethod
    def lg_p1119(ac=FastIO()):
        # 模板：利用 Floyd 算法特点和修复的中转站更新最短距离（无向图）
        n, m = ac.read_list_ints()
        repair = ac.read_list_ints()
        # 设置初始值距离
        dis = [[inf] * n for _ in range(n)]
        for i in range(m):
            a, b, c = ac.read_list_ints()
            dis[a][b] = dis[b][a] = c
        for i in range(n):
            dis[i][i] = 0

        # 修复村庄之后用Floyd算法更新以该村庄为中转的距离
        k = 0
        for _ in range(ac.read_int()):
            x, y, t = ac.read_list_ints()
            # 离线算法
            while k < n and repair[k] <= t:
                # k修复则更新以k为中转站的距离
                for a in range(n):
                    for b in range(a + 1, n):
                        dis[a][b] = dis[b][a] = ac.min(dis[a][k] + dis[k][b], dis[b][a])
                k += 1
            if dis[x][y] < inf and x < k and y < k:
                ac.st(dis[x][y])
            else:
                ac.st(-1)
        return

    @staticmethod
    def lg_p1476(ac=FastIO()):
        # 模板：Floyd 求索引从 1 到 n 的最长路并求所有在最长路上的点（有向图）
        n = ac.read_int() + 1
        m = ac.read_int()
        dp = [[-inf] * (n + 1) for _ in range(n + 1)]
        for _ in range(m):
            i, j, k = ac.read_list_ints()
            dp[i][j] = k
        for i in range(n + 1):
            dp[i][i] = 0

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                for k in range(1, n + 1):
                    if dp[i][j] < dp[i][k] + dp[k][j]:
                        dp[i][j] = dp[i][k] + dp[k][j]

        length = dp[1][n]
        path = []
        for i in range(1, n + 1):
            if dp[1][i] + dp[i][n] == dp[1][n]:
                path.append(i)
        ac.st(length)
        ac.lst(path)
        return

    @staticmethod
    def lg_p3906(ac=FastIO()):
        # 模板：Floyd 求索引从 u 到 v 的最短路并求所有在最短路上的点（无向图）
        n, m = ac.read_list_ints()
        dp = [[inf] * (n + 1) for _ in range(n + 1)]
        for _ in range(m):
            i, j = ac.read_list_ints()
            dp[i][j] = dp[j][i] = 1
        for i in range(1, n + 1):
            dp[i][i] = 0

        for k in range(1, n + 1):
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):  # 无向图这里就可以优化
                    dp[j][i] = dp[i][j] = ac.min(dp[i][j], dp[i][k] + dp[k][j])

        for _ in range(ac.read_int()):
            u, v = ac.read_list_ints()
            dis = min(dp[u][k] + dp[k][v] for k in range(1, n + 1))
            ac.lst([x for x in range(1, n + 1) if dp[u][x] + dp[x][v] == dis])
        return

    @staticmethod
    def lg_b3611(ac=FastIO()):
        # 模板：传递闭包模板题
        n = ac.read_int()
        dp = [ac.read_list_ints() for _ in range(n)]
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dp[i][k] and dp[k][j]:
                        dp[i][j] = 1
        for g in dp:
            ac.lst(g)
        return

    @staticmethod
    def abc_51d_1(ac=FastIO()):
        # 模板：经典脑筋急转弯Floyd计算最短路的必经边，也可直接使用Dijkstra
        n, m = ac.read_list_ints()
        dp = [[inf]*n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 0

        edges = [ac.read_list_ints() for _ in range(m)]
        for i, j, w in edges:
            i -= 1
            j -= 1
            dp[i][j] = dp[j][i] = w

        for k in range(n):  # 中间节点
            for i in range(n):  # 起始节点
                for j in range(i+1, n):  # 结束节点
                    a, b = dp[i][j], dp[i][k] + dp[k][j]
                    dp[i][j] = dp[j][i] = a if a < b else b
        ans = 0
        for i, j, w in edges:
            i -= 1
            j -= 1
            if dp[i][j] < w:  # 如果最短距离小于该边则必然不经过该边
                ans += 1
        ac.st(ans)
        return

    @staticmethod
    def abc_51d_2(ac=FastIO()):
        # 模板：经典脑筋急转弯Floyd计算最短路的必经边，也可直接使用Dijkstra
        n, m = ac.read_list_ints()
        edges = [ac.read_list_ints() for _ in range(m)]
        dct = [[] for _ in range(n)]
        for i, j, w in edges:
            i -= 1
            j -= 1
            dct[i].append([j, w])
            dct[j].append([i, w])
        dis = []
        for i in range(n):
            dis.append(Dijkstra().get_dijkstra_result(dct, i))
        ans = 0
        for i, j, w in edges:
            i -= 1
            j -= 1
            if dis[i][j] < w:
                ans += 1
        ac.st(ans)
        return

    @staticmethod
    def abc_74d(ac=FastIO()):
        # 模板：经典最短路生成图，使用Floyd维护最小生成图
        n = ac.read_int()
        grid = [ac.read_list_ints() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if grid[i][j] != grid[j][i]:
                    ac.st(-1)
                    return
            if grid[i][i]:
                ac.st(-1)
                return

        edges = []
        for i in range(n):
            for j in range(i+1, n):
                edges.append([i, j, grid[i][j]])
        edges.sort(key=lambda it: it[2])
        ans = 0
        dis = [[inf]*n for _ in range(n)]
        for i in range(n):
            dis[i][i] = 0
        # 逐渐更新最短距离
        for i, j, w in edges:
            if dis[i][j] < grid[i][j]:
                ac.st(-1)
                return
            if dis[i][j] == w:
                continue
            ans += w
            for x in range(n):
                for y in range(x+1, n):
                    a, b = dis[x][y], dis[x][i]+w+dis[j][y]
                    a = a if a < b else b
                    b = dis[x][j]+w+dis[i][y]
                    a = a if a < b else b
                    dis[x][y] = dis[y][x] = a
        ac.st(ans)
        return

    @staticmethod
    def abc_143e(ac=FastIO()):
        # 模板：Floyd建图最短路，两种最短路，建两次图
        n, m, ll = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        dis = [[inf] * n for _ in range(n)]
        for i in range(n):
            dis[i][i] = 0
        for _ in range(m):
            x, y, z = ac.read_list_ints()
            x -= 1
            y -= 1
            dct[x].append([y, z])
            dct[y].append([x, z])
            a, b = dis[x][y], z
            dis[x][y] = dis[y][x] = a if a < b else b

        for k in range(n):
            for i in range(n):
                for j in range(i + 1, n):
                    cur = dis[i][k]+dis[k][j]
                    if cur < dis[i][j]:
                        dis[i][j] = dis[j][i] = cur

        dp = [[inf]*n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 0
            for j in range(i+1, n):
                if dis[i][j] <= ll:
                    dp[i][j] = dp[j][i] = 0

        for k in range(n):
            for i in range(n):
                for j in range(i + 1, n):
                    cur = dp[i][k]+dp[k][j]+1
                    if cur < dp[i][j]:
                        dp[i][j] = dp[j][i] = cur

        for _ in range(ac.read_int()):
            x, y = ac.read_list_ints_minus_one()
            ans = dp[x][y]
            ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def cf_472d(ac=FastIO()):
        # 模板：使用 Floyd 的思想判断最短路矩阵是否合理存在
        n = ac.read_int()
        grid = [ac.read_list_ints() for _ in range(n)]
        for i in range(n):
            if grid[i][i]:
                ac.st("NO")
                return
            for j in range(i+1, n):
                if grid[i][j] != grid[j][i] or not grid[i][j]:
                    ac.st("NO")
                    return
        if n == 1:
            ac.st("YES")
            return
        for i in range(n):
            r = 1 if not i else 0
            for j in range(n):
                if grid[i][j] < grid[i][r] and i != j:
                    r = j
            for k in range(n):
                if abs(grid[i][k]-grid[r][k]) != grid[i][r]:
                    ac.st("NO")
                    return
        ac.st("YES")
        return

    @staticmethod
    def lg_p1613(ac=FastIO()):
        # 模板：建立新图计算Floyd最短路
        n, m = ac.read_list_ints()

        # 表示节点i与j之间距离为2^k的路径是否存在
        dp = [[[0] * 32 for _ in range(n)] for _ in range(n)]
        for _ in range(m):
            u, v = ac.read_list_ints_minus_one()
            dp[u][v][0] = 1

        # 结合倍增思想进行Floyd建新图
        for x in range(1, 32):
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        if dp[i][k][x - 1] and dp[k][j][x - 1]:
                            dp[i][j][x] = 1

        dis = [[inf] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for x in range(32):
                    if dp[i][j][x]:
                        dis[i][j] = 1
                        break

        # Floyd新图计算最短路
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dis[i][j] = ac.min(dis[i][j], dis[i][k] + dis[k][j])
        ac.st(dis[0][n - 1])
        return

    @staticmethod
    def lg_p8312(ac=FastIO()):
        # 模板：经典最多k条边的最短路跑k遍Floyd
        n, m = ac.read_list_ints()
        dis = [[inf] * n for _ in range(n)]
        for i in range(n):
            dis[i][i] = 0

        for _ in range(m):
            a, b, c = ac.read_list_ints_minus_one()
            c += 1
            dis[a][b] = ac.min(dis[a][b], c)

        dct = [d[:] for d in dis]
        k, q = ac.read_list_ints()
        nums = [ac.read_list_ints_minus_one() for _ in range(q)]
        k = ac.min(k, n)
        for _ in range(k - 1):
            cur = [d[:] for d in dis]
            for p in range(n):
                for i in range(n):
                    for j in range(n):
                        cur[i][j] = ac.min(cur[i][j], dis[i][p] + dct[p][j])
            dis = [d[:] for d in cur]

        for c, d in nums:
            res = dis[c][d]
            ac.st(res if res < inf else -1)
        return

    @staticmethod
    def lg_p8794(ac=FastIO()):
        # 模板：经典二分加Floyd计算

        def get_dijkstra_result_mat(mat: List[List[int]], src: int) -> List[float]:
            # 模板: Dijkstra求最短路，变成负数求可以求最长路（还是正权值）
            len(mat)
            dis = [inf] * n
            stack = [[0, src]]
            dis[src] = 0
            visit = set(list(range(n)))
            while stack:
                d, ii = heappop(stack)
                if dis[ii] < d:
                    continue
                visit.discard(ii)
                for j in visit:
                    dj = mat[ii][j] + d
                    if dj < dis[j]:
                        dis[j] = dj
                        heappush(stack, [dj, j])
            return dis

        n, q = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(n)]
        lower = [ac.read_list_ints() for _ in range(n)]
        ans = 0
        for i in range(n):
            ans += sum(get_dijkstra_result_mat(lower, i))
        if ans > q:
            ac.st(-1)
            return

        ans = 0
        for i in range(n):
            ans += sum(get_dijkstra_result_mat(grid, i))
        if ans <= q:
            ac.st(0)
            return

        def check(x):
            cnt = [x // n] * n
            for y in range(x % n):
                cnt[y] += 1
            cur = [[0] * n for _ in range(n)]
            for a in range(n):
                for b in range(n):
                    cur[a][b] = ac.max(lower[a][b], grid[a][b] - cnt[a] - cnt[b])
            dis = 0
            for y in range(n):
                dis += sum(get_dijkstra_result_mat(cur, y))
            return dis <= q

        def check2(x):
            cnt = [x // n] * n
            for y in range(x % n):
                cnt[y] += 1
            cur = [[0] * n for _ in range(n)]
            for a in range(n):
                for b in range(n):
                    cur[a][b] = ac.max(lower[a][b], grid[a][b] - cnt[a] - cnt[b])
            # Floyd计算全源最短路
            for k in range(n):
                for a in range(n):
                    for b in range(a+1, n):
                        cur[a][b] = cur[b][a] = ac.min(cur[a][b], cur[a][k]+cur[k][b])
            return sum(sum(c) for c in cur) <= q

        low = 1
        high = n * 10**5
        BinarySearch().find_int_left(low, high, check)
        ans = BinarySearch().find_int_left(low, high, check2)
        ac.st(ans)
        return
