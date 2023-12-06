"""
Algorithm：Dijkstra（单源最短路经算法）、严格次短路、要保证加和最小因此只支持非负数权值、或者取反全部为非正数计算最长路、最短路生成树
Function：计算点到有向或者无向图里面其他点的最近距离、带约束的最短路、分层Dijkstra、有向图最小环、无向图最小环

====================================LeetCode====================================
42（https://leetcode.com/problems/trapping-rain-water/）一维接雨水，计算前后缀最大值的最小值再减去自身值
407（https://leetcode.com/problems/trapping-rain-water-ii/）经典最短路变种问题，求解路径上边权的最大值
787（https://leetcode.com/problems/cheapest-flights-within-k-stops/）使用带约束的最短路计算最终结果
1293（https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/）使用带约束的最短路计算最终结果
2203（https://leetcode.com/problems/minimum-weighted-subgraph-with-the-required-paths/）使用三个Dijkstra最短路获得结果
2258（https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/）使用双源BFS计算等待时间后最短路求出路径上最小等待时间的最大值
2290（https://leetcode.com/problems/minimum-obstacle-removal-to-reach-corner/）计算最小代价
499（https://leetcode.com/problems/the-maze-iii/?envType=study-plan-v2&id=premium-algo-100）两个参数变量的最短路
6442（https://leetcode.com/problems/modify-graph-edge-weights/）经典两遍最短路，贪心动态更新路径权值
2714（https://leetcode.com/problems/find-shortest-path-with-k-hops/）经典带约束的最短路，也可以使用分层Dijkstra求解
2699（https://leetcode.com/problems/modify-graph-edge-weights/）经典Dijkstra最短路贪心应用
1786（https://leetcode.com/problems/number-of-restricted-paths-from-first-to-last-node/）经典dijkstra受限最短路计数（类似最短路计数），也可以将无向图转换为DAG问题
1928（https://leetcode.com/problems/minimum-cost-to-reach-destination-in-time/）经典Dijkstra带约束的最短路，也可根据无后效性类似Floyd的动态规划求解
LCP 75（https://leetcode.com/problems/rdmXM7/）首先BFS之后计算最大值最小的最短路
1976（https://leetcode.com/problems/number-of-ways-to-arrive-at-destination/）经典Dijkstra最短路计数模板题
2045（https://leetcode.com/problems/second-minimum-time-to-reach-destination/）严格次短路计算模板题，距离更新时需要注意变化
2093（https://leetcode.com/problems/minimum-cost-to-reach-city-with-discounts/）经典Dijkstra带约束的最短路
882（https://leetcode.com/problems/reachable-nodes-in-subdivided-graph/description/）Dijkstra模板题
2577（https://leetcode.com/problems/minimum-time-to-visit-a-cell-in-a-grid/）Dijkstra经典变形二维矩阵题目
2065（https://leetcode.com/problems/maximum-path-quality-of-a-graph/）经典回溯，正解使用Dijkstra跑最短路剪枝

=====================================LuoGu======================================
3371（https://www.luogu.com.cn/problem/P3371）最短路模板题
4779（https://www.luogu.com.cn/problem/P4779）最短路模板题
1629（https://www.luogu.com.cn/problem/P1629）正反两个方向的最短路进行计算往返路程
1462（https://www.luogu.com.cn/problem/P1462）使用带约束的最短路计算最终结果
1339（https://www.luogu.com.cn/problem/P1339）标准最短路计算
1342（https://www.luogu.com.cn/problem/P1342）正反两遍最短路
1576（https://www.luogu.com.cn/problem/P1576）堆优化转换成负数求最短路
1821（https://www.luogu.com.cn/problem/P1821）正反两遍最短路
1882（https://www.luogu.com.cn/problem/P1882）转换为最短路求解最短路距离最远的点
1907（https://www.luogu.com.cn/problem/P1907）自定义建图计算最短路
1744（https://www.luogu.com.cn/problem/P1744）裸题最短路
1529（https://www.luogu.com.cn/problem/P1529）裸题最短路
1649（https://www.luogu.com.cn/problem/P1649）自定义距离计算的最短路
2083（https://www.luogu.com.cn/problem/P2083）反向最短路
2299（https://www.luogu.com.cn/problem/P2299）最短路裸题
2683（https://www.luogu.com.cn/problem/P2683）最短路裸题结合并查集查询
1396（https://www.luogu.com.cn/problem/P1396）最短路变种问题，求解路径上边权的最大值，类似接雨水
1346（https://www.luogu.com.cn/problem/P1346）建图跑最短路
1339（https://www.luogu.com.cn/record/list?user=739032&status=12&page=11）最短路裸题
2784（https://www.luogu.com.cn/problem/P2784）最大乘积的路径
1318（https://www.luogu.com.cn/problem/P1318）一维接雨水，计算前后缀最大值的最小值再减去自身值
2888（https://www.luogu.com.cn/problem/P2888）最短路计算路径上最大边权值最小的路径
2935（https://www.luogu.com.cn/problem/P2935）最短路裸题
2951（https://www.luogu.com.cn/problem/P2951）最短路裸题
2984（https://www.luogu.com.cn/problem/P2984）最短路裸题
3003（https://www.luogu.com.cn/problem/P3003）三遍最短路
3094（https://www.luogu.com.cn/problem/P3094）预处理最短路之后进行查询
3905（https://www.luogu.com.cn/problem/P3905）逆向思维，重新建图后计算最短路
5764（https://www.luogu.com.cn/problem/P5764）五遍最短路裸题计算
5767（https://www.luogu.com.cn/problem/P5767）经典建图求解公交与地铁的最短换乘
6770（https://www.luogu.com.cn/problem/P6770）最短路裸题
6833（https://www.luogu.com.cn/problem/P6833）三遍最短路后，进行枚举计算
7551（https://www.luogu.com.cn/problem/P7551）最短路裸题，注意重边与自环
6175（https://www.luogu.com.cn/problem/P6175）使用Dijkstra枚举边计算或者使用DFS枚举点，带权
4568（https://www.luogu.com.cn/problem/P4568）K层建图计算Dijkstra最短路
2865（https://www.luogu.com.cn/problem/P2865）严格次短路模板题
2622（https://www.luogu.com.cn/problem/P2622）状压加dijkstra最短路计算
1073（https://www.luogu.com.cn/problem/P1073）正反两遍建图，Dijkstra进行计算路径最大最小值
1300（https://www.luogu.com.cn/problem/P1300）Dijkstra求最短路
1354（https://www.luogu.com.cn/problem/P1354）建图Dijkstra求最短路
1608（https://www.luogu.com.cn/problem/P1608）使用Dijkstra计算有向与无向、带权与不带权的最短路数量
1828（https://www.luogu.com.cn/problem/P1828）多个单源Dijkstra最短路计算
2047（https://www.luogu.com.cn/problem/P2047）Dijkstra计算经过每个点的所有最短路条数占比，也可以使用Floyd进行计算
2269（https://www.luogu.com.cn/problem/P2269）比较两个项的最短路计算
2349（https://www.luogu.com.cn/problem/P2349）比较两个项相加的最短路
2914（https://www.luogu.com.cn/problem/P2914）Dijkstra动态建图计算距离
3020（https://www.luogu.com.cn/problem/P3020）Dijkstra求最短路
3057（https://www.luogu.com.cn/problem/P3057）Dijkstra求最短路
3753（https://www.luogu.com.cn/problem/P3753）最短路变形两个维度的比较
3956（https://www.luogu.com.cn/problem/P3956）多维状态的Dijkstra
4880（https://www.luogu.com.cn/problem/P4880）枚举终点使用 Dijkstra计算最短路
4943（https://www.luogu.com.cn/problem/P4943）枚举路径跑四遍最短路
5201（https://www.luogu.com.cn/problem/P5201）经典最短路生成树建图，再使用树形 DP 计算最优解
5663（https://www.luogu.com.cn/problem/P5663）经典最短路变形题目，计算最短的奇数与偶数距离
5683（https://www.luogu.com.cn/problem/P5683）计算三遍最短路枚举中间节点到三者之间的距离
5837（https://www.luogu.com.cn/problem/P5837）经典Dijkstra变形问题，带多个状态
5930（https://www.luogu.com.cn/problem/P5930）经典Dijkstra应用接雨水
6063（https://www.luogu.com.cn/problem/P6063）经典Dijkstra应用接雨水
6512（https://www.luogu.com.cn/problem/P6512）经典最短路加DP
8385（https://www.luogu.com.cn/problem/P8385）经典脑筋急转弯建图最短路
8724（https://www.luogu.com.cn/problem/P8724）分层最短路Dijkstra计算
8802（https://www.luogu.com.cn/problem/P8802）Dijkstra基础权重变形题
2176（https://www.luogu.com.cn/problem/P2176）枚举最短路上的边修改后，重新计算最短路

===================================CodeForces===================================
20C（https://codeforces.com/problemset/problem/20/C）正权值最短路计算，并记录返回生成路径
1343E（https://codeforces.com/problemset/problem/1343/E）使用三个01BFS求最短路加贪心枚举计算
715B（https://codeforces.com/contest/715/problem/B）经典两遍最短路，贪心动态更新路径权值
1433G（https://codeforces.com/contest/1433/problem/G）经典全源Dijkstra最短路枚举
1650G（https://codeforces.com/contest/1650/problem/G）经典Dijkstra最短路与严格次短路计数，正解为01BFS

====================================AtCoder=====================================
F - Pure（https://atcoder.jp/contests/abc142/tasks/abc142_f）经典子图寻找，转换为有向图的最小环问题

=====================================AcWing=====================================
176（https://www.acwing.com/problem/content/178/）经典加油题，使用dijkstra模仿状态
3628（https://www.acwing.com/problem/content/3631/）经典最短路生成树模板题
3772（https://www.acwing.com/problem/content/description/3775/）经典建立反图并使用Dijkstra最短路计数贪心模拟
3797（https://www.acwing.com/problem/content/description/3800/）经典最短路枚举增边排序贪心
4196（https://www.acwing.com/problem/content/4199/）计算最短路长度与返回任意一条最短路

================================LibraryChecker====================================
Shortest Path（https://judge.yosupo.jp/problem/shortest_path）find distance from src to dsc and relative path

"""
from collections import defaultdict, deque, Counter
from heapq import heappush, heappop, heapify
from itertools import accumulate
from math import inf
from operator import add
from typing import List

from src.graph.dijkstra.template import UnDirectedShortestCycle, Dijkstra
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p6175_1(ac=FastIO()):
        # 模板：使用Dijkstra枚举边的方式计算最小环
        n, m = ac.read_list_ints()
        dct = [defaultdict(lambda: inf) for _ in range(n)]
        edges = []
        for _ in range(m):
            i, j, w = ac.read_list_ints()
            dct[i - 1][j - 1] = ac.min(dct[i - 1][j - 1], w)
            dct[j - 1][i - 1] = ac.min(dct[j - 1][i - 1], w)
        for i in range(n):
            for j in dct[i]:
                if j > i:
                    edges.append([i, j, dct[i][j]])
        ans = UnDirectedShortestCycle.find_shortest_cycle_with_edge(n, dct, edges)
        ac.st(ans if ans != -1 else "No solution.")
        return

    @staticmethod
    def lg_p6175_2(ac=FastIO()):
        # 模板：使用Dijkstra枚举点的方式计算最小环
        n, m = ac.read_list_ints()
        dct = [defaultdict(lambda: inf) for _ in range(n)]
        for _ in range(m):
            i, j, w = ac.read_list_ints()
            dct[i - 1][j - 1] = ac.min(dct[i - 1][j - 1], w)
            dct[j - 1][i - 1] = ac.min(dct[j - 1][i - 1], w)
        ans = UnDirectedShortestCycle().find_shortest_cycle_with_node(n, dct)
        ac.st(ans if ans != -1 else "No solution.")
        return

    @staticmethod
    def cf_1343e(ac=FastIO()):
        # 模板：使用01BFS求三个最短路
        for _ in range(ac.read_int()):
            n, m, a, b, c = ac.read_list_ints()
            a -= 1
            b -= 1
            c -= 1
            prices = sorted(ac.read_list_ints())
            prices = list(accumulate(prices, add, initial=0))

            dct = [[] for _ in range(n)]
            for _ in range(m):
                u, v = ac.read_list_ints_minus_one()
                dct[u].append(v)
                dct[v].append(u)

            dis_a = Dijkstra().get_shortest_path_by_bfs(dct, a)
            dis_b = Dijkstra().get_shortest_path_by_bfs(dct, b)
            dis_c = Dijkstra().get_shortest_path_by_bfs(dct, c)
            ans = inf
            for x in range(n):
                up = dis_b[x]
                down = dis_a[x] + dis_c[x]
                if up + down <= m:
                    cur = prices[up] + prices[up + down]
                    ans = ac.min(ans, cur)
            ac.st(ans)
        return

    @staticmethod
    def cf_1650g(ac=FastIO()):
        # 模板：最短路与严格次短路计数，因为不带权，所以正解为01BFS
        mod = 10 ** 9 + 7
        for _ in range(ac.read_int()):
            ac.read_str()
            n, m = ac.read_list_ints()
            s, t = ac.read_list_ints_minus_one()
            dct = [[] for _ in range(n)]
            for _ in range(m):
                x, y = ac.read_list_ints_minus_one()
                dct[x].append([y, 1])
                dct[y].append([x, 1])

            dis, cnt = Dijkstra().get_cnt_of_second_shortest_path(dct, s, mod)
            ans = cnt[t][0]
            if dis[t][1] == dis[t][0] + 1:
                ans += cnt[t][1]
            ac.st(ans % mod)
        return

    @staticmethod
    def lc_787(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        # 模板：Dijkstra 带约束的最短路
        dct = [dict() for _ in range(n)]
        for u, v, p in flights:
            dct[u][v] = p

        # 第一维是代价，第二维是次数
        stack = [[0, 0, src]]
        dis = [inf] * n
        while stack:
            cost, cnt, i = heappop(stack)
            # 前面的代价已经比当前小了若是换乘次数更多则显然不可取
            if dis[i] <= cnt or cnt >= k + 2:  # 超过 k 次换乘也不行
                continue
            if i == dst:
                return cost
            dis[i] = cnt
            for j in dct[i]:
                if cnt + 1 < dis[j]:
                    heappush(stack, [cost + dct[i][j], cnt + 1, j])
        return -1

    @staticmethod
    def lc_2045(n: int, edges: List[List[int]], time: int, change: int) -> any:
        # 模板：严格次短路计算模板题，距离更新时需要注意变化
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i - 1].append(j - 1)
            dct[j - 1].append(i - 1)

        src = 0
        dis = [[inf] * 2 for _ in range(n)]
        dis[src][0] = 0
        stack = [(0, src)]
        while stack:
            d, i = heappop(stack)
            if dis[i][1] < d:
                continue
            for j in dct[i]:
                # 注意此时的更新策略
                if (d // change) % 2 == 0:
                    nex_d = d + time  # 绿灯
                else:
                    nex_d = (d // change + 1) * change + time  # 红灯需要等待
                if dis[j][0] > nex_d:
                    dis[j][1] = dis[j][0]
                    dis[j][0] = nex_d
                    heappush(stack, [nex_d, j])
                elif dis[j][0] < nex_d < dis[j][1]:  # 非严格修改为 d+w < dis[j][1]
                    dis[j][1] = nex_d
                    heappush(stack, [nex_d, j])
        return dis[-1][1]

    @staticmethod
    def lc_2065(values: List[int], edges: List[List[int]], max_time: int) -> int:
        # 模板：经典回溯，正解使用Dijkstra跑最短路剪枝
        n = len(values)
        dct = [[] for _ in range(n)]
        for i, j, t in edges:
            dct[i].append([j, t])
            dct[j].append([i, t])
        dis = Dijkstra().get_shortest_path(dct, 0)

        stack = [[0, 0, {0}]]
        ans = 0
        visit = {tuple(sorted({0}) + [0]): 0}
        while stack:
            t, x, nodes = heappop(stack)
            if dis[x] + t <= max_time:
                cur = sum(values[j] for j in nodes)
                if cur > ans:
                    ans = cur
            for y, w in dct[x]:
                if t + w + dis[y] <= max_time:
                    state = tuple(sorted(nodes.union({y})) + [y])
                    if visit.get(state, inf) > t + w:
                        visit[state] = t + w
                        heappush(stack, [t + w, y, nodes.union({y})])
        return ans

    @staticmethod
    def lc_2093(n: int, highways: List[List[int]], discounts: int) -> int:
        # 模板：Dijkstra 带约束的最短路
        dct = [[] for _ in range(n)]
        for u, v, p in highways:
            dct[u].append([v, p])
            dct[v].append([u, p])

        # 第一维是花费，第二维是折扣次数
        stack = [[0, 0, 0]]
        dis = [inf] * n
        while stack:
            cost, cnt, i = heappop(stack)
            # 前面的代价已经比当前小了若是折扣次数更多则显然不可取
            if dis[i] <= cnt:
                continue
            if i == n - 1:
                return cost
            dis[i] = cnt
            for j, w in dct[i]:
                if cnt < dis[j]:
                    heappush(stack, [cost + w, cnt, j])
                if cnt + 1 < dis[j] and cnt + 1 <= discounts:
                    heappush(stack, [cost + w // 2, cnt + 1, j])

        return -1

    @staticmethod
    def lc_882(edges: List[List[int]], max_moves: int, n: int) -> int:
        # 模板：Dijkstra模板题
        dct = [[] for _ in range(n)]
        for i, j, c in edges:
            dct[i].append([j, c + 1])
            dct[j].append([i, c + 1])

        dis = Dijkstra().get_shortest_path(dct, 0)

        ans = sum(dis[i] <= max_moves for i in range(n))
        for i, j, c in edges:
            if c:
                if dis[i] <= max_moves:
                    a, b = max_moves - dis[i], c
                    left = a if a < b else b
                else:
                    left = 0

                if dis[j] <= max_moves:
                    a, b = max_moves - dis[j], c
                    right = a if a < b else b
                else:
                    right = 0

                if left + right <= c:
                    ans += left + right
                else:
                    ans += c
        return ans

    @staticmethod
    def lc_1293(grid: List[List[int]], k: int) -> int:
        # 模板：Dijkstra 带约束的最短路
        m, n = len(grid), len(grid[0])
        visit = defaultdict(lambda: float("inf"))
        # 第一维是距离，第二维是代价
        stack = [[0, 0, 0, 0]]
        while stack:
            dis, cost, i, j = heappop(stack)
            # 距离更远所以要求消除的障碍物更少
            if visit[(i, j)] <= cost or cost > k:  # 超过 k 次不满足条件
                continue
            if i == m - 1 and j == n - 1:
                return dis
            visit[(i, j)] = cost
            for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                if 0 <= x < m and 0 <= y < n and cost + grid[x][y] < visit[(x, y)]:
                    heappush(stack, [dis + 1, cost + grid[x][y], x, y])
        return -1

    @staticmethod
    def lg_p1462(ac=FastIO()):
        # 模板：Dijkstra 带约束的最短路
        n, m, s = ac.read_list_ints()
        cost = [ac.read_int() for _ in range(n)]
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            a, b, c = ac.read_list_ints()
            a -= 1
            b -= 1
            # 取权值较小的边
            if b not in dct[a] or dct[a][b] > c:
                dct[a][b] = c
            if a not in dct[b] or dct[b][a] > c:
                dct[b][a] = c

        visit = [0] * n
        stack = [[cost[0], 0, s]]
        # 第一维是花费，第二维是血量
        while stack:
            dis, i, bd = heappop(stack)
            # 前期花费更少，就要求当前血量更高
            if visit[i] > bd:
                continue
            if i == n - 1:
                ac.st(dis)
                return
            visit[i] = bd
            for j in dct[i]:
                bj = bd - dct[i][j]
                # 必须是非负数才能存活
                if bj >= visit[j]:
                    visit[j] = bj
                    heappush(stack, [ac.max(dis, cost[j]), j, bj])
        ac.st("AFK")
        return

    @staticmethod
    def lg_p4568(ac=FastIO()):
        # 模板：建立 k+1 层图计算最短路
        n, m, k = ac.read_list_ints()
        s, t = ac.read_list_ints_minus_one()
        dct = [dict() for _ in range(n * (k + 1))]

        def add_edge(x, y, w):
            dct[x][y] = w
            return

        for _ in range(m):
            a, b, c = ac.read_list_ints()
            a -= 1
            b -= 1
            d = dct[a].get(b, inf)
            c = c if c < d else d
            add_edge(a, b, c)
            add_edge(b, a, c)
            for i in range(1, k + 1):
                add_edge(a + i * n, b + i * n, c)
                add_edge(b + i * n, a + i * n, c)

                add_edge(b + (i - 1) * n, a + i * n, 0)
                add_edge(a + (i - 1) * n, b + i * n, 0)

        dis = Dijkstra().get_shortest_path(dct, s)
        ans = inf
        for i in range(k + 1):
            ans = ac.min(ans, dis[t + i * n])
        ac.st(ans)
        return

    @staticmethod
    def lg_p1629(ac=FastIO()):
        # 模板：正反方向建图加两边最短路计算加和即可
        n, m = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        rev = [dict() for _ in range(n)]
        for _ in range(m):
            u, v, w = ac.read_list_ints()
            u -= 1
            v -= 1
            c = dct[u].get(v, inf)
            c = ac.min(c, w)
            dct[u][v] = c
            rev[v][u] = c
        dis1 = Dijkstra().get_shortest_path(dct, 0)
        dis2 = Dijkstra().get_shortest_path(rev, 0)
        ans = sum(dis1[i] + dis2[i] for i in range(n))
        ac.st(ans)
        return

    @staticmethod
    def lg_p2865(ac=FastIO()):
        # 模板：严格次短路计算模板题
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            u, v, w = ac.read_list_ints()
            u -= 1
            v -= 1
            dct[u].append([v, w])
            dct[v].append([u, w])

        dis = Dijkstra().get_second_shortest_path(dct, 0)
        ac.st(dis[n - 1][1])
        return

    @staticmethod
    def lc_lcp75(maze: List[str]) -> int:
        # 模板：最短路逃离
        m, n = len(maze), len(maze[0])
        start = [-1, -1]
        end = [-1, -1]
        for i in range(m):
            for j in range(n):
                w = maze[i][j]
                if w == "S":
                    start = [i, j]
                elif w == "T":
                    end = [i, j]

        # 反向计算到达终点距离
        bfs = [[inf] * n for _ in range(m)]
        bfs[end[0]][end[1]] = 0
        stack = deque([end])
        while stack:
            i, j = stack.popleft()
            for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                if 0 <= x < m and 0 <= y < n and maze[x][y] != "#" and bfs[x][y] == inf:
                    bfs[x][y] = bfs[i][j] + 1
                    stack.append([x, y])

        # 使用魔法卷轴更新正向距离
        dis = [[inf] * n for _ in range(n)]
        for i in range(m):
            for j in range(n):
                if bfs[i][j] < inf:
                    dis[i][j] = 0
                    if maze[i][j] == ".":
                        if maze[m - 1 - i][j] != "#":
                            dis[i][j] = max(dis[i][j], bfs[m - 1 - i][j])
                        if maze[i][n - 1 - j] != "#":
                            dis[i][j] = max(dis[i][j], bfs[i][n - 1 - j])

        # 使用dijkstra计算最短路径边权最小的最大值
        visit = [[inf] * n for _ in range(n)]
        stack = [[dis[start[0]][start[1]], start[0], start[1]]]
        visit[start[0]][start[1]] = dis[start[0]][start[1]]
        while stack:
            d, i, j = heappop(stack)
            if visit[i][j] < d:
                continue
            visit[i][j] = d
            for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                if 0 <= x < m and 0 <= y < n and maze[x][y] != "#":
                    dj = max(d, dis[x][y])
                    if dj < visit[x][y]:
                        visit[x][y] = dj
                        heappush(stack, [dj, x, y])
        x, y = end
        return visit[x][y] if visit[x][y] < inf else -1

    @staticmethod
    def lg_p2622(ac=FastIO()):
        # 模板：Dijkstra加状压最短路
        n = ac.read_int()
        m = ac.read_int()
        grid = [ac.read_list_ints() for _ in range(m)]
        visit = [inf] * (1 << n)
        visit[(1 << n) - 1] = 0
        stack = [[0, (1 << n) - 1]]
        while stack:
            d, state = heappop(stack)
            if visit[state] < d:
                continue
            for i in range(m):
                cur = state
                for j in range(n):
                    if grid[i][j] == 1 and cur & (1 << j):
                        cur ^= (1 << j)
                    elif grid[i][j] == -1 and not cur & (1 << j):
                        cur ^= (1 << j)
                if d + 1 < visit[cur]:
                    visit[cur] = d + 1
                    heappush(stack, [d + 1, cur])
        ans = visit[0]
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def lg_p1073(ac=FastIO()):
        # 模板：正反两遍建图，计算两个最短路
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        rev = [[] for _ in range(n)]
        for _ in range(m):
            x, y, z = ac.read_list_ints_minus_one()
            dct[x].append(y)
            rev[y].append(x)
            if z == 1:
                dct[y].append(x)
                rev[x].append(y)

        # 前面最小值
        floor = [inf] * n
        stack = [[nums[0], 0]]
        floor[0] = nums[0]
        while stack:
            d, i = heappop(stack)
            if floor[i] < d:
                continue
            for j in dct[i]:
                dj = ac.min(d, nums[j])
                if dj < floor[j]:
                    floor[j] = dj
                    heappush(stack, (dj, j))

        # 后面最大值
        ceil = [-inf] * n
        ceil[n - 1] = nums[n - 1]
        stack = [[-nums[n - 1], n - 1]]
        while stack:
            d, i = heappop(stack)
            if ceil[i] < d:
                continue
            for j in rev[i]:
                dj = ac.max(-d, nums[j])
                if dj > ceil[j]:
                    ceil[j] = dj
                    heappush(stack, [-dj, j])
        ans = max(ceil[i] - floor[i] for i in range(n))
        ac.st(ans)
        return

    @staticmethod
    def lg_p1300(ac=FastIO()):
        # 模板：Dijkstra求最短路
        m, n = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]
        ind = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        dct = {"E": 0, "S": 1, "W": 2, "N": 3}
        start = [-1, -1]
        d = -1
        end = [-1, -1]
        for i in range(m):
            for j in range(n):
                w = grid[i][j]
                if w in dct:
                    start = [i, j]
                    d = dct[w]
                if w == "F":
                    end = [i, j]

        dis = [[[inf] * 4 for _ in range(n)] for _ in range(m)]
        dis[start[0]][start[1]][d] = 0
        stack = [[0, start[0], start[1], d]]
        while stack:
            pre, i, j, d = heappop(stack)
            if dis[i][j][d] < pre:
                continue
            flag = False
            for cost, r in [[1, (d - 1) % 4], [5, (d + 1) % 4], [0, d]]:
                x, y = i + ind[r][0], j + ind[r][1]
                if 0 <= x < m and 0 <= y < n and grid[x][y] != ".":
                    dj = pre + cost
                    if dj < dis[x][y][r]:
                        dis[x][y][r] = dj
                        heappush(stack, [dj, x, y, r])
                        flag = True
            if not flag:
                cost, r = 10, (d + 2) % 4
                x, y = i + ind[r][0], j + ind[r][1]
                if 0 <= x < m and 0 <= y < n and grid[x][y] != ".":
                    dj = pre + cost
                    if dj < dis[x][y][r]:
                        dis[x][y][r] = dj
                        heappush(stack, [dj, x, y, r])
        ac.st(min(dis[end[0]][end[1]]))
        return

    @staticmethod
    def lg_p1354(ac=FastIO()):

        # 模板：建图求最短路

        def dis(x1, y1, x2, y2):
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

        n = ac.read_int()
        nodes = [[0, 5], [10, 5]]
        line = []
        for _ in range(n):
            x, a1, a2, b1, b2 = ac.read_list_ints()
            nodes.append([x, a1])
            nodes.append([x, a2])
            nodes.append([x, b1])
            nodes.append([x, b2])
            line.append([x, a1, a2, b1, b2])

        def check():
            for xx, aa1, aa2, bb1, bb2 in line:
                if left <= x <= right:
                    if not (aa1 <= k * xx + bb <= aa2) and not (bb1 <= k * xx + bb <= bb2):
                        return False
            return True

        start = 0
        end = 1
        m = len(nodes)
        dct = [dict() for _ in range(m)]
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i]
                c, d = nodes[j]
                if a == c:
                    continue
                k = (d - b) / (c - a)
                bb = d - k * c
                left, right = min(a, c), max(a, c)
                if check():
                    x = dis(a, b, c, d)
                    dct[i][j] = dct[j][i] = x
        ans = Dijkstra().get_shortest_path(dct, start)[end]
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p1608(ac=FastIO()):
        # 模板：使用Dijkstra计算有向与无向、带权与不带权的最短路数量（最短路计数）
        n, m = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            i, j, w = ac.read_list_ints()
            dct[i - 1][j - 1] = ac.min(dct[i - 1].get(j - 1, inf), w)
        cnt, dis = Dijkstra().get_cnt_of_shortest_path(dct, 0)
        if dis[-1] == inf:
            ac.st("No answer")
        else:
            ac.lst([dis[-1], cnt[-1]])
        return

    @staticmethod
    def lg_p1828(ac=FastIO()):

        # 模板：多个单源Dijkstra最短路计算
        n, p, c = ac.read_list_ints()
        pos = [ac.read_int() - 1 for _ in range(n)]
        dct = [dict() for _ in range(p)]
        for _ in range(c):
            i, j, w = ac.read_list_ints()
            i -= 1
            j -= 1
            dct[i][j] = dct[j][i] = ac.min(dct[i].get(j, inf), w)

        # 也可以从牧出发，但是最好选择较小的集合遍历计算可达最短路
        cnt = Counter(pos)
        total = [0] * p
        for i in cnt:
            dis = Dijkstra().get_shortest_path(dct, i)
            for j in range(p):
                total[j] += dis[j] * cnt[i]
        ac.st(min(total))
        return

    @staticmethod
    def lg_p2047(ac=FastIO()):
        # 模板：Dijkstra计算经过每个点的所有最短路条数占比
        n, m = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            a, b, c = ac.read_list_ints()
            a -= 1
            b -= 1
            dct[a][b] = dct[b][a] = c

        # 计算最短路距离与条数
        dis = []
        cnt = []
        for i in range(n):
            cc, dd = Dijkstra().get_cnt_of_shortest_path(dct, i)
            dis.append(dd)
            cnt.append(cc)

        # 枚举起点与终点计算作为最短路的边的参与次数
        for i in range(n):
            ans = 0
            for j in range(n):
                for k in range(n):
                    if j != i and k != i:
                        if dis[j][i] + dis[i][k] == dis[j][k]:
                            ans += cnt[j][i] * cnt[i][k] / cnt[j][k]
            ac.st("%.3f" % ans)
        return

    @staticmethod
    def lg_p2176(ac=FastIO()):

        # 模板：枚举最短路上的边修改后，重新计算最短路
        n, m = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            dct[i][j] = dct[j][i] = w + 1
        path, dis = Dijkstra().get_shortest_path_from_src_to_dst(dct, 0, n - 1)

        # 枚举边重新计算最短路
        ans = 0
        k = len(path)
        for a in range(k - 1):
            i, j = path[a], path[a + 1]
            dct[i][j] = dct[j][i] = dct[j][i] * 2
            _, cur = Dijkstra().get_shortest_path_from_src_to_dst(dct, 0, n - 1)
            ans = ac.max(ans, cur - dis)
            dct[i][j] = dct[j][i] = dct[j][i] // 2
        ac.st(ans)
        return

    @staticmethod
    def lg_p2269(ac=FastIO()):
        # 模板：比较两个维度的Dijkstra计算
        n, src, dst = ac.read_list_ints()
        src -= 1
        dst -= 1
        time = [ac.read_list_ints() for _ in range(n)]
        loss = [ac.read_list_floats() for _ in range(n)]

        # 丢失率与时延
        dis = [[inf, inf] for _ in range(n)]
        stack = [[0, 0, src]]
        dis[src] = [0, 0]
        # 最短路
        while stack:
            ll, tt, i = heappop(stack)
            if dis[i] < [ll, tt]:
                continue
            if i == dst:
                break
            for j in range(n):
                if time[i][j] != -1 and loss[i][j] != -1:
                    nex_ll = 1 - (1 - ll) * (1 - loss[i][j])
                    nex_tt = tt + time[i][j]
                    if [nex_ll, nex_tt] < dis[j]:
                        dis[j] = [nex_ll, nex_tt]
                        heappush(stack, [nex_ll, nex_tt, j])
        res_ll = dis[dst][0]
        res_tt = dis[dst][1]
        ac.lst([res_tt, "%.4f" % res_ll])
        return

    @staticmethod
    def lg_p2349(ac=FastIO()):

        # 模板：比较两个项相加的最短路
        n, m = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            u, v, w = ac.read_list_ints()
            u -= 1
            v -= 1
            dct[u][v] = ac.min(dct[u].get(v, inf), w)
            dct[v][u] = ac.min(dct[v].get(u, inf), w)

        # 最短路模板
        dis = [inf] * n
        stack = [[0, 0, 0, 0]]
        dis[0] = 0
        while stack:
            dd, d, ceil, i = heappop(stack)
            if dis[i] < dd:
                continue
            if i == n - 1:
                break
            for j in dct[i]:
                dj = d + dct[i][j] + ac.max(ceil, dct[i][j])
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, [dj, d + dct[i][j], ac.max(ceil, dct[i][j]), j])
        ac.st(dis[n - 1])
        return

    @staticmethod
    def lg_p2914(ac=FastIO()):
        # 模板：Dijkstra动态建图计算距离

        def dis(x, y):
            if y in dct[x]:
                return 0
            x1, y1 = nums[x]
            x2, y2 = nums[y]
            res = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            return res if res <= m else inf

        n, w = ac.read_list_ints()
        m = ac.read_float()
        nums = [ac.read_list_ints() for _ in range(n)]
        dct = [set() for _ in range(n)]
        for _ in range(w):
            i, j = ac.read_list_ints_minus_one()
            dct[i].add(j)
            dct[j].add(i)

        n = len(dct)
        visit = [inf] * n
        stack = [[0, 0]]
        visit[0] = 0
        while stack:
            d, i = heappop(stack)
            if visit[i] < d:
                continue
            if i == n - 1:
                break
            for j in range(n):
                dj = dis(i, j) + d
                if dj < visit[j]:
                    visit[j] = dj
                    heappush(stack, (dj, j))
        ac.st(int(visit[-1] * 1000) if visit[-1] < inf else -1)
        return

    @staticmethod
    def lc_6442(n: int, edges: List[List[int]], source: int, destination: int, target: int) -> List[List[int]]:
        dct = [[] for _ in range(n)]
        m = len(edges)
        book = [0] * m
        for ind, (i, j, w) in enumerate(edges):
            if w == -1:
                w = 1
                book[ind] = 1
                edges[ind][-1] = w
            dct[i].append([ind, j])
            dct[j].append([ind, i])

        # 第一遍最短路计算最小情况下的距离
        dis0 = [inf] * n
        stack = [[0, source]]
        dis0[source] = 0
        while stack:
            d, i = heappop(stack)
            if dis0[i] < d:
                continue
            for ind, j in dct[i]:
                dj = edges[ind][2] + d
                if dj < dis0[j]:
                    dis0[j] = dj
                    heappush(stack, (dj, j))
        if dis0[destination] > target:
            return []

        # 第二遍最短路
        dis1 = [inf] * n
        stack = [[0, source]]
        dis1[source] = 0
        while stack:
            d, i = heappop(stack)
            if dis1[i] < d:
                continue
            for ind, j in dct[i]:
                if book[ind]:
                    # 假设 (i, j) 是最短路上的边
                    if (edges[ind][2] + dis1[i]) + (dis0[destination] - dis0[j]) < target:
                        # 此时还有一些增长空间即（当前到达 j 的距离）加上（剩余 j 到 destination）的距离仍旧小于 target
                        x = target - (edges[ind][2] + dis1[i]) - (dis0[destination] - dis0[j])
                        edges[ind][2] += x
                    book[ind] = 0
                dj = edges[ind][2] + d
                if dj < dis1[j]:
                    dis1[j] = dj
                    heappush(stack, (dj, j))

        if dis1[destination] == target:
            return edges
        return []

    @staticmethod
    def cf_715b(ac=FastIO()):
        # 模板：经典两遍最短路，贪心动态更新路径权值
        n, m, target, source, destination = ac.read_list_ints()
        edges = []
        dct = [[] for _ in range(n)]
        book = [0] * m
        for ind in range(m):
            i, j, w = ac.read_list_ints()
            if w == 0:
                w = 1
                book[ind] = 1
            edges.append([i, j, w])
            dct[i].append([ind, j])
            dct[j].append([ind, i])

        # 第一遍最短路计算最小情况下的距离
        dis0 = [inf] * n
        stack = [[0, source]]
        dis0[source] = 0
        while stack:
            d, i = heappop(stack)
            if dis0[i] < d:
                continue
            for ind, j in dct[i]:
                dj = edges[ind][2] + d
                if dj < dis0[j]:
                    dis0[j] = dj
                    heappush(stack, (dj, j))
        if dis0[destination] > target:
            ac.st("NO")
            return

        # 第二遍最短路
        dis1 = [inf] * n
        stack = [[0, source]]
        dis1[source] = 0
        while stack:
            d, i = heappop(stack)
            if dis1[i] < d:
                continue
            for ind, j in dct[i]:
                if book[ind]:
                    # 假设 (i, j) 是最短路上的边
                    if (edges[ind][2] + dis1[i]) + (dis0[destination] - dis0[j]) < target:
                        # 此时还有一些增长空间即（当前到达 j 的距离）加上（剩余 j 到 destination）的距离仍旧小于 target
                        x = target - (edges[ind][2] + dis1[i]) - (dis0[destination] - dis0[j])
                        edges[ind][2] += x
                    book[ind] = 0
                dj = edges[ind][2] + d
                if dj < dis1[j]:
                    dis1[j] = dj
                    heappush(stack, (dj, j))

        if dis1[destination] == target:
            ac.st("YES")
            for e in edges:
                ac.lst(e)
        else:
            ac.st("NO")
        return

    @staticmethod
    def lg_p3753(ac=FastIO()):
        # 模板：最短路变形两个维度的比较
        n, m = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        cnt = 0
        for _ in range(m):
            x, y, s = ac.read_list_ints()
            cnt += s
            x -= 1
            y -= 1
            dct[x][y] = s
            dct[y][x] = s

        dis = [[inf, -inf] for _ in range(n)]
        stack = [[0, 0, 0]]
        dis[0] = [0, 0]

        while stack:
            dd, one, i = heappop(stack)
            if dis[i] < [dd, one]:
                continue
            for j in dct[i]:
                w = dct[i][j]
                dj = [dd + 1, one - w]
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, [dj[0], dj[1], j])
        ans = cnt + dis[-1][1] + dis[-1][0] + dis[-1][1]
        ac.st(ans)
        return

    @staticmethod
    def lg_p3956(ac=FastIO()):
        # 模板：Dijkstra计算最小代价

        m, n = ac.read_list_ints()
        grid = [[-1] * m for _ in range(m)]
        for _ in range(n):
            x, y, c = ac.read_list_ints()
            grid[x - 1][y - 1] = c

        stack = [[0, 0, grid[0][0], 0, 0]]
        final = -1
        visit = defaultdict(lambda: inf)
        while stack:
            cost, magic, color, i, j = heappop(stack)
            if visit[(i, j, color)] <= cost:
                continue
            visit[(i, j, color)] = cost
            if i == j == m - 1:
                final = cost
                break
            for a, b in [[i - 1, j], [i, j + 1], [i, j - 1], [i + 1, j]]:
                if 0 <= a < m and 0 <= b < m:
                    if grid[i][j] != -1:
                        if grid[a][b] != -1:
                            heappush(stack, [cost + int(color != grid[a][b]), 0, grid[a][b], a, b])
                        else:
                            heappush(stack, [cost + 2 + int(color != 0), 1, 0, a, b])
                            heappush(stack, [cost + 2 + int(color != 1), 1, 1, a, b])

                    else:
                        if grid[a][b] != -1:
                            heappush(stack, [cost + int(color != grid[a][b]), 0, grid[a][b], a, b])
        ac.st(final)
        return

    @staticmethod
    def lg_p4880(ac=FastIO()):
        # 模板：枚举终点使用 Dijkstra计算最短路
        lst = []
        while True:
            cur = ac.read_list_ints()
            if not cur:
                break
            lst.extend(cur)
        lst = deque(lst)
        n, m, b, e = lst.popleft(), lst.popleft(), lst.popleft(), lst.popleft()
        b -= 1
        e -= 1
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            x, y, z = lst.popleft(), lst.popleft(), lst.popleft()
            x -= 1
            y -= 1
            dct[x][y] = dct[y][x] = z

        dis = Dijkstra().get_shortest_path(dct, b)

        t = lst.popleft()
        if not t:
            ac.st(dis[e])
            return

        # 枚举被抓住的点
        nums = [[0, e + 1]] + [[lst.popleft(), lst.popleft()] for _ in range(t)]
        nums.sort()
        for i in range(t):
            pos = nums[i][1] - 1
            tt = nums[i + 1][0]
            if dis[pos] < tt:
                ac.st(ac.max(dis[pos], nums[i][0]))
                return
        ac.st(ac.max(dis[nums[-1][1]], nums[-1][0]))
        return

    @staticmethod
    def lg_p4943(ac=FastIO()):
        # 模板：枚举路径跑四遍最短路
        n, m, k = ac.read_list_ints()
        if k:
            visit = set(ac.read_list_ints_minus_one())
        else:
            visit = set()
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            a, b, c = ac.read_list_ints_minus_one()
            dct[a][b] = dct[b][a] = ac.min(dct[a].get(a, inf), c + 1)
        x, y = ac.read_list_ints_minus_one()

        dis1 = Dijkstra().get_dijkstra_result_limit(dct, 0, visit, {0, x, y})
        dis11 = Dijkstra().get_dijkstra_result_limit(dct, x, visit, {0, x, y})
        dis2 = Dijkstra().get_dijkstra_result_limit(dct, 0, set(), {0, x, y})
        dis22 = Dijkstra().get_dijkstra_result_limit(dct, x, set(), {0, x, y})

        ans = min(ac.max(dis1[x], dis2[y]), ac.max(dis1[y], dis2[x]),
                  dis1[x] + dis11[y], dis1[y] + dis11[y],
                  dis2[x] + dis22[y], dis2[y] + dis22[y])
        ac.st(ans)
        return

    @staticmethod
    def lg_p5201(ac=FastIO()):
        # 模板：经典 最短路生成树 建图，再使用树形 DP 计算最优解
        n, m, t = ac.read_list_ints()
        nums = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            a, b, c = ac.read_list_ints_minus_one()
            dct[a].append([b, c + 1])
            dct[b].append([a, c + 1])
        for i in range(n):
            dct[i].sort()
        # 先跑一遍最短路
        dis = [inf] * n
        stack = [[0, 0]]
        dis[0] = 0
        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            for j, w in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, (dj, j))

        # 选择字典序较小的边建立最短路生成树
        edge = [[] for _ in range(n)]
        visit = [0] * n
        for i in range(n):
            for j, w in dct[i]:
                if visit[j]:
                    continue
                if dis[i] + w == dis[j]:
                    edge[i].append(j)
                    edge[j].append(i)
                    visit[j] = 1

        # 树形 DP 计算
        stack = [[0, -1]]
        ans = 0
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in edge[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                for j in edge[i]:
                    if j != fa:
                        nums[i] += nums[j]
                ans = ac.max(ans, nums[i] * (dis[i] - t))
        ac.st(ans)
        return

    @staticmethod
    def lg_p5663(ac=FastIO()):
        # 模板：使用 01 BFS 计算最短的奇数与偶数距离
        n, m, q = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for i in range(m):
            x, y = ac.read_list_ints_minus_one()
            dct[x].append(y)
            dct[y].append(x)

        dis = Dijkstra().get_shortest_path_by_bfs(dct, 0, inf)
        for _ in range(q):
            x, y = ac.read_list_ints()
            x -= 1
            # 只要同奇偶性的最短距离小于等于 y 就有解
            if dis[x][y % 2] > y:
                ac.st("No")
            else:
                # 差距可在两个节点之间反复横跳
                ac.st("Yes")
        return

    @staticmethod
    def lg_p5683(ac=FastIO()):
        # 模板：计算三遍最短路枚举中间节点到三者之间的距离
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        nums = []
        while len(nums) < 2 * m:
            nums.extend(ac.read_list_ints_minus_one())
        for i in range(0, 2 * m, 2):
            x, y = nums[i], nums[i + 1]
            dct[x].append(y)
            dct[y].append(x)
        # 出发与时间约束
        s1, t1, s2, t2 = ac.read_list_ints()
        s1 -= 1
        s2 -= 1
        dis0 = Dijkstra().get_shortest_path_by_bfs(dct, 0, inf)
        dis1 = Dijkstra().get_shortest_path_by_bfs(dct, s1, inf)
        dis2 = Dijkstra().get_shortest_path_by_bfs(dct, s2, inf)
        ans = inf
        for i in range(n):
            cur = dis0[i] + dis1[i] + dis2[i]
            # 注意时间限制
            if dis1[i] + dis0[i] <= t1 and dis2[i] + dis0[i] <= t2:
                ans = ac.min(ans, cur)
        ac.st(-1 if ans == inf else m - ans)
        return

    @staticmethod
    def lg_p5837(ac=FastIO()):
        # 模板：Dijkstra变形问题，带多个状态
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j, c, f = ac.read_list_ints()
            dct[i - 1].append([j - 1, c, f])
            dct[j - 1].append([i - 1, c, f])

        dis = [inf] * n
        stack = [[0, 0, 0, inf]]
        dis[0] = 0
        while stack:
            d, i, cost, flow = heappop(stack)
            if dis[i] < d:
                continue
            for j, c, f in dct[i]:
                dj = -ac.min(flow, f) / (cost + c)
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, [dj, j, cost + c, ac.min(flow, f)])
        ac.st(int(-dis[-1] * 10 ** 6))
        return

    @staticmethod
    def lg_p5930(ac=FastIO()):
        # 模板：经典接雨水使用 Dijkstra 进行计算
        m, n = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        visit = [[inf] * n for _ in range(n)]
        stack = []
        for i in [0, m - 1]:
            for j in range(n):
                stack.append([grid[i][j], i, j])
                visit[i][j] = grid[i][j]
        for j in [0, n - 1]:
            for i in range(1, m - 1):
                stack.append([grid[i][j], i, j])
                visit[i][j] = grid[i][j]
        heapify(stack)
        while stack:
            d, i, j = heappop(stack)
            if visit[i][j] < d:
                continue
            for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                if 0 <= x < m and 0 <= y < n:
                    # 每条路径边权最大值当中的最小值
                    dj = ac.max(grid[x][y], d)
                    if dj < visit[x][y]:
                        visit[x][y] = dj
                        heappush(stack, [dj, x, y])
        ans = 0
        for i in range(m):
            for j in range(n):
                ans += visit[i][j] - grid[i][j]
        ac.st(ans)
        return

    @staticmethod
    def lg_p6063(ac=FastIO()):
        # 模板：经典Dijkstra应用接雨水
        n, m = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]

        # 使用虚拟化超级汇点初始化起点
        stack = []
        for i in [0, m - 1]:
            for j in range(n):
                stack.append([grid[i][j], i, j])
        for i in range(1, m - 1):
            for j in [0, n - 1]:
                stack.append([grid[i][j], i, j])
        heapify(stack)

        # 使用最短路算法寻找每个格子到达超级汇点的路径途中最大值里面的最小值
        ans = 0
        while stack:
            dis, i, j = heappop(stack)
            if grid[i][j] == -1:
                continue
            ans += 0 if dis < grid[i][j] else dis - grid[i][j]
            grid[i][j] = -1
            for x, y in [[i - 1, j], [i + 1, j], [i, j + 1], [i, j - 1]]:
                if 0 <= x < m and 0 <= y < n and grid[x][y] != -1:
                    heappush(stack, [dis if dis > grid[x][y] else grid[x][y], x, y])
        ac.st(ans)
        return

    @staticmethod
    def lc_2714_1(n: int, edges: List[List[int]], s: int, d: int, k: int) -> int:
        # 模板：经典带约束的最短路，也可以使用分层 Dijkstra 求解
        dct = [[] for _ in range(n)]
        for u, v, w in edges:
            dct[u].append([v, w])
            dct[v].append([u, w])

        visit = [[inf] * (k + 1) for _ in range(n)]
        stack = [[0, 0, s]]
        visit[s][0] = 0
        while stack:
            dis, c, i = heappop(stack)
            if i == d:
                return dis
            if visit[i][c] < dis:
                continue
            for j, w in dct[i]:
                if c + 1 <= k and dis < visit[j][c + 1]:
                    visit[j][c + 1] = dis
                    heappush(stack, [dis, c + 1, j])
                if dis + w < visit[j][c]:
                    visit[j][c] = dis + w
                    heappush(stack, [dis + w, c, j])
        return -1

    @staticmethod
    def lc_2714_2(n: int, edges: List[List[int]], s: int, d: int, k: int) -> int:
        # 模板：经典带约束的最短路，也可以使用分层 Dijkstra 求解
        dct = [[] for _ in range(n)]
        for u, v, w in edges:
            dct[u].append([v, w])
            dct[v].append([u, w])

        n = len(dct)
        cnt = [inf] * n
        stack = [[0, 0, s]]
        while stack:
            dis, c, i = heappop(stack)
            if i == d:
                return dis
            if cnt[i] < c:
                continue
            cnt[i] = c
            for j, w in dct[i]:
                if c + 1 < cnt[j] and c + 1 <= k:
                    heappush(stack, [dis, c + 1, j])
                if c < cnt[j]:
                    heappush(stack, [dis + w, c, j])
        return -1

    @staticmethod
    def lc_2577(grid: List[List[int]]) -> int:
        # 模板：Dijkstra经典变形二维矩阵题目

        m, n = len(grid), len(grid[0])
        if grid[0][1] > 1 and grid[1][0] > 1:
            return -1

        dis = [[inf] * n for _ in range(m)]
        dis[0][0] = 0
        stack = [[0, 0, 0]]

        while stack:
            d, i, j = heappop(stack)
            if dis[i][j] < d:
                continue
            for x, y in [[i + 1, j], [i - 1, j], [i, j + 1], [i, j - 1]]:
                if 0 <= x < m and 0 <= y < n:
                    if grid[x][y] <= d + 1:
                        dj = d + 1
                    else:
                        dj = d + 1 + 2 * ((grid[x][y] - d - 1 + 1) // 2)
                    if dj < dis[x][y]:
                        dis[x][y] = dj
                        heappush(stack, [dj, x, y])
        return dis[-1][-1]

    @staticmethod
    def lc_2699(n: int, edges: List[List[int]], source: int, destination: int, target: int) -> List[List[int]]:

        # 模板：经典Dijkstra最短路贪心应用

        dct = [[] for _ in range(n)]
        m = len(edges)
        book = [0] * m
        for ind, (i, j, w) in enumerate(edges):
            if w == -1:
                w = 1
                book[ind] = 1
                edges[ind][-1] = w
            dct[i].append([ind, j])
            dct[j].append([ind, i])

        # 第一遍最短路计算最小情况下的距离
        dis0 = [inf] * n
        stack = [[0, source]]
        dis0[source] = 0
        while stack:
            d, i = heappop(stack)
            if dis0[i] < d:
                continue
            for ind, j in dct[i]:
                dj = edges[ind][2] + d
                if dj < dis0[j]:
                    dis0[j] = dj
                    heappush(stack, (dj, j))
        if dis0[destination] > target:
            return []

        # 第二遍最短路
        dis1 = [inf] * n
        stack = [[0, source]]
        dis1[source] = 0
        while stack:
            d, i = heappop(stack)
            if dis1[i] < d:
                continue
            for ind, j in dct[i]:
                if book[ind]:
                    # 假设 (i, j) 是最短路上的边
                    if (edges[ind][2] + dis1[i]) + (dis0[destination] - dis0[j]) < target:
                        # 此时还有一些增长空间即（当前到达 j 的距离）加上（剩余 j 到 destination）的距离仍旧小于 target
                        x = target - (edges[ind][2] + dis1[i]) - (dis0[destination] - dis0[j])
                        edges[ind][2] += x
                    book[ind] = 0
                dj = edges[ind][2] + d
                if dj < dis1[j]:
                    dis1[j] = dj
                    heappush(stack, (dj, j))

        if dis1[destination] == target:
            return edges
        return []

    @staticmethod
    def lg_p6512(ac=FastIO()):
        # 模板：经典最短路加DP
        n, m, k = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j, w = ac.read_list_ints_minus_one()
            dct[i].append([j, w + 1])
            dct[j].append([i, w + 1])
        dis = []
        for i in range(n):
            dis.append(Dijkstra().get_shortest_path(dct, i))
        dp = [-inf] * (k + 1)
        dp[0] = 0
        pos = [[0, 0]] + [ac.read_list_ints() for _ in range(k)]
        pos.sort()
        for i in range(k):
            t, v = pos[i + 1]
            v -= 1
            lst = [dp[j] + 1 for j in range(i + 1) if dis[v][pos[j][1] - 1] + pos[j][0] <= t] + [0]
            dp[i + 1] = max(lst)
        ac.st(max(dp))
        return

    @staticmethod
    def lg_p8385(ac=FastIO()):
        # 模板：经典脑筋急转弯建图最短路
        n = ac.read_int()
        price = [ac.read_int() for _ in range(n)]
        dct = [[] for _ in range(2 * n)]
        for _ in range(ac.read_int()):
            a, b, c = ac.read_list_ints_minus_one()
            c += 1
            dct[a].append([b, c])
            dct[a + n].append([b + n, c])
        for i in range(n):
            dct[i].append([i + n, price[i] / 2])

        dis = [inf] * 2 * n
        stack = [[0, 0]]
        dis[0] = 0
        while stack:
            total, i = heappop(stack)
            if dis[i] < total:
                continue
            for j, w in dct[i]:
                dj = total + w
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, (dj, j))
        ac.st(int(dis[n]))
        return

    @staticmethod
    def lc_1786(n: int, edges: List[List[int]]) -> int:

        # 模板：经典dijkstra受限最短路计数（类似最短路计数）
        dct = defaultdict(dict)
        for i, j, w in edges:
            dct[i - 1][j - 1] = w
            dct[j - 1][i - 1] = w
        mod = 10 ** 9 + 7
        # 使用倒序进行最短路搜寻
        dis = [float('inf')] * n
        cnt = [0] * n
        cnt[n - 1] = 1
        dis[n - 1] = 0
        # 定义好初始值
        stack = [[0, n - 1]]
        while stack:
            cur_dis, cur = heappop(stack)
            if dis[cur] < cur_dis:
                continue
            dis[cur] = cur_dis
            for nex in dct[cur]:
                # 如果到达下一个点更近，则更新值
                if dis[nex] > dis[cur] + dct[cur][nex]:
                    dis[nex] = dis[cur] + dct[cur][nex]
                    heappush(stack, [dis[nex], nex])
                # 可以形成有效的路径
                if dis[cur] < dis[nex]:
                    cnt[nex] += cnt[cur]
                    cnt[nex] %= mod
        return cnt[0]

    @staticmethod
    def lc_1928_1(max_time: int, edges: List[List[int]], passing_fees: List[int]) -> int:
        # 模板：经典Dijkstra带约束的最短路，也可根据无后效性类似Floyd的动态规划求解
        n = len(passing_fees)
        dct = [[] for _ in range(n)]
        for i, j, w in edges:
            dct[i].append([j, w])
            dct[j].append([i, w])

        # 堆的第一维是代价，第二维是时间，第三维是节点
        stack = [[passing_fees[0], 0, 0]]
        dis = [max_time + 1] * n  # 哈希存的是第二维时间结果，需要持续递减
        while stack:
            cost, tm, i = heappop(stack)
            # 前面的代价已经比当前小了若是换乘次数更多则显然不可取
            if dis[i] <= tm:
                continue
            if i == n - 1:
                return cost
            dis[i] = tm
            for j, w in dct[i]:
                if tm + w < dis[j]:
                    heappush(stack, [cost + passing_fees[j], tm + w, j])
        return -1

    @staticmethod
    def lc_1928_2(max_time: int, edges: List[List[int]], passing_fees: List[int]) -> int:
        # 模板：经典Dijkstra带约束的最短路，也可根据无后效性类似Floyd的动态规划求解
        n = len(passing_fees)
        dp = [[inf] * (max_time + 1) for _ in range(n)]
        dp[0][0] = passing_fees[0]
        for t in range(max_time + 1):
            for i, j, w in edges:
                if w <= t:
                    a, b = dp[j][t], dp[i][t - w] + passing_fees[j]
                    dp[j][t] = a if a < b else b

                    a, b = dp[i][t], dp[j][t - w] + passing_fees[i]
                    dp[i][t] = a if a < b else b
        ans = min(dp[-1])
        return ans if ans < inf else -1

    @staticmethod
    def lc_1976(n: int, roads: List[List[int]]) -> int:
        # 模板：经典Dijkstra最短路计数模板题
        mod = 10 ** 9 + 7
        dct = [dict() for _ in range(n)]
        for i, j, t in roads:
            dct[i][j] = dct[j][i] = t
        return Dijkstra().get_cnt_of_shortest_path(dct, 0)[0][n - 1] % mod

    @staticmethod
    def abc_142f(ac=FastIO()):
        # 模板：经典子图寻找，转换为有向图的最小环问题（可使用BFS优化）
        n, m = ac.read_list_ints()
        dct = [set() for _ in range(n)]
        edges = []
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            dct[x].add((y, 1))
            edges.append([x, y])

        # 枚举边
        ans = inf
        res = []
        for x, y in edges:
            dct[x].discard((y, 1))

            # 使用dijkstra寻找最小环信息
            path, dis = Dijkstra().get_shortest_path_from_src_to_dst([list(e) for e in dct], y, x)
            if dis < ans:
                ans = dis
                res = path[:]
            dct[x].add((y, 1))
        if ans == inf:
            ac.st(-1)
            return
        ac.st(len(res))
        for a in res:
            ac.st(a + 1)
        return

    @staticmethod
    def ac_3628(ac=FastIO()):
        # 模板：经典最短路生成树模板题
        n, m, k = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for ind in range(m):
            x, y, w = ac.read_list_ints()
            x -= 1
            y -= 1
            dct[x].append([y, w, ind])
            dct[y].append([x, w, ind])

        for i in range(n):
            dct[i].sort()

        # 先跑一遍最短路
        dis = [inf] * n
        stack = [[0, 0]]
        dis[0] = 0
        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            for j, w, _ in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, (dj, j))

        # 选择字典序较小的边建立最短路树
        edge = [[] for _ in range(n)]
        visit = [0] * n
        for i in range(n):
            for j, w, ind in dct[i]:
                if visit[j]:
                    continue
                if dis[i] + w == dis[j]:
                    edge[i].append([j, ind])
                    edge[j].append([i, ind])
                    visit[j] = 1

        # 最后一遍BFS确定选择的边
        ans = []
        stack = [[0, -1]]
        while stack:
            x, fa = stack.pop()
            for y, ind in edge[x]:
                if y != fa:
                    ans.append(ind + 1)
                    stack.append([y, x])
        ans = ans[:k]
        ac.st(len(ans))
        ac.lst(ans)
        return

    @staticmethod
    def ac_3772(ac=FastIO()):
        # 模板：经典建立反图并使用Dijkstra最短路计数贪心模拟
        n, m = ac.read_list_ints()
        rev = [[] for _ in range(n)]
        for _ in range(m):
            u, v = ac.read_list_ints_minus_one()
            rev[v].append([u, 1])

        k = ac.read_int()
        p = ac.read_list_ints_minus_one()
        cnt, dis = Dijkstra().get_cnt_of_shortest_path(rev, p[-1])

        floor = 0
        for i in range(k - 1):
            if k - i - 1 == dis[p[i]]:
                break
            if dis[p[i - 1]] == dis[p[i]] + 1:
                continue
            else:
                floor += 1

        ceil = 0
        for i in range(k - 1):
            if dis[p[i]] == dis[p[i + 1]] + 1 and cnt[p[i]] == cnt[p[i + 1]]:
                continue
            else:
                ceil += 1
        ac.lst([floor, ceil])

        return

    @staticmethod
    def ac_3797(ac=FastIO()):
        # 模板：经典最短路枚举增边排序贪心
        n, m, k = ac.read_list_ints()
        nums = ac.read_list_ints_minus_one()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            dct[x].append(y)
            dct[y].append(x)
        dis0 = Dijkstra().get_shortest_path_by_bfs(dct, 0)
        dis1 = Dijkstra().get_shortest_path_by_bfs(dct, n - 1)
        nums.sort(key=lambda it: dis0[it] - dis1[it])

        ans = -1
        pre = -inf
        d = dis0[-1]
        for i in range(k):
            cur = pre + 1 + dis1[nums[i]]
            if cur > ans:
                ans = cur
            if dis0[nums[i]] > pre:
                pre = dis0[nums[i]]

        if ans > d:
            ans = d
        ac.st(ans)
        return

    @staticmethod
    def ac_4196(ac=FastIO()):
        # 模板：计算最短路长度与返回任意一条最短路
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j, w = ac.read_list_ints()
            i -= 1
            j -= 1
            dct[i].append([j, w])
            dct[j].append([i, w])
        path, ans = Dijkstra().get_shortest_path_from_src_to_dst(dct, 0, n - 1)
        if ans == inf:
            ac.st(-1)
        else:
            path.reverse()
            ac.lst([x + 1 for x in path])
        return