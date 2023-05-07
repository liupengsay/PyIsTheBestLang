
import heapq
import unittest
from collections import defaultdict, deque
from itertools import accumulate
from operator import add
from typing import List, Dict

from algorithm.src.fast_io import FastIO, inf

"""
算法：Dijkstra（单源最短路经算法）、严格次短路、要保证加和最小因此只支持非负数权值、或者取反全部为非正数计算最长路
功能：计算点到有向或者无向图里面其他点的最近距离
题目：

===================================力扣===================================
42. 接雨水（https://leetcode.cn/problems/trapping-rain-water/）一维接雨水，计算前后缀最大值的最小值再减去自身值
407. 接雨水 II（https://leetcode.cn/problems/trapping-rain-water-ii/）经典最短路变种问题，求解路径上边权的最大值
787. K 站中转内最便宜的航班（https://leetcode.cn/problems/cheapest-flights-within-k-stops/）使用带约束的最短路计算最终结果
1293. 网格中的最短路径（https://leetcode.cn/problems/shortest-path-in-a-grid-with-obstacles-elimination/）使用带约束的最短路计算最终结果
2203. 得到要求路径的最小带权子图（https://leetcode.cn/problems/minimum-weighted-subgraph-with-the-required-paths/）使用三个Dijkstra最短路获得结果
2258. 逃离火灾（https://leetcode.cn/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/）使用双源BFS计算等待时间后最短路求出路径上最小等待时间的最大值
2290. 到达角落需要移除障碍物的最小数（https://leetcode.cn/problems/minimum-obstacle-removal-to-reach-corner/）计算最小代价
499. 迷宫 III（https://leetcode.cn/problems/the-maze-iii/?envType=study-plan-v2&id=premium-algo-100）两个参数变量的最短路
LCP 75. 传送卷轴（https://leetcode.cn/problems/rdmXM7/）一层BFS之后计算最大值最小的最短路

===================================洛谷===================================
P3371 单源最短路径（弱化版）（https://www.luogu.com.cn/problem/P3371）最短路模板题
P4779 【模板】单源最短路径（标准版）（https://www.luogu.com.cn/problem/P4779）最短路模板题
P1629 邮递员送信（https://www.luogu.com.cn/problem/P1629）正反两个方向的最短路进行计算往返路程
P1462 通往奥格瑞玛的道路（https://www.luogu.com.cn/problem/P1462）使用带约束的最短路计算最终结果

P1339 [USACO09OCT]Heat Wave G（https://www.luogu.com.cn/problem/P1339）标准最短路计算
P1342 请柬（https://www.luogu.com.cn/problem/P1342）正反两遍最短路
P1576 最小花费（https://www.luogu.com.cn/problem/P1576）堆优化转换成负数求最短路

P1821 [USACO07FEB] Cow Party S（https://www.luogu.com.cn/problem/P1821）正反两遍最短路
P1882 接力赛跑（https://www.luogu.com.cn/problem/P1882）转换为最短路求解最短路距离最远的点
P1907 设计道路（https://www.luogu.com.cn/problem/P1907）自定义建图计算最短路
P1744 采购特价商品（https://www.luogu.com.cn/problem/P1744）裸题最短路
P1529 [USACO2.4]回家 Bessie Come Home（https://www.luogu.com.cn/problem/P1529）裸题最短路
P1649 [USACO07OCT]Obstacle Course S（https://www.luogu.com.cn/problem/P1649）自定义距离计算的最短路
P2083 找人（https://www.luogu.com.cn/problem/P2083）反向最短路
P2299 Mzc和体委的争夺战（https://www.luogu.com.cn/problem/P2299）最短路裸题
P2683 小岛（https://www.luogu.com.cn/problem/P2683）最短路裸题结合并查集查询

P1396 营救（https://www.luogu.com.cn/problem/P1396）最短路变种问题，求解路径上边权的最大值，类似接雨水
P1346 电车（https://www.luogu.com.cn/problem/P1346）建图跑最短路
P1339 [USACO09OCT]Heat Wave G（https://www.luogu.com.cn/record/list?user=739032&status=12&page=11）最短路裸题
P2784 化学1（chem1）- 化学合成（https://www.luogu.com.cn/problem/P2784）最大乘积的路径

P1318 积水面积（https://www.luogu.com.cn/problem/P1318）一维接雨水，计算前后缀最大值的最小值再减去自身值
P2888 [USACO07NOV]Cow Hurdles S（https://www.luogu.com.cn/problem/P2888）最短路计算路径上最大边权值最小的路径
P2935 [USACO09JAN]Best Spot S（https://www.luogu.com.cn/problem/P2935）最短路裸题
P2951 [USACO09OPEN]Hide and Seek S（https://www.luogu.com.cn/problem/P2951）最短路裸题
P2984 [USACO10FEB]Chocolate Giving S（https://www.luogu.com.cn/problem/P2984）最短路裸题
P3003 [USACO10DEC]Apple Delivery S（https://www.luogu.com.cn/problem/P3003）三遍最短路
P3094 [USACO13DEC]Vacation Planning S（https://www.luogu.com.cn/problem/P3094）预处理最短路之后进行查询
P3905 道路重建（https://www.luogu.com.cn/problem/P3905）逆向思维，重新建图后计算最短路
P5764 [CQOI2005]新年好（https://www.luogu.com.cn/problem/P5764）五遍最短路裸题计算
P5767 [NOI1997] 最优乘车（https://www.luogu.com.cn/problem/P5767）经典建图求解公交与地铁的最短换乘
P6770 [USACO05MAR]Checking an Alibi 不在场的证明（https://www.luogu.com.cn/problem/P6770）最短路裸题
P6833 [Cnoi2020]雷雨（https://www.luogu.com.cn/problem/P6833）三遍最短路后，进行枚举计算
P7551 [COCI2020-2021#6] Alias（https://www.luogu.com.cn/problem/P7551）最短路裸题，注意重边与自环

P6175 无向图的最小环问题（https://www.luogu.com.cn/problem/P6175）使用Dijkstra枚举边计算或者使用DFS枚举点，带权
P4568 [JLOI2011] 飞行路线（https://www.luogu.com.cn/problem/P4568）K层建图计算Dijkstra最短路
P2865 [USACO06NOV]Roadblocks G（https://www.luogu.com.cn/problem/P2865）严格次短路模板题
P2622 关灯问题II（https://www.luogu.com.cn/problem/P2622）状压加dijkstra最短路计算
P1608 路径统计（https://www.luogu.com.cn/problem/P1608）dijkstra计算最短路径条数

================================CodeForces================================
C. Dijkstra?（https://codeforces.com/problemset/problem/20/C）正权值最短路计算，并记录返回生成路径
E. Weights Distributing（https://codeforces.com/problemset/problem/1343/E）使用三个01BFS求最短路加贪心枚举计算

================================AcWing====================================
176. 装满的油箱（https://www.acwing.com/problem/content/178/）经典加油题，使用dijkstra模仿状态

参考：OI WiKi（xx）
"""


class Dijkstra:
    def __init__(self):
        return

    @staticmethod
    def get_dijkstra_result(dct: List[Dict], src: int) -> List[float]:
        # 模板: Dijkstra求最短路，变成负数求可以求最长路（还是正权值）
        n = len(dct)
        dis = [float("inf")]*n
        stack = [[0, src]]
        dis[src] = 0

        while stack:
            d, i = heapq.heappop(stack)
            if dis[i] < d:
                continue
            for j in dct[i]:
                dj = dct[i][j] + d
                if dj < dis[j]:
                    dis[j] = dj
                    heapq.heappush(stack, [dj, j])
        return dis

    @staticmethod
    def get_dijkstra_cnt(dct: List[Dict], src: int) -> (List[int], List[float]):
        # 模板: Dijkstra求最短路条数
        n = len(dct)
        dis = [float("inf")]*n
        stack = [[0, src]]
        dis[src] = 0
        cnt = [0]*n
        cnt[src] = 1
        while stack:
            d, i = heapq.heappop(stack)
            if dis[i] < d:
                continue
            for j in dct[i]:
                dj = dct[i][j] + d
                if dj < dis[j]:
                    dis[j] = dj
                    cnt[j] = cnt[i]
                    heapq.heappush(stack, [dj, j])
                elif dj == dis[j]:
                    cnt[j] += cnt[i]
        return cnt, dis

    @staticmethod
    def dijkstra_src_to_dst_path(dct: List[Dict], src: int, dst: int) -> float:
        # 模板: Dijkstra求起终点的最短路，注意只能是正权值可以提前返回结果，并返回对应经过的路径
        n = len(dct)
        dis = [float("inf")] * n
        stack = [[0, src]]
        dis[src] = 0
        father = [-1] * n  # 记录最短路的上一跳
        while stack:
            d, i = heapq.heappop(stack)
            if dis[i] < d:
                continue
            if i == dst:
                return d
            for j in dct[i]:
                dj = dct[i][j] + d
                if dj < dis[j]:
                    dis[j] = dj
                    heapq.heappush(stack, [dj, j])
        # 向上回溯路径
        path = []
        i = dst
        while i != -1:
            path.append(i + 1)
            i = father[i]
        path.reverse()
        return dis[dst]

    @staticmethod
    def gen_dijkstra_max_result(dct, src, dsc):

        # 求乘积最大的路，取反后求最短路径
        dis = defaultdict(lambda: float("-inf"))
        stack = [[-1, src]]
        dis[src] = 1
        while stack:
            d, i = heapq.heappop(stack)
            d = -d
            if dis[i] > d:
                continue
            for j in dct[i]:
                dj = dct[i][j] * d
                if dj > dis[j]:
                    dis[j] = dj
                    heapq.heappush(stack, [-dj, j])
        return dis[dsc]

    @staticmethod
    def get_shortest_by_bfs(dct: List[List[int]], src):
        # 模板: 使用01BFS求最短路
        n = len(dct)
        dis = [-1] * n
        stack = deque([src])
        dis[src] = 0
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                if dis[j] == -1:
                    dis[j] = dis[i] + 1
                    stack.append(j)
        return dis

    @staticmethod
    def get_second_shortest_path(dct: List[List[int]], src):
        # 模板：使用Dijkstra计算严格次短路
        n = len(dct)
        inf = float("inf")
        dis = [[inf]*2 for _ in range(n)]
        dis[src][0] = 0
        stack = [[0, 0]]
        while stack:
            d, i = heapq.heappop(stack)
            if d > dis[i][1]:
                continue
            for j, w in dct[i]:
                if dis[j][0] > d+w:
                    dis[j][1] = dis[j][0]
                    dis[j][0] = d+w
                    heapq.heappush(stack, [d + w, j])
                elif dis[j][0] < d+w < dis[j][1]:
                    dis[j][1] = d+w
                    heapq.heappush(stack, [d+w, j])
        return dis

class UnDirectedShortestCycle:
    def __init__(self):
        return

    @staticmethod
    def find_shortest_cycle_with_node(n: int, dct) -> int:
        # 模板：求无向图的最小环长度，枚举点
        ans = inf
        for i in range(n):
            dist = [inf] * n
            par = [-1] * n
            dist[i] = 0
            q = [[0, i]]
            while q:
                _, x = heapq.heappop(q)
                for child in dct[x]:
                    if dist[x] > ans:
                        break
                    if dist[child] > dct[x][child] + dist[x]:
                        dist[child] = dct[x][child] + dist[x]
                        par[child] = x
                        heapq.heappush(q, [dist[child], child])
                    elif par[x] != child and par[child] != x:
                        cur = dist[x] + dist[child] + dct[x][child]
                        ans = ans if ans < cur else cur
        return ans if ans != inf else -1

    @staticmethod
    def find_shortest_cycle_with_edge(n: int, dct, edges) -> int:
        # 模板：求无向图的最小环长度，枚举边

        ans = inf
        for x, y, w in edges:
            dct[x].pop(y)
            dct[y].pop(x)

            dis = [inf] * n
            stack = [[0, x]]
            dis[x] = 0

            while stack:
                d, i = heapq.heappop(stack)
                if dis[i] < d:
                    continue
                if i == y:
                    break
                for j in dct[i]:
                    dj = dct[i][j] + d
                    if dj < dis[j]:
                        dis[j] = dj
                        heapq.heappush(stack, [dj, j])

            ans = ans if ans < dis[y] + w else dis[y] + w
            dct[x][y] = w
            dct[y][x] = w
        return ans if ans < inf else -1


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p6175_1(ac=FastIO()):
        # 模板：使用Dijkstra枚举边的方式计算最小环
        n, m = ac.read_ints()
        dct = [defaultdict(lambda: inf) for _ in range(n)]
        edges = []
        for _ in range(m):
            i, j, w = ac.read_ints()
            dct[i-1][j-1] = ac.min(dct[i-1][j-1], w)
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
        n, m = ac.read_ints()
        dct = [defaultdict(lambda: inf) for _ in range(n)]
        for _ in range(m):
            i, j, w = ac.read_ints()
            dct[i-1][j-1] = ac.min(dct[i-1][j-1], w)
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
                u, v = ac.read_ints_minus_one()
                dct[u].append(v)
                dct[v].append(u)

            dis_a = Dijkstra().get_shortest_by_bfs(dct, a)
            dis_b = Dijkstra().get_shortest_by_bfs(dct, b)
            dis_c = Dijkstra().get_shortest_by_bfs(dct, c)
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
    def lc_787(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        # 模板：Dijkstra 带约束的最短路
        dct = [dict() for _ in range(n)]
        for u, v, p in flights:
            dct[u][v] = p

        # 第一维是代价，第二维是次数
        stack = [[0, 0, src]]
        dis = [float("inf")] * n
        while stack:
            cost, cnt, i = heapq.heappop(stack)
            # 前面的代价已经比当前小了若是换乘次数更多则显然不可取
            if dis[i] <= cnt or cnt >= k + 2:  # 超过 k 次换乘也不行
                continue
            if i == dst:
                return cost
            dis[i] = cnt
            for j in dct[i]:
                if cnt + 1 < dis[j]:
                    heapq.heappush(stack, [cost + dct[i][j], cnt + 1, j])
        return -1

    @staticmethod
    def lc_1293(grid: List[List[int]], k: int) -> int:
        # 模板：Dijkstra 带约束的最短路
        m, n = len(grid), len(grid[0])
        visit = defaultdict(lambda: float("inf"))
        # 第一维是距离，第二维是代价
        stack = [[0, 0, 0, 0]]
        while stack:
            dis, cost, i, j = heapq.heappop(stack)
            # 距离更远所以要求消除的障碍物更少
            if visit[(i, j)] <= cost or cost > k:  # 超过 k 次不满足条件
                continue
            if i == m - 1 and j == n - 1:
                return dis
            visit[(i, j)] = cost
            for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                if 0 <= x < m and 0 <= y < n and cost + grid[x][y] < visit[(x, y)]:
                    heapq.heappush(stack, [dis + 1, cost + grid[x][y], x, y])
        return -1

    @staticmethod
    def lg_p1462(ac=FastIO()):
        # 模板：Dijkstra 带约束的最短路
        n, m, s = ac.read_ints()
        cost = [ac.read_int() for _ in range(n)]
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            a, b, c = ac.read_ints()
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
            dis, i, bd = heapq.heappop(stack)
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
                    heapq.heappush(stack, [ac.max(dis, cost[j]), j, bj])
        ac.st("AFK")
        return

    @staticmethod
    def lg_p4568(ac=FastIO()):
        # 模板：建立 k+1 层图计算最短路
        n, m, k = ac.read_ints()
        s, t = ac.read_ints_minus_one()
        dct = [dict() for _ in range(n * (k + 1))]

        def add_edge(x, y, w):
            dct[x][y] = w
            return

        for _ in range(m):
            a, b, c = ac.read_ints()
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

        dis = Dijkstra().get_dijkstra_result(dct, s)
        ans = inf
        for i in range(k + 1):
            ans = ac.min(ans, dis[t + i * n])
        ac.st(ans)
        return

    @staticmethod
    def lg_p1629(ac=FastIO()):
        # 模板：正反方向建图加两边最短路计算加和即可
        n, m = ac.read_ints()
        dct = [dict() for _ in range(n)]
        rev = [dict() for _ in range(n)]
        for _ in range(m):
            u, v, w = ac.read_ints()
            u -= 1
            v -= 1
            c = dct[u].get(v, inf)
            c = ac.min(c, w)
            dct[u][v] = c
            rev[v][u] = c
        dis1 = Dijkstra().get_dijkstra_result(dct, 0)
        dis2 = Dijkstra().get_dijkstra_result(rev, 0)
        ans = sum(dis1[i]+dis2[i] for i in range(n))
        ac.st(ans)
        return

    @staticmethod
    def lg_p2865(ac=FastIO()):
        # 模板：严格次短路计算模板题
        n, m = ac.read_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            u, v, w = ac.read_ints()
            u -= 1
            v -= 1
            dct[u].append([v, w])
            dct[v].append([u, w])

        dis = Dijkstra().get_second_shortest_path(dct, 0)
        ac.st(dis[n-1][1])
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
            d, i, j = heapq.heappop(stack)
            if visit[i][j] < d:
                continue
            visit[i][j] = d
            for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                if 0 <= x < m and 0 <= y < n and maze[x][y] != "#":
                    dj = max(d, dis[x][y])
                    if dj < visit[x][y]:
                        visit[x][y] = dj
                        heapq.heappush(stack, [dj, x, y])
        x, y = end
        return visit[x][y] if visit[x][y] < inf else -1

    @staticmethod
    def lg_p2622(ac=FastIO()):
        # 模板：Dijkstra加状压最短路
        n = ac.read_int()
        m = ac.read_int()
        grid = [ac.read_list_ints() for _ in range(m)]
        visit = [inf]*(1<<n)
        visit[(1<<n)-1] = 0
        stack = [[0,  (1<<n)-1]]
        while stack:
            d, state = heapq.heappop(stack)
            if visit[state] < d:
                continue
            for i in range(m):
                cur = state
                for j in range(n):
                    if grid[i][j] == 1 and cur & (1<<j):
                        cur ^= (1<<j)
                    elif grid[i][j] == -1 and not cur & (1<<j):
                        cur ^= (1 << j)
                if d+1 < visit[cur]:
                    visit[cur] = d+1
                    heapq.heappush(stack, [d+1, cur])
        ans = visit[0]
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def lg_p1608(ac=FastIO()):
        # 模板：有向有权图最短路计数
        n, m = ac.read_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            i, j, w = ac.read_ints()
            dct[i-1][j-1] = ac.min(dct[i-1].get(j-1, inf), w)
        cnt, dis = Dijkstra().get_dijkstra_cnt(dct, 0)
        if dis[-1] == inf:
            ac.st("No answer")
        else:
            ac.lst([dis[-1], cnt[-1]])
        return


class TestGeneral(unittest.TestCase):

    def test_dijkstra(self):
        djk = Dijkstra()
        dct = [{1: 1, 2: 4}, {2: 2}, {}]
        assert djk.get_dijkstra_result(dct, 0) == [0, 1, 3]
        return


if __name__ == '__main__':
    unittest.main()
