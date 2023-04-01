
import heapq
import unittest
from collections import defaultdict, deque
from itertools import accumulate
from operator import add
from typing import List

from algorithm.src.fast_io import FastIO

"""
算法：Dijkstra（单源最短路经算法）
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

================================CodeForces================================
C. Dijkstra?（https://codeforces.com/problemset/problem/20/C）正权值最短路计算，并记录返回生成路径
E. Weights Distributing（https://codeforces.com/problemset/problem/1343/E）使用三个01BFS求最短路加贪心枚举计算

参考：OI WiKi（xx）
"""


class Dijkstra:
    def __init__(self):
        return

    @staticmethod
    def get_dijkstra_result(dct, src):
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
    def dijkstra_src_to_dst_path(dct, src, dst):
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

        # 求乘积最大的路
        inf = float("inf")
        dis = defaultdict(lambda: -inf)
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


class Solution:
    def __init__(self):
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


class TestGeneral(unittest.TestCase):

    def test_dijkstra(self):
        djk = Dijkstra()
        dct = [{1: 1, 2: 4}, {2: 2}, {}]
        assert djk.get_dijkstra_result(dct, 0) == [0, 1, 3]
        return


if __name__ == '__main__':
    unittest.main()
