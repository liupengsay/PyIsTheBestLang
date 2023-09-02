
import heapq
import unittest
from collections import defaultdict, deque
from itertools import accumulate
from operator import add
from typing import List, Dict, Set
from collections import Counter

from src.fast_io import FastIO, inf
from src.graph.spfa import SPFA

"""
算法：Dijkstra（单源最短路经算法）、严格次短路、要保证加和最小因此只支持非负数权值、或者取反全部为非正数计算最长路、最短路生成树
功能：计算点到有向或者无向图里面其他点的最近距离、带约束的最短路、分层Dijkstra
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
6442. 修改图中的边权（https://leetcode.cn/problems/modify-graph-edge-weights/）经典两遍最短路，贪心动态更新路径权值
2714. 找到最短路径的 K 次跨越（https://leetcode.cn/problems/find-shortest-path-with-k-hops/）经典带约束的最短路，也可以使用分层Dijkstra求解
2699. 修改图中的边权（https://leetcode.cn/problems/modify-graph-edge-weights/）经典Dijkstra最短路贪心应用
1786. 从第一个节点出发到最后一个节点的受限路径数（https://leetcode.cn/problems/number-of-restricted-paths-from-first-to-last-node/）经典dijkstra受限最短路计数（类似最短路计数）
1928. 规定时间内到达终点的最小花费（https://leetcode.cn/problems/minimum-cost-to-reach-destination-in-time/）经典Dijkstra带约束的最短路，也可根据无后效性类似Floyd的动态规划求解
LCP 75. 传送卷轴（https://leetcode.cn/problems/rdmXM7/）首先BFS之后计算最大值最小的最短路
1976. 到达目的地的方案数（https://leetcode.cn/problems/number-of-ways-to-arrive-at-destination/）经典Dijkstra最短路计数模板题
2045. 到达目的地的第二短时间（https://leetcode.cn/problems/second-minimum-time-to-reach-destination/）不带权的严格次短路耗时模拟计算
2093. 前往目标城市的最小费用（https://leetcode.cn/problems/minimum-cost-to-reach-city-with-discounts/）经典Dijkstra带约束的最短路
882. 细分图中的可到达节点（https://leetcode.cn/problems/reachable-nodes-in-subdivided-graph/description/）Dijkstra模板题
2577. 在网格图中访问一个格子的最少时间（https://leetcode.cn/problems/minimum-time-to-visit-a-cell-in-a-grid/）Dijkstra经典变形二维矩阵题目


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
P1073 [NOIP2009 提高组] 最优贸易（https://www.luogu.com.cn/problem/P1073）正反两遍建图，Dijkstra进行计算路径最大最小值
P1300 城市街道交通费系统（https://www.luogu.com.cn/problem/P1300）Dijkstra求最短路
P1354 房间最短路问题（https://www.luogu.com.cn/problem/P1354）建图Dijkstra求最短路
P1608 路径统计（https://www.luogu.com.cn/problem/P1608）使用Dijkstra计算有向与无向、带权与不带权的最短路数量
P1828 [USACO3.2]香甜的黄油 Sweet Butter（https://www.luogu.com.cn/problem/P1828）多个单源Dijkstra最短路计算
P2047 [NOI2007] 社交网络（https://www.luogu.com.cn/problem/P2047）Dijkstra计算经过每个点的所有最短路条数占比，也可以使用Floyd进行计算
P2269 [HNOI2002]高质量的数据传输（https://www.luogu.com.cn/problem/P2269）比较两个项的最短路计算
P2349 金字塔（https://www.luogu.com.cn/problem/P2349）比较两个项相加的最短路
P2914 [USACO08OCT]Power Failure G（https://www.luogu.com.cn/problem/P2914）Dijkstra动态建图计算距离
P3020 [USACO11MAR]Package Delivery S（https://www.luogu.com.cn/problem/P3020）Dijkstra求最短路
P3057 [USACO12NOV]Distant Pastures S（https://www.luogu.com.cn/problem/P3057）Dijkstra求最短路
P3753 国事访问（https://www.luogu.com.cn/problem/P3753）最短路变形两个维度的比较
P3956 [NOIP2017 普及组] 棋盘（https://www.luogu.com.cn/problem/P3956）多维状态的Dijkstra
P4880 抓住czx（https://www.luogu.com.cn/problem/P4880）枚举终点使用 Dijkstra计算最短路
P4943 密室（https://www.luogu.com.cn/problem/P4943）枚举路径跑四遍最短路
P5201 [USACO19JAN]Shortcut G（https://www.luogu.com.cn/problem/P5201）经典最短路生成树建图，再使用树形 DP 计算最优解
P5663 [CSP-J2019] 加工零件（https://www.luogu.com.cn/problem/P5663）经典最短路变形题目，计算最短的奇数与偶数距离
P5683 [CSP-J2019 江西] 道路拆除（https://www.luogu.com.cn/problem/P5683）计算三遍最短路枚举中间节点到三者之间的距离
P5837 [USACO19DEC]Milk Pumping G（https://www.luogu.com.cn/problem/P5837）经典Dijkstra变形问题，带多个状态
P5905 【模板】Johnson 全源最短路（https://www.luogu.com.cn/problem/P5905）有向带权图可能有负权 Johnson 全源最短路计算所有点对的最短路
P5930 [POI1999] 降水（https://www.luogu.com.cn/problem/P5930）经典Dijkstra应用接雨水
P6063 [USACO05JAN]The Wedding Juicer G（https://www.luogu.com.cn/problem/P6063）经典Dijkstra应用接雨水
P6512 [QkOI#R1] Quark and Flying Pigs（https://www.luogu.com.cn/problem/P6512）经典最短路加DP
P8385 [POI 2003] Smugglers（https://www.luogu.com.cn/problem/P8385）经典脑筋急转弯建图最短路
P8724 [蓝桥杯 2020 省 AB3] 限高杆（https://www.luogu.com.cn/problem/P8724）分层最短路Dijkstra计算
P8802 [蓝桥杯 2022 国 B] 出差（https://www.luogu.com.cn/problem/P8802）Dijkstra基础权重变形题
P2176 [USACO11DEC] RoadBlock S / [USACO14FEB]Roadblock G/S（https://www.luogu.com.cn/problem/P2176）枚举最短路上的边修改后，重新计算最短路
================================CodeForces================================
C. Dijkstra?（https://codeforces.com/problemset/problem/20/C）正权值最短路计算，并记录返回生成路径
E. Weights Distributing（https://codeforces.com/problemset/problem/1343/E）使用三个01BFS求最短路加贪心枚举计算
B. Complete The Graph（https://codeforces.com/contest/715/problem/B）经典两遍最短路，贪心动态更新路径权值

================================AcWing====================================
176. 装满的油箱（https://www.acwing.com/problem/content/178/）经典加油题，使用dijkstra模仿状态
3628. 边的删减（https://www.acwing.com/problem/content/3631/）经典最短路生成树模板题
3772. 更新线路（https://www.acwing.com/problem/content/description/3775/）经典建立反图并使用Dijkstra最短路计数贪心模拟
3797. 最大化最短路（https://www.acwing.com/problem/content/description/3800/）经典最短路枚举增边排序贪心
4196. 最短路径（https://www.acwing.com/problem/content/4199/）计算最短路长度与返回任意一条最短路

参考：OI WiKi（xx）
"""


class Dijkstra:
    def __init__(self):
        return

    @staticmethod
    def get_dijkstra_result(dct: List[List[int]], src: int) -> List[float]:
        # 模板: Dijkstra求最短路，变成负数求可以求最长路（还是正权值）
        n = len(dct)
        dis = [inf]*n
        stack = [[0, src]]
        dis[src] = 0

        while stack:
            d, i = heapq.heappop(stack)
            if dis[i] < d:
                continue
            for j, w in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    heapq.heappush(stack, [dj, j])
        return dis

    @staticmethod
    def get_dijkstra_cnt(dct: List[List[int]], src: int) -> (List[int], List[any]):
        # 模板: Dijkstra求最短路条数（最短路计算）
        n = len(dct)
        dis = [inf]*n
        stack = [[0, src]]
        dis[src] = 0
        cnt = [0]*n
        cnt[src] = 1
        while stack:
            d, i = heapq.heappop(stack)
            if dis[i] < d:
                continue
            for j, w in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    # 最短距离更新，重置计数
                    cnt[j] = cnt[i]
                    heapq.heappush(stack, [dj, j])
                elif dj == dis[j]:
                    # 最短距离一致，增加计数
                    cnt[j] += cnt[i]
        return cnt, dis

    @staticmethod
    def get_dijkstra_result_limit(dct: List[List[int]], src: int, limit: Set[int], target: Set[int]) -> List[float]:
        # 模板: Dijkstra求最短路，变成负数求可以求最长路（还是正权值）
        n = len(dct)
        dis = [float("inf")] * n

        dis[src] = 0 if src not in limit else inf
        stack = [[dis[src], src]]
        # 限制只能跑 limit 的点到 target 中的点
        while stack and target:
            d, i = heapq.heappop(stack)
            if i in target:
                target.discard(i)
            if dis[i] < d:
                continue
            for j, w in dct[i]:
                if j not in limit:
                    dj = w + d
                    if dj < dis[j]:
                        dis[j] = dj
                        heapq.heappush(stack, [dj, j])
        return dis

    @staticmethod
    def dijkstra_src_to_dst_path(dct: List[List[int]], src: int, dst: int) -> (List[int], any):
        # 模板: Dijkstra求起终点的最短路，注意只能是正权值可以提前返回结果，并返回对应经过的路径
        n = len(dct)
        dis = [inf] * n
        stack = [[0, src]]
        dis[src] = 0
        father = [-1] * n  # 记录最短路的上一跳
        while stack:
            d, i = heapq.heappop(stack)
            if dis[i] < d:
                continue
            if i == dst:
                break
            for j, w in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    father[j] = i
                    heapq.heappush(stack, [dj, j])
        if dis[dst] == inf:
            return [], inf
        # 向上回溯路径
        path = []
        i = dst
        while i != -1:
            path.append(i)
            i = father[i]
        return path, dis[dst]

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
        # 模板：使用Dijkstra计算严格次短路  # 也可以计算非严格次短路
        n = len(dct)
        dis = [[inf] * 2 for _ in range(n)]
        dis[src][0] = 0
        stack = [[0, src]]
        while stack:
            d, i = heapq.heappop(stack)
            if dis[i][1] < d:
                continue
            for j, w in dct[i]:
                if dis[j][0] > d + w:
                    dis[j][1] = dis[j][0]
                    dis[j][0] = d + w
                    heapq.heappush(stack, [d + w, j])
                elif dis[j][0] < d + w < dis[j][1]:  # 非严格修改为 d+w < dis[j][1]
                    dis[j][1] = d + w
                    heapq.heappush(stack, [d + w, j])
        return dis

    @staticmethod
    def get_shortest_by_bfs_inf_odd(dct: List[List[int]], src):
        # 模板: 使用 01BFS 求最短的奇数距离与偶数距离
        n = len(dct)
        dis = [[inf, inf] for _ in range(n)]
        stack = deque([[src, 0]])
        dis[0][0] = 0
        while stack:
            i, x = stack.popleft()
            for j in dct[i]:
                dd = x + 1
                if dis[j][dd % 2] == inf:
                    dis[j][dd % 2] = x + 1
                    stack.append([j, x + 1])
        return dis

    @staticmethod
    def get_shortest_by_bfs_inf(dct: List[List[int]], src):
        # 模板: 使用 01 BFS 求最短路
        n = len(dct)
        dis = [inf] * n
        stack = deque([src])
        dis[src] = 0
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                if dis[j] == inf:
                    dis[j] = dis[i] + 1
                    stack.append(j)
        return dis

    @staticmethod
    def get_dijkstra_result_edge(dct: List[List[int]], src: int) -> List[float]:
        # 模板: Dijkstra求最短路，变成负数求可以求最长路（还是正权值）
        n = len(dct)
        dis = [inf] * n
        stack = [[0, src]]
        dis[src] = 0

        while stack:
            d, i = heapq.heappop(stack)
            if dis[i] < d:
                continue
            for j, w in dct[i]:  # 链式前向星支持自环与重边
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    heapq.heappush(stack, [dj, j])
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
        dis = [inf] * n
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
            cost, cnt, i = heapq.heappop(stack)
            # 前面的代价已经比当前小了若是折扣次数更多则显然不可取
            if dis[i] <= cnt:
                continue
            if i == n - 1:
                return cost
            dis[i] = cnt
            for j, w in dct[i]:
                if cnt < dis[j]:
                    heapq.heappush(stack, [cost + w, cnt, j])
                if cnt + 1 < dis[j] and cnt + 1 <= discounts:
                    heapq.heappush(stack, [cost + w // 2, cnt + 1, j])

        return -1
    
    @staticmethod
    def lc_882(edges: List[List[int]], max_moves: int, n: int) -> int:
        # 模板：Dijkstra模板题
        dct = [[] for _ in range(n)]
        for i, j, c in edges:
            dct[i].append([j, c + 1])
            dct[j].append([i, c + 1])

        dis = Dijkstra().get_dijkstra_result(dct, 0)

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
    def lg_p1073(ac=FastIO()):
        # 模板：正反两遍建图，计算两个最短路
        n, m = ac.read_ints()
        nums = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        rev = [[] for _ in range(n)]
        for _ in range(m):
            x, y, z = ac.read_ints_minus_one()
            dct[x].append(y)
            rev[y].append(x)
            if z == 1:
                dct[y].append(x)
                rev[x].append(y)

        # 前面最小值
        floor = [inf]*n
        stack = [[nums[0], 0]]
        floor[0] = nums[0]
        while stack:
            d, i = heapq.heappop(stack)
            if floor[i] < d:
                continue
            for j in dct[i]:
                dj = ac.min(d, nums[j])
                if dj < floor[j]:
                    floor[j] = dj
                    heapq.heappush(stack, [dj, j])

        # 后面最大值
        ceil = [-inf]*n
        ceil[n - 1] = nums[n - 1]
        stack = [[-nums[n-1], n-1]]
        while stack:
            d, i = heapq.heappop(stack)
            if ceil[i] < d:
                continue
            for j in rev[i]:
                dj = ac.max(-d, nums[j])
                if dj > ceil[j]:
                    ceil[j] = dj
                    heapq.heappush(stack, [-dj, j])
        ans = max(ceil[i]-floor[i] for i in range(n))
        ac.st(ans)
        return

    @staticmethod
    def lg_p1300(ac=FastIO()):
        # 模板：Dijkstra求最短路
        m, n = ac.read_ints()
        grid = [ac.read_str() for _ in range(m)]
        ind = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        dct = {"E":0, "S": 1, "W": 2, "N": 3}
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

        dis = [[[inf]*4 for _ in range(n)] for _ in range(m)]
        dis[start[0]][start[1]][d] = 0
        stack = [[0, start[0], start[1], d]]
        while stack:
            pre, i, j, d = heapq.heappop(stack)
            if dis[i][j][d] < pre:
                continue
            flag = False
            for cost, r in [[1, (d-1)%4], [5, (d+1)%4], [0, d]]:
                x, y = i+ind[r][0], j+ind[r][1]
                if 0<=x<m and 0<=y<n and grid[x][y] != ".":
                    dj = pre+cost
                    if dj < dis[x][y][r]:
                        dis[x][y][r] = dj
                        heapq.heappush(stack, [dj, x, y, r])
                        flag = True
            if not flag:
                cost, r = 10, (d+2)%4
                x, y = i+ind[r][0], j+ind[r][1]
                if 0<=x<m and 0<=y<n and grid[x][y] != ".":
                    dj = pre+cost
                    if dj < dis[x][y][r]:
                        dis[x][y][r] = dj
                        heapq.heappush(stack, [dj, x, y, r])
        ac.st(min(dis[end[0]][end[1]]))
        return

    @staticmethod
    def lg_p1354(ac=FastIO()):

        # 模板：建图求最短路

        def dis(x1, y1, x2, y2):
            return ((x1-x2)**2+(y1-y2)**2)**0.5

        n = ac.read_int()
        nodes = [[0, 5], [10, 5]]
        line = []
        for _ in range(n):
            x, a1, a2, b1, b2 = ac.read_floats()
            nodes.append([x, a1])
            nodes.append([x, a2])
            nodes.append([x, b1])
            nodes.append([x, b2])
            line.append([x, a1, a2, b1, b2])

        def check():
            for x, a1, a2, b1, b2 in line:
                if left <= x <= right:
                    if not (a1<=k*x+bb<=a2) and not (b1<=k*x+bb<=b2):
                        return False
            return True

        start = 0
        end = 1
        m = len(nodes)
        dct = [dict() for _ in range(m)]
        for i in range(m):
            for j in range(i+1, m):
                a, b = nodes[i]
                c, d = nodes[j]
                if a == c:
                    continue
                k = (d-b)/(c-a)
                bb = d-k*c
                left, right = min(a, c), max(a, c)
                if check():
                    x = dis(a, b, c, d)
                    dct[i][j] = dct[j][i] = x
        ans = Dijkstra().get_dijkstra_result(dct, start)[end]
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p1608(ac=FastIO()):
        # 模板：使用Dijkstra计算有向与无向、带权与不带权的最短路数量（最短路计数）
        n, m = ac.read_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            i, j, w = ac.read_ints()
            dct[i - 1][j - 1] = ac.min(dct[i - 1].get(j - 1, inf), w)
        cnt, dis = Dijkstra().get_dijkstra_cnt(dct, 0)
        if dis[-1] == inf:
            ac.st("No answer")
        else:
            ac.lst([dis[-1], cnt[-1]])
        return

    @staticmethod
    def lg_p1828(ac=FastIO()):

        # 模板：多个单源Dijkstra最短路计算
        n, p, c = ac.read_ints()
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
            dis = Dijkstra().get_dijkstra_result(dct, i)
            for j in range(p):
                total[j] += dis[j] * cnt[i]
        ac.st(min(total))
        return

    @staticmethod
    def lg_p2047(ac=FastIO()):
        # 模板：Dijkstra计算经过每个点的所有最短路条数占比
        n, m = ac.read_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            a, b, c = ac.read_ints()
            a -= 1
            b -= 1
            dct[a][b] = dct[b][a] = c

        # 计算最短路距离与条数
        dis = []
        cnt = []
        for i in range(n):
            cc, dd = Dijkstra().get_dijkstra_cnt(dct, i)
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
        n, m = ac.read_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            i, j, w = ac.read_ints_minus_one()
            dct[i][j] = dct[j][i] = w + 1
        path, dis = Dijkstra().dijkstra_src_to_dst_path(dct, 0, n - 1)

        # 枚举边重新计算最短路
        ans = 0
        k = len(path)
        for a in range(k - 1):
            i, j = path[a], path[a + 1]
            dct[i][j] = dct[j][i] = dct[j][i] * 2
            _, cur = Dijkstra().dijkstra_src_to_dst_path(dct, 0, n - 1)
            ans = ac.max(ans, cur - dis)
            dct[i][j] = dct[j][i] = dct[j][i] // 2
        ac.st(ans)
        return

    @staticmethod
    def lg_p2269(ac=FastIO()):
        # 模板：比较两个维度的Dijkstra计算
        n, src, dst = ac.read_ints()
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
            ll, tt, i = heapq.heappop(stack)
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
                        heapq.heappush(stack, [nex_ll, nex_tt, j])
        res_ll = dis[dst][0]
        res_tt = dis[dst][1]
        ac.lst([res_tt, "%.4f" % res_ll])
        return

    @staticmethod
    def lg_p2349(ac=FastIO()):

        # 模板：比较两个项相加的最短路
        n, m = ac.read_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            u, v, w = ac.read_ints()
            u -= 1
            v -= 1
            dct[u][v] = ac.min(dct[u].get(v, inf), w)
            dct[v][u] = ac.min(dct[v].get(u, inf), w)

        # 最短路模板
        dis = [inf] * n
        stack = [[0, 0, 0, 0]]
        dis[0] = 0
        while stack:
            dd, d, ceil, i = heapq.heappop(stack)
            if dis[i] < dd:
                continue
            if i == n - 1:
                break
            for j in dct[i]:
                dj = d + dct[i][j] + ac.max(ceil, dct[i][j])
                if dj < dis[j]:
                    dis[j] = dj
                    heapq.heappush(stack, [dj, d + dct[i][j], ac.max(ceil, dct[i][j]), j])
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

        n, w = ac.read_ints()
        m = ac.read_float()
        nums = [ac.read_list_ints() for _ in range(n)]
        dct = [set() for _ in range(n)]
        for _ in range(w):
            i, j = ac.read_ints_minus_one()
            dct[i].add(j)
            dct[j].add(i)

        n = len(dct)
        visit = [inf] * n
        stack = [[0, 0]]
        visit[0] = 0
        while stack:
            d, i = heapq.heappop(stack)
            if visit[i] < d:
                continue
            if i == n - 1:
                break
            for j in range(n):
                dj = dis(i, j) + d
                if dj < visit[j]:
                    visit[j] = dj
                    heapq.heappush(stack, [dj, j])
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
            d, i = heapq.heappop(stack)
            if dis0[i] < d:
                continue
            for ind, j in dct[i]:
                dj = edges[ind][2] + d
                if dj < dis0[j]:
                    dis0[j] = dj
                    heapq.heappush(stack, [dj, j])
        if dis0[destination] > target:
            return []

        # 第二遍最短路
        dis1 = [inf] * n
        stack = [[0, source]]
        dis1[source] = 0
        while stack:
            d, i = heapq.heappop(stack)
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
                    heapq.heappush(stack, [dj, j])

        if dis1[destination] == target:
            return edges
        return []

    @staticmethod
    def cf_715b(ac=FastIO()):
        # 模板：经典两遍最短路，贪心动态更新路径权值
        n, m, target, source, destination = ac.read_ints()
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
            d, i = heapq.heappop(stack)
            if dis0[i] < d:
                continue
            for ind, j in dct[i]:
                dj = edges[ind][2] + d
                if dj < dis0[j]:
                    dis0[j] = dj
                    heapq.heappush(stack, [dj, j])
        if dis0[destination] > target:
            ac.st("NO")
            return

        # 第二遍最短路
        dis1 = [inf] * n
        stack = [[0, source]]
        dis1[source] = 0
        while stack:
            d, i = heapq.heappop(stack)
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
                    heapq.heappush(stack, [dj, j])

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
        n, m = ac.read_ints()
        dct = [dict() for _ in range(n)]
        cnt = 0
        for _ in range(m):
            x, y, s = ac.read_ints()
            cnt += s
            x -= 1
            y -= 1
            dct[x][y] = s
            dct[y][x] = s

        dis = [[inf, -inf] for _ in range(n)]
        stack = [[0, 0, 0]]
        dis[0] = [0, 0]

        while stack:
            dd, one, i = heapq.heappop(stack)
            if dis[i] < [dd, one]:
                continue
            for j in dct[i]:
                w = dct[i][j]
                dj = [dd+1, one-w]
                if dj < dis[j]:
                    dis[j] = dj
                    heapq.heappush(stack, [dj[0], dj[1], j])
        ans = cnt+dis[-1][1]+dis[-1][0]+dis[-1][1]
        ac.st(ans)
        return

    @staticmethod
    def lg_p3956(ac=FastIO()):
        # 模板：Dijkstra计算最小代价

        m, n = ac.read_ints()
        grid = [[-1] * m for _ in range(m)]
        for _ in range(n):
            x, y, c = ac.read_ints()
            grid[x - 1][y - 1] = c

        stack = [[0, 0, grid[0][0], 0, 0]]
        final = -1
        visit = defaultdict(lambda: inf)
        while stack:
            cost, magic, color, i, j = heapq.heappop(stack)
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
                            heapq.heappush(stack, [cost + int(color != grid[a][b]), 0, grid[a][b], a, b])
                        else:
                            heapq.heappush(stack, [cost + 2 + int(color != 0), 1, 0, a, b])
                            heapq.heappush(stack, [cost + 2 + int(color != 1), 1, 1, a, b])

                    else:
                        if grid[a][b] != -1:
                            heapq.heappush(stack, [cost + int(color != grid[a][b]), 0, grid[a][b], a, b])
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

        dis = Dijkstra().get_dijkstra_result(dct, b)

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
        n, m, k = ac.read_ints()
        if k:
            visit = set(ac.read_list_ints_minus_one())
        else:
            visit = set()
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            a, b, c = ac.read_ints_minus_one()
            dct[a][b] = dct[b][a] = ac.min(dct[a].get(a, inf), c + 1)
        x, y = ac.read_ints_minus_one()

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
        n, m, t = ac.read_ints()
        nums = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            a, b, c = ac.read_ints_minus_one()
            dct[a].append([b, c + 1])
            dct[b].append([a, c + 1])
        for i in range(n):
            dct[i].sort()
        # 先跑一遍最短路
        dis = [inf] * n
        stack = [[0, 0]]
        dis[0] = 0
        while stack:
            d, i = heapq.heappop(stack)
            if dis[i] < d:
                continue
            for j, w in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    heapq.heappush(stack, [dj, j])

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
        n, m, q = ac.read_ints()
        dct = [[] for _ in range(n)]
        for i in range(m):
            x, y = ac.read_ints_minus_one()
            dct[x].append(y)
            dct[y].append(x)

        dis = Dijkstra().get_shortest_by_bfs_inf_odd(dct, 0)
        for _ in range(q):
            x, y = ac.read_ints()
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
        n, m = ac.read_ints()
        dct = [[] for _ in range(n)]
        nums = []
        while len(nums) < 2 * m:
            nums.extend(ac.read_list_ints_minus_one())
        for i in range(0, 2 * m, 2):
            x, y = nums[i], nums[i + 1]
            dct[x].append(y)
            dct[y].append(x)
        # 出发与时间约束
        s1, t1, s2, t2 = ac.read_ints()
        s1 -= 1
        s2 -= 1
        dis0 = Dijkstra().get_shortest_by_bfs_inf(dct, 0)
        dis1 = Dijkstra().get_shortest_by_bfs_inf(dct, s1)
        dis2 = Dijkstra().get_shortest_by_bfs_inf(dct, s2)
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
        n, m = ac.read_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j, c, f = ac.read_ints()
            dct[i - 1].append([j - 1, c, f])
            dct[j - 1].append([i - 1, c, f])

        dis = [inf] * n
        stack = [[0, 0, 0, inf]]
        dis[0] = 0
        while stack:
            d, i, cost, flow = heapq.heappop(stack)
            if dis[i] < d:
                continue
            for j, c, f in dct[i]:
                dj = -ac.min(flow, f) / (cost + c)
                if dj < dis[j]:
                    dis[j] = dj
                    heapq.heappush(stack, [dj, j, cost + c, ac.min(flow, f)])
        ac.st(int(-dis[-1] * 10**6))
        return

    @staticmethod
    def lg_p5905(ac=FastIO()):
        # 模板：有向带权图可能有负权 Johnson 全源最短路计算所有点对的最短路
        n, m = ac.read_ints()
        dct = [[] for _ in range(n + 1)]
        for _ in range(m):
            u, v, w = ac.read_ints()
            dct[u].append([v, w])
        for i in range(1, n + 1):
            dct[0].append([i, 0])
        # 首先使用 Bellman-Ford 的队列实现算法 SPFA 判断有没有负环
        flag, h, _ = SPFA().negative_circle_edge(dct)
        if flag == "YES":
            ac.st(-1)
            return
        # 其次建立新图枚举起点跑 Dijkstra
        for i in range(n + 1):
            k = len(dct[i])
            for x in range(k):
                j, w = dct[i][x]
                dct[i][x][1] = w + h[i] - h[j]
        dj = Dijkstra()
        for i in range(1, n + 1):
            ans = 0
            dis = dj.get_dijkstra_result_edge(dct, i)
            for j in range(1, n + 1):
                # 还原之后才为原图最短路
                ans += j * (dis[j] + h[j] - h[i]) if dis[j] < inf else j * 10**9
            ac.st(ans)
        return

    @staticmethod
    def lg_p5930(ac=FastIO()):
        # 模板：经典接雨水使用 Dijkstra 进行计算
        m, n = ac.read_ints()
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
        heapq.heapify(stack)
        while stack:
            d, i, j = heapq.heappop(stack)
            if visit[i][j] < d:
                continue
            for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                if 0 <= x < m and 0 <= y < n:
                    # 每条路径边权最大值当中的最小值
                    dj = ac.max(grid[x][y], d)
                    if dj < visit[x][y]:
                        visit[x][y] = dj
                        heapq.heappush(stack, [dj, x, y])
        ans = 0
        for i in range(m):
            for j in range(n):
                ans += visit[i][j] - grid[i][j]
        ac.st(ans)
        return

    @staticmethod
    def lg_p6063(ac=FastIO()):
        # 模板：经典Dijkstra应用接雨水
        n, m = ac.read_ints()
        grid = [ac.read_list_ints() for _ in range(m)]

        # 使用虚拟化超级汇点初始化起点
        stack = []
        for i in [0, m - 1]:
            for j in range(n):
                stack.append([grid[i][j], i, j])
        for i in range(1, m - 1):
            for j in [0, n - 1]:
                stack.append([grid[i][j], i, j])
        heapq.heapify(stack)

        # 使用最短路算法寻找每个格子到达超级汇点的路径途中最大值里面的最小值
        ans = 0
        while stack:
            dis, i, j = heapq.heappop(stack)
            if grid[i][j] == -1:
                continue
            ans += 0 if dis < grid[i][j] else dis - grid[i][j]
            grid[i][j] = -1
            for x, y in [[i - 1, j], [i + 1, j], [i, j + 1], [i, j - 1]]:
                if 0 <= x < m and 0 <= y < n and grid[x][y] != -1:
                    heapq.heappush(stack, [dis if dis > grid[x][y] else grid[x][y], x, y])
        ac.st(ans)
        return

    @staticmethod
    def lc_2714_1(n: int, edges: List[List[int]], s: int, d: int, k: int) -> int:
        # 模板：经典带约束的最短路，也可以使用分层 Dijkstra 求解
        dct = [[] for _ in range(n)]
        for u, v, w in edges:
            dct[u].append([v, w])
            dct[v].append([u, w])

        visit = [[inf]*(k+1) for _ in range(n)]
        stack = [[0, 0, s]]
        visit[s][0] = 0
        while stack:
            dis, c, i = heapq.heappop(stack)
            if i == d:
                return dis
            if visit[i][c] < dis:
                continue
            for j, w in dct[i]:
                if c + 1 <= k and dis < visit[j][c+1]:
                    visit[j][c + 1] = dis
                    heapq.heappush(stack, [dis, c + 1, j])
                if dis + w < visit[j][c]:
                    visit[j][c] = dis + w
                    heapq.heappush(stack, [dis + w, c, j])
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
            dis, c, i = heapq.heappop(stack)
            if i == d:
                return dis
            if cnt[i] < c:
                continue
            cnt[i] = c
            for j, w in dct[i]:
                if c + 1 < cnt[j] and c + 1 <= k:
                    heapq.heappush(stack, [dis, c + 1, j])
                if c < cnt[j]:
                    heapq.heappush(stack, [dis + w, c, j])
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
            d, i, j = heapq.heappop(stack)
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
                        heapq.heappush(stack, [dj, x, y])
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
            d, i = heapq.heappop(stack)
            if dis0[i] < d:
                continue
            for ind, j in dct[i]:
                dj = edges[ind][2] + d
                if dj < dis0[j]:
                    dis0[j] = dj
                    heapq.heappush(stack, [dj, j])
        if dis0[destination] > target:
            return []

        # 第二遍最短路
        dis1 = [inf] * n
        stack = [[0, source]]
        dis1[source] = 0
        while stack:
            d, i = heapq.heappop(stack)
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
                    heapq.heappush(stack, [dj, j])

        if dis1[destination] == target:
            return edges
        return []

    @staticmethod
    def lg_p6512(ac=FastIO()):
        # 模板：经典最短路加DP
        n, m, k = ac.read_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j, w = ac.read_ints_minus_one()
            dct[i].append([j, w + 1])
            dct[j].append([i, w + 1])
        dis = []
        for i in range(n):
            dis.append(Dijkstra().get_dijkstra_result_edge(dct, i))
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
            a, b, c = ac.read_ints_minus_one()
            c += 1
            dct[a].append([b, c])
            dct[a + n].append([b + n, c])
        for i in range(n):
            dct[i].append([i + n, price[i] / 2])

        dis = [inf] * 2 * n
        stack = [[0, 0]]
        dis[0] = 0
        while stack:
            total, i = heapq.heappop(stack)
            if dis[i] < total:
                continue
            for j, w in dct[i]:
                dj = total + w
                if dj < dis[j]:
                    dis[j] = dj
                    heapq.heappush(stack, [dj, j])
        ac.st(int(dis[n]))
        return

    @staticmethod
    def lc_1786(n: int, edges: List[List[int]]) -> int:

        # 模板：经典dijkstra受限最短路计数（类似最短路计数）
        dct = defaultdict(dict)
        for i, j, w in edges:
            dct[i-1][j-1] = w
            dct[j-1][i-1] = w
        mod = 10**9 + 7
        # 使用倒序进行最短路搜寻
        dis = [float('inf')]*n
        cnt = [0]*n
        cnt[n-1] = 1
        dis[n-1] = 0
        # 定义好初始值
        stack = [[0, n-1]]
        while stack:
            cur_dis, cur = heapq.heappop(stack)
            if dis[cur] < cur_dis:
                continue
            dis[cur] = cur_dis
            for nex in dct[cur]:
                # 如果到达下一个点更近，则更新值
                if dis[nex] > dis[cur] + dct[cur][nex]:
                    dis[nex] = dis[cur] + dct[cur][nex]
                    heapq.heappush(stack, [dis[nex], nex])
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
            cost, tm, i = heapq.heappop(stack)
            # 前面的代价已经比当前小了若是换乘次数更多则显然不可取
            if dis[i] <= tm: 
                continue
            if i == n-1:
                return cost
            dis[i] = tm
            for j, w in dct[i]:
                if tm + w < dis[j]:
                    heapq.heappush(stack, [cost + passing_fees[j], tm + w, j])
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
        return Dijkstra().get_dijkstra_cnt(dct, 0)[0][n-1] % mod

    @staticmethod
    def ac_3628(ac=FastIO()):
        # 模板：经典最短路生成树模板题
        n, m, k = ac.read_ints()
        dct = [[] for _ in range(n)]
        for ind in range(m):
            x, y, w = ac.read_ints()
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
            d, i = heapq.heappop(stack)
            if dis[i] < d:
                continue
            for j, w, _ in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    heapq.heappush(stack, [dj, j])

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
        n, m = ac.read_ints()
        rev = [[] for _ in range(n)]
        for _ in range(m):
            u, v = ac.read_ints_minus_one()
            rev[v].append([u, 1])

        k = ac.read_int()
        p = ac.read_list_ints_minus_one()
        cnt, dis = Dijkstra().get_dijkstra_cnt(rev, p[-1])

        floor = 0
        for i in range(k-1):
            if k-i-1 == dis[p[i]]:
                break
            if dis[p[i-1]] == dis[p[i]] + 1:
                continue
            else:
                floor += 1

        ceil = 0
        for i in range(k-1):
            if dis[p[i]] == dis[p[i+1]] + 1 and cnt[p[i]] == cnt[p[i+1]]:
                continue
            else:
                ceil += 1
        ac.lst([floor, ceil])

        return

    @staticmethod
    def ac_3797(ac=FastIO()):
        # 模板：经典最短路枚举增边排序贪心
        n, m, k = ac.read_ints()
        nums = ac.read_list_ints_minus_one()
        dct =[[] for _ in range(n)]
        for _ in range(m):
            x, y = ac.read_ints_minus_one()
            dct[x].append(y)
            dct[y].append(x)
        dis0 = Dijkstra().get_shortest_by_bfs(dct, 0)
        dis1 = Dijkstra().get_shortest_by_bfs(dct, n-1)
        nums.sort(key=lambda it: dis0[it]-dis1[it])

        ans = -1
        pre = -inf
        d = dis0[-1]
        for i in range(k):
            cur = pre+1+dis1[nums[i]]
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
        n, m = ac.read_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j, w = ac.read_ints()
            i -= 1
            j -= 1
            dct[i].append([j, w])
            dct[j].append([i, w])
        path, ans = Dijkstra().dijkstra_src_to_dst_path(dct, 0, n-1)
        if ans == inf:
            ac.st(-1)
        else:
            path.reverse()
            ac.lst([x+1 for x in path])
        return


class TestGeneral(unittest.TestCase):

    def test_dijkstra(self):
        djk = Dijkstra()
        dct = [{1: 1, 2: 4}, {2: 2}, {}]
        assert djk.get_dijkstra_result(dct, 0) == [0, 1, 3]
        return


if __name__ == '__main__':
    unittest.main()
