
import heapq
import unittest
from collections import defaultdict, deque
from heapq import heappush, heappop
from itertools import accumulate
from operator import add
from typing import List, Set
from collections import Counter

from src.fast_io import FastIO, inf

"""
算法：Dijkstra（单源最短路经算法）、严格次短路、要保证加和最小因此只支持非负数权值、或者取反全部为非正数计算最长路、最短路生成树
功能：计算点到有向或者无向图里面其他点的最近距离、带约束的最短路、分层Dijkstra、有向图最小环、无向图最小环
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
1786. 从第一个节点出发到最后一个节点的受限路径数（https://leetcode.cn/problems/number-of-restricted-paths-from-first-to-last-node/）经典dijkstra受限最短路计数（类似最短路计数），也可以将无向图转换为DAG问题
1928. 规定时间内到达终点的最小花费（https://leetcode.cn/problems/minimum-cost-to-reach-destination-in-time/）经典Dijkstra带约束的最短路，也可根据无后效性类似Floyd的动态规划求解
LCP 75. 传送卷轴（https://leetcode.cn/problems/rdmXM7/）首先BFS之后计算最大值最小的最短路
1976. 到达目的地的方案数（https://leetcode.cn/problems/number-of-ways-to-arrive-at-destination/）经典Dijkstra最短路计数模板题
2045. 到达目的地的第二短时间（https://leetcode.cn/problems/second-minimum-time-to-reach-destination/）严格次短路计算模板题，距离更新时需要注意变化
2093. 前往目标城市的最小费用（https://leetcode.cn/problems/minimum-cost-to-reach-city-with-discounts/）经典Dijkstra带约束的最短路
882. 细分图中的可到达节点（https://leetcode.cn/problems/reachable-nodes-in-subdivided-graph/description/）Dijkstra模板题
2577. 在网格图中访问一个格子的最少时间（https://leetcode.cn/problems/minimum-time-to-visit-a-cell-in-a-grid/）Dijkstra经典变形二维矩阵题目
2065. 最大化一张图中的路径价值（https://leetcode.cn/problems/maximum-path-quality-of-a-graph/）经典回溯，正解使用Dijkstra跑最短路剪枝

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
G. Reducing Delivery Cost（https://codeforces.com/contest/1433/problem/G）经典全源Dijkstra最短路枚举
G. Counting Shortcuts（https://codeforces.com/contest/1650/problem/G）经典Dijkstra最短路与严格次短路计数，正解为01BFS

================================AtCoder================================
F - Pure（https://atcoder.jp/contests/abc142/tasks/abc142_f）经典子图寻找，转换为有向图的最小环问题

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
        stack = [(0, src)]
        dis[src] = 0

        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            for j, w in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, (dj, j))
        return dis

    @staticmethod
    def get_dijkstra_cnt(dct: List[List[int]], src: int) -> (List[int], List[any]):
        # 模板: Dijkstra求最短路条数（最短路计算）
        n = len(dct)
        dis = [inf]*n
        stack = [(0, src)]
        dis[src] = 0
        cnt = [0]*n
        cnt[src] = 1
        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            for j, w in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    # 最短距离更新，重置计数
                    cnt[j] = cnt[i]
                    heappush(stack, (dj, j))
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
        stack = [(dis[src], src)]
        # 限制只能跑 limit 的点到 target 中的点
        while stack and target:
            d, i = heappop(stack)
            if i in target:
                target.discard(i)
            if dis[i] < d:
                continue
            for j, w in dct[i]:
                if j not in limit:
                    dj = w + d
                    if dj < dis[j]:
                        dis[j] = dj
                        heappush(stack, (dj, j))
        return dis

    @staticmethod
    def dijkstra_src_to_dst_path(dct: List[List[int]], src: int, dst: int) -> (List[int], any):
        # 模板: Dijkstra求起终点的最短路，注意只能是正权值可以提前返回结果，并返回对应经过的路径
        n = len(dct)
        dis = [inf] * n
        stack = [(0, src)]
        dis[src] = 0
        father = [-1] * n  # 记录最短路的上一跳
        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            if i == dst:
                break
            for j, w in dct[i]:
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    father[j] = i
                    heappush(stack, (dj, j))
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
        stack = [(-1, src)]
        dis[src] = 1
        while stack:
            d, i = heappop(stack)
            d = -d
            if dis[i] > d:
                continue
            for j in dct[i]:
                dj = dct[i][j] * d
                if dj > dis[j]:
                    dis[j] = dj
                    heappush(stack, (-dj, j))
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
        stack = [(0, src)]
        while stack:
            d, i = heappop(stack)
            if dis[i][1] < d:
                continue
            for j, w in dct[i]:
                if dis[j][0] > d + w:
                    dis[j][1] = dis[j][0]
                    dis[j][0] = d + w
                    heappush(stack, (d + w, j))
                elif dis[j][0] < d + w < dis[j][1]:  # 非严格修改为 d+w < dis[j][1]
                    dis[j][1] = d + w
                    heappush(stack, (d + w, j))
        return dis

    @staticmethod
    def get_second_shortest_path_cnt(dct: List[List[int]], src, mod=-1):
        # 模板：使用Dijkstra计算严格次短路的条数   # 也可以计算非严格次短路
        n = len(dct)
        dis = [[inf] * 2 for _ in range(n)]
        dis[src][0] = 0
        stack = [(0, src, 0)]
        cnt = [[0]*2 for _ in range(n)]
        cnt[src][0] = 1
        while stack:
            d, i, state = heappop(stack)
            if dis[i][1] < d:
                continue
            pre = cnt[i][state]
            for j, w in dct[i]:
                dd = d+w
                if dis[j][0] > dd:
                    dis[j][0] = dd
                    cnt[j][0] = pre
                    heappush(stack, (d + w, j, 0))
                elif dis[j][0] == dd:
                    cnt[j][0] += pre
                    if mod != -1:
                        cnt[j][0] %= mod
                elif dis[j][0] < dd < dis[j][1]:  # 非严格修改为 d+w < dis[j][1]
                    dis[j][1] = d + w
                    cnt[j][1] = pre
                    heappush(stack, (d + w, j, 1))
                elif dd == dis[j][1]:
                    cnt[j][1] += pre
                    if mod != -1:
                        cnt[j][1] %= mod
        return dis, cnt


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
        stack = [(0, src)]
        dis[src] = 0

        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            for j, w in dct[i]:  # 链式前向星支持自环与重边
                dj = w + d
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, (dj, j))
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
                _, x = heappop(q)
                for child in dct[x]:
                    if dist[x] > ans:
                        break
                    if dist[child] > dct[x][child] + dist[x]:
                        dist[child] = dct[x][child] + dist[x]
                        par[child] = x
                        heappush(q, [dist[child], child])
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
                d, i = heappop(stack)
                if dis[i] < d:
                    continue
                if i == y:
                    break
                for j in dct[i]:
                    dj = dct[i][j] + d
                    if dj < dis[j]:
                        dis[j] = dj
                        heappush(stack, (dj, j))

            ans = ans if ans < dis[y] + w else dis[y] + w
            dct[x][y] = w
            dct[y][x] = w
        return ans if ans < inf else -1


