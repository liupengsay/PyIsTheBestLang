
import unittest
from collections import defaultdict
from functools import lru_cache

from algorithm.src.fast_io import FastIO

"""

算法：Floyd（多源最短路经算法）、可以处理有向图无向图以及正负权边
功能：计算点到有向或者无向图里面其他点的最短路，也可以计算最长路，以及所有最长路最短路上经过的点（关键节点）
题目：

===================================洛谷===================================
P1119 灾后重建 （https://www.luogu.com.cn/problem/P1119）离线查询加Floyd动态更新经过中转站的起终点距离
P1476 休息中的小呆（https://www.luogu.com.cn/problem/P1476）Floyd 求索引从 1 到 n 的最长路并求所有在最长路上的点
P3906 Geodetic集合（https://www.luogu.com.cn/problem/P3906）Floyd算法计算最短路径上经过的点集合

P2009 跑步（https://www.luogu.com.cn/problem/P2009）Floyd求最短路
P2419 [USACO08JAN]Cow Contest S（https://www.luogu.com.cn/problem/P2419）看似拓扑排序其实是使用Floyd进行拓扑排序
P2910 [USACO08OPEN]Clear And Present Danger S（https://www.luogu.com.cn/problem/P2910）最短路计算之后进行查询，Floyd模板题
P6464 [传智杯 #2 决赛] 传送门（https://www.luogu.com.cn/problem/P6464）枚举边之后进行Floyd算法更新计算，经典理解Floyd的原理题，经典借助中间两点更新最短距离
P6175 无向图的最小环问题（https://www.luogu.com.cn/problem/P6175）经典使用Floyd枚举三个点之间的距离和，O(n^3)，也可以使用BFS或者Dijkstra计算
B3611 【模板】传递闭包（https://www.luogu.com.cn/problem/B3611）传递闭包模板题，使用FLoyd解法

参考：OI WiKi（xx）
"""


class Floyd:
    def __init__(self):
        # 模板：Floyd算法
        return

    @staticmethod
    def shortest_path(n, dp):
        # 使用 floyd 算法计算所有点对之间的最短路
        for k in range(n):  # 中间节点
            for i in range(n):  # 起始节点
                for j in range(n):  # 结束节点
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j])
        return dp


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1119(ac=FastIO()):
        # 模板：利用 Floyd 算法特点和修复的中转站更新最短距离（无向图）
        n, m = ac.read_ints()
        repair = ac.read_list_ints()
        # 设置初始值距离
        dis = [[ac.inf] * n for _ in range(n)]
        for i in range(m):
            a, b, c = ac.read_ints()
            dis[a][b] = dis[b][a] = c
        for i in range(n):
            dis[i][i] = 0

        # 修复村庄之后用Floyd算法更新以该村庄为中转的距离
        k = 0
        for _ in range(ac.read_int()):
            x, y, t = ac.read_ints()
            # 离线算法
            while k < n and repair[k] <= t:
                # k修复则更新以k为中转站的距离
                for a in range(n):
                    for b in range(a + 1, n):
                        dis[a][b] = dis[b][a] = ac.min(dis[a][k] + dis[k][b], dis[b][a])
                k += 1
            if dis[x][y] < ac.inf and x < k and y < k:
                ac.st(dis[x][y])
            else:
                ac.st(-1)
        return

    @staticmethod
    def lg_p1476(ac=FastIO()):
        # 模板：Floyd 求索引从 1 到 n 的最长路并求所有在最长路上的点（有向图）
        n = ac.read_int() + 1
        m = ac.read_int()
        dp = [[-ac.inf] * (n + 1) for _ in range(n + 1)]
        for _ in range(m):
            i, j, k = ac.read_ints()
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
        n, m = ac.read_ints()
        dp = [[ac.inf] * (n + 1) for _ in range(n + 1)]
        for _ in range(m):
            i, j = ac.read_ints()
            dp[i][j] = dp[j][i] = 1
        for i in range(1, n + 1):
            dp[i][i] = 0

        for k in range(1, n + 1):
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):  # 无向图这里就可以优化
                    dp[j][i] = dp[i][j] = ac.min(dp[i][j], dp[i][k] + dp[k][j])

        for _ in range(ac.read_int()):
            u, v = ac.read_ints()
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


class TestGeneral(unittest.TestCase):

    def test_solution(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
