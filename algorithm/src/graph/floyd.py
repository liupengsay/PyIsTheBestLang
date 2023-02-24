
import unittest
from collections import defaultdict
from functools import lru_cache

from algorithm.src.fast_io import FastIO

"""

算法：Floyd（单源最短路经算法）
功能：计算点到有向或者无向图里面其他点的最近距离，也可以计算最长路
题目：

===================================洛谷===================================
P1119 灾后重建 （https://www.luogu.com.cn/problem/P1119）离线查询加Floyd动态更新经过中转站的起终点距离
P1476 休息中的小呆（https://www.luogu.com.cn/problem/P1476）Floyd求最长路
P2009 跑步（https://www.luogu.com.cn/problem/P2009）Floyd求最短路
P2419 [USACO08JAN]Cow Contest S（https://www.luogu.com.cn/problem/P2419）看似拓扑排序其实是使用Floyd进行拓扑排序
P2910 [USACO08OPEN]Clear And Present Danger S（https://www.luogu.com.cn/problem/P2910）最短路计算之后进行查询
P3906 Geodetic集合（https://www.luogu.com.cn/problem/P3906）Floyd算法计算最短路径上经过的点集合
P6464 [传智杯 #2 决赛] 传送门（https://www.luogu.com.cn/problem/P6464）枚举边之后进行Floyd算法更新计算，经典理解Floyd的原理题，经典借助中间两点更新最短距离
P6175 无向图的最小环问题（https://www.luogu.com.cn/problem/P6175）经典使用Floyd枚举三个点之间的距离和

参考：OI WiKi（xx）
"""


class Floyd:
    def __init__(self):
        return

    @staticmethod
    def shortest_path_node(n, m, edges, i, j):
        # 模板: 计算i与j之间所有可行的最短路经过的点
        inf = float("inf")
        dp = [[inf] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 0
        for u, v in edges:
            dp[u][v] = dp[v][u] = 1
        for k in range(n):
            for i in range(n):
                for j in range(i+1, n):  # 优化
                    a = dp[i][k] + dp[k][j]
                    b = dp[i][j]
                    dp[j][i] = dp[i][j] = a if a < b else b

        ans = [x + 1 for x in range(n) if dp[i][x] + dp[x][j] == dp[i][j]]
        return ans

    @staticmethod
    def longest_path_length(edges, n):

        # 索引从1-n并求1-n的最长路
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        for i, j, k in edges:  # k >= 0
            dp[i][j] = k

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                for k in range(1, n + 1):
                    if i != j and j != k and i != k and dp[i][k] and dp[k][j]:
                        if dp[i][j] < dp[i][k] + dp[k][j]:
                            dp[i][j] = dp[i][k] + dp[k][j]

        length = dp[1][n]
        path = []
        for i in range(1, n + 1):
            if dp[1][i] + dp[i][n] == dp[1][n]:
                path.append(i)
        return length, path

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
        # 模板：利用Floyd算法特点和修复的中转站更新最短距离
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

class TestGeneral(unittest.TestCase):

    def test_luogu(self):
        luogu = Luogu()
        n = 4
        repair = [1, 2, 3, 4]
        edges = [[0, 2, 1], [2, 3, 1], [3, 1, 2], [2, 1, 4], [0, 3, 5]]
        queries = [[2, 0, 2], [0, 1, 2], [0, 1, 3], [0, 1, 4]]
        assert luogu.main_p1119(n, repair, edges, queries) == [-1, -1, 5, 4]
        return


if __name__ == '__main__':
    unittest.main()
