from math import inf
from typing import List

from utils.fast_io import FastIO


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


class Graph:
    # 模板：Floyd动态更新最短路 LC2642 也可以使用 Dijkstra 暴力求解
    def __init__(self, n: int, edges: List[List[int]]):
        d = [[inf] * n for _ in range(n)]
        for i in range(n):
            d[i][i] = 0
        for x, y, w in edges:
            d[x][y] = w  # 添加一条边（输入保证没有重边和自环）
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j])
        self.d = d

    def add_edge(self, e: List[int]) -> None:
        d = self.d
        n = len(d)
        x, y, w = e
        if w >= d[x][y]:  # 无需更新
            return
        for i in range(n):
            for j in range(n):
                # Floyd 在增加一条边时动态更新最短路
                d[i][j] = min(d[i][j], d[i][x] + w + d[y][j])

    def shortest_path(self, start: int, end: int) -> int:
        ans = self.d[start][end]
        return ans if ans < inf else -1

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
