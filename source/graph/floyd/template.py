from math import inf
from typing import List


class Floyd:
    def __init__(self):
        return

    @staticmethod
    def shortest_path(n, dp):
        # Calculate the shortest path between all point pairs using the Floyd algorithm
        for k in range(n):  # mid point
            for i in range(n):  # start point
                for j in range(n):  # end point
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j])
        return dp


class GraphLC2642:
    def __init__(self, n: int, edges: List[List[int]]):
        d = [[inf] * n for _ in range(n)]
        for i in range(n):
            d[i][i] = 0
        for x, y, w in edges:
            d[x][y] = w  # initial
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j])
        self.d = d

    def add_edge(self, e: List[int]) -> None:
        d = self.d
        n = len(d)
        x, y, w = e
        if w >= d[x][y]:
            return
        for i in range(n):
            for j in range(n):
                # add another edge
                d[i][j] = min(d[i][j], d[i][x] + w + d[y][j])

    def shortest_path(self, start: int, end: int) -> int:
        ans = self.d[start][end]
        return ans if ans < inf else -1
