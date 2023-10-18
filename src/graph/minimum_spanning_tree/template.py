import math
from collections import deque
from heapq import heappop, heappush
from typing import List
from math import inf

from graph.union_find.template import UnionFind


class MinimumSpanningTree:
    def __init__(self, edges, n, method="kruskal"):
        # n个节点
        self.n = n
        # m条权值边edges
        self.edges = edges
        self.cost = 0
        self.cnt = 0
        self.gen_minimum_spanning_tree(method)
        return

    def gen_minimum_spanning_tree(self, method):

        if method == "kruskal":
            # 边优先
            self.edges.sort(key=lambda item: item[2])
            # 贪心按照权值选择边进行连通合并
            uf = UnionFind(self.n)
            for x, y, z in self.edges:
                if uf.union(x, y):
                    self.cost += z
            # 不能形成生成树
            if uf.part != 1:
                self.cost = -1
        else:
            # 点优先使用 Dijkstra求解
            dct = [dict() for _ in range(self.n)]
            for i, j, w in self.edges:
                c = dct[i].get(j, float("inf"))
                c = c if c < w else w
                dct[i][j] = dct[j][i] = c
            dis = [inf] * self.n
            dis[0] = 0
            visit = [0]*self.n
            stack = [[0, 0]]
            while stack:
                d, i = heappop(stack)
                if visit[i]:
                    continue
                visit[i] = 1
                self.cost += d  # 连通花费的代价
                self.cnt += 1  # 连通的节点数
                for j in dct[i]:
                    w = dct[i][j]
                    if w < dis[j]:
                        dis[j] = w
                        heappush(stack, [w, j])
        return


class TreeAncestorWeightSecond:

    def __init__(self, dct):
        # 默认以 0 为根节点
        n = len(dct)
        self.parent = [-1] * n
        self.depth = [-1] * n
        stack = deque([0])
        self.depth[0] = 0
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                if self.depth[j] == -1:
                    self.depth[j] = self.depth[i] + 1
                    self.parent[j] = i
                    stack.append(j)

        # 根据节点规模设置层数
        self.cols = max(2, math.ceil(math.log2(n)))
        self.dp = [[-1] * self.cols for _ in range(n)]
        self.weight = [[[-1, -1] for _ in range(self.cols)] for _ in range(n)]  # 边权的最大值与次大值
        for i in range(n):
            self.dp[i][0] = self.parent[i]
            if self.parent[i] != -1:
                self.weight[i][0] = [dct[self.parent[i]][i], -1]

        # 动态规划设置祖先初始化, dp[node][j] 表示 node 往前推第 2^j 个祖先
        for j in range(1, self.cols):
            for i in range(n):
                father = self.dp[i][j - 1]
                self.weight[i][j] = self.update(self.weight[i][j], self.weight[i][j-1])
                if father != -1:
                    self.dp[i][j] = self.dp[father][j - 1]
                    self.weight[i][j] = self.update(self.weight[i][j], self.weight[father][j-1])
        return

    @staticmethod
    def update(lst1, lst2):
        a, b = lst1
        c, d = lst2
        # 更新最大值与次大值
        for x in [c, d]:
            if x >= a:
                a, b = x, a
            elif x >= b:
                b = x
        return [a, b]

    def get_dist_weight_max_second(self, x: int, y: int) -> List[int]:
        # 计算任意点的最短路上的权重最大值与次大值
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        ans = [-1, -1]
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            ans = self.update(ans, self.weight[x][int(math.log2(d))])
            x = self.dp[x][int(math.log2(d))]
        if x == y:
            return ans

        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.dp[x][k] != self.dp[y][k]:
                ans = self.update(ans, self.weight[x][k])
                ans = self.update(ans, self.weight[y][k])
                x = self.dp[x][k]
                y = self.dp[y][k]

        ans = self.update(ans, self.weight[x][0])
        ans = self.update(ans, self.weight[y][0])
        return ans
