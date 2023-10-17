import unittest
from collections import deque
from math import inf
from typing import List, Dict

from utils.fast_io import FastIO
from src.graph.dijkstra import Dijkstra



class SPFA:
    def __init__(self):
        return

    @staticmethod
    def negative_circle_edge(dct: List[List[int]], src=0, initial=0) -> (str, List[float], List[int]):
        # 模板: 判断是否存在负环与求解最短路（正数取反即可判断是否存在正权环以及最长路）
        n = len(dct)
        # 初始化距离
        dis = [inf] * n
        # 标识当前节点是否在栈中
        visit = [False] * n
        # 当前最小距离的路径边数
        cnt = [0] * n
        # 求带负权的最短路距离与路径边数
        queue = deque([src])
        # 队列与起点初始化默认从 0 出发
        dis[src] = initial
        visit[src] = True

        while queue:
            # 取出队列中的第一个节点
            u = queue.popleft()
            visit[u] = False
            # 更新当前节点的相邻节点的距离
            for v, w in dct[u]:  # 链式前向星支持自环与重边
                if dis[v] > dis[u] + w:
                    dis[v] = dis[u] + w
                    cnt[v] = cnt[u] + 1
                    if cnt[v] >= n:
                        return "YES", dis, cnt
                    # 如果相邻节点还没有在队列中，将它加入队列
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        # 不存在从起点出发的负环
        return "NO", dis, cnt

    @staticmethod
    def negative_circle(dct: List[Dict], src=0, initial=0) -> (str, List[float], List[int]):
        # 模板: 判断是否存在负环与求解最短路（正数取反即可判断是否存在正权环以及最长路）
        n = len(dct)
        # 初始化距离
        dis = [inf for _ in range(n)]
        # 标识当前节点是否在栈中
        visit = [False] * n
        # 当前最小距离的路径边数
        cnt = [0] * n
        # 求带负权的最短路距离与路径边数
        queue = deque([src])
        # 队列与起点初始化默认从 0 出发
        dis[src] = initial
        visit[src] = True

        while queue:
            # 取出队列中的第一个节点
            u = queue.popleft()
            visit[u] = False
            # 更新当前节点的相邻节点的距离
            for v in dct[u]:
                w = dct[u][v]
                if dis[v] > dis[u] + w:
                    dis[v] = dis[u] + w
                    cnt[v] = cnt[u] + 1
                    if cnt[v] >= n:
                        return "YES", dis, cnt
                    # 如果相邻节点还没有在队列中，将它加入队列
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        # 不存在从起点出发的负环
        return "NO", dis, cnt

    @staticmethod
    def count_shortest_path(dct, mod=10 ** 9 + 7):
        # 无向无权图最短路计数

        n = len(dct)
        # 初始化距离
        dis = [inf for _ in range(n)]
        # 标识当前节点是否在栈中
        visit = [False] * n
        # 当前最小距离的路径边数
        cnt = [0] * n
        queue = deque([0])
        # 队列与起点初始化默认从 0 出发
        dis[0] = 0
        visit[0] = True
        cnt[0] = 1
        while queue:
            # 取出队列中的第一个节点
            u = queue.popleft()
            visit[u] = False
            # 更新当前节点的相邻节点的距离
            for v in dct[u]:
                w = dct[u][v]
                if dis[v] > dis[u] + 1:
                    dis[v] = dis[u] + 1
                    cnt[v] = w * cnt[u]  # 此处 w 为重合边数
                    cnt[v] %= mod
                    # 如果相邻节点还没有在队列中，将它加入队列
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
                elif dis[v] == dis[u] + 1:
                    cnt[v] += w * cnt[u]
                    cnt[v] %= mod
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        return cnt

    @staticmethod
    def negative_circle_mul(dct, src=0, initial=0) -> (str, List[float], List[int]):
        # 模板: 判断是否存在乘积大于1的环
        n = len(dct)
        # 初始化距离
        dis = [inf for _ in range(n)]
        # 标识当前节点是否在栈中
        visit = [False] * n
        # 当前最小距离的路径边数
        cnt = [0] * n
        # 求带负权的最短路距离与路径边数
        queue = deque([src])
        # 队列与起点初始化默认从 0 出发
        dis[src] = initial
        visit[src] = True

        while queue:
            # 取出队列中的第一个节点
            u = queue.popleft()
            visit[u] = False
            # 更新当前节点的相邻节点的距离
            for v in dct[u]:
                w = dct[u][v]
                if dis[v] > dis[u] * w:
                    dis[v] = dis[u] * w
                    cnt[v] = cnt[u] + 1
                    if cnt[v] >= n:
                        return "YES", dis, cnt
                    # 如果相邻节点还没有在队列中，将它加入队列
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        # 不存在从起点出发的负环
        return "NO", dis, cnt

    def differential_constraint(self, ineq: List[List[int]], n: int):
        # 模板：差分约束计算不等式组是否有解
        dct = [dict() for _ in range(n + 1)]
        for i in range(1, n + 1):  # 节点索引从 1 开始，添加 0 为虚拟根节点
            dct[0][i] = 0
        for a, b, c in ineq:  # a-b<=c
            w = dct[b].get(a, inf)  # 取较小值的约束
            w = w if w < c else c
            dct[b][a] = w
        ans, dis, _ = self.negative_circle(dct, 0, 0)
        return ans, dis

