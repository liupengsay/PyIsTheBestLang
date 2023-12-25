"""

Algorithm：dinic_max_flow
Description：dinic_max_flow

====================================LeetCode====================================

=====================================LuoGu======================================
P3376（https://www.luogu.com.cn/problem/P3376）dinic_max_flow
P1343（https://www.luogu.com.cn/problem/P1343）dinic_max_flow
P2740（https://www.luogu.com.cn/problem/P2740）dinic_max_flow

===================================CodeForces===================================


"""
import math
from collections import defaultdict

from src.graph.max_flow.template import DinicMaxflow
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p3376(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3376
        tag: dinic_max_flow
        """
        n, m, s, t = ac.read_list_ints()
        s -= 1
        t -= 1
        flow = DinicMaxflow(n)
        graph = [defaultdict(int) for _ in range(n)]
        for _ in range(m):
            u, v, w = ac.read_list_ints()
            graph[u - 1][v - 1] += w
        for u in range(n):
            for v in graph[u]:
                flow.add_edge(u, v, graph[u][v])
        ans = flow.max_flow(s, t)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1343(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1343
        tag: dinic_max_flow
        """
        n, m, x = ac.read_list_ints()
        s, t = 0, n - 1
        flow = DinicMaxflow(n)
        graph = [defaultdict(int) for _ in range(n)]
        for _ in range(m):
            u, v, w = ac.read_list_ints()
            graph[u - 1][v - 1] += w
        for u in range(n):
            for v in graph[u]:
                flow.add_edge(u, v, graph[u][v])
        ans = flow.max_flow(s, t)
        if ans < 1:
            ac.st("Orz Ni Jinan Saint Cow!")
        else:
            ac.lst([ans, math.ceil(x / ans)])
        return

    @staticmethod
    def lg_p2740(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2740
        tag: dinic_max_flow
        """
        m, n = ac.read_list_ints()
        s, t = 0, n - 1
        flow = DinicMaxflow(n)
        graph = [defaultdict(int) for _ in range(n)]
        for _ in range(m):
            u, v, w = ac.read_list_ints()
            graph[u - 1][v - 1] += w
        for u in range(n):
            for v in graph[u]:
                flow.add_edge(u, v, graph[u][v])
        ans = flow.max_flow(s, t)
        ac.st(ans)
        return
