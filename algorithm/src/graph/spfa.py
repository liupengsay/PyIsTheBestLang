import unittest
from collections import deque

from algorithm.src.fast_io import FastIO

"""
算法：SPFA：路径边数优先的广度优先搜索（可以使用带负权值）也可以计算最短路

功能：SPFA（Shortest Path Faster Algorithm）是一种用于计算单源最短路径的算法。它通过使用队列和松弛操作来不断更新路径长度，从而更快地找到最短路径。

下面是一个简单的 Python SPFA 模板，其中 graph 是图的邻接表表示，src 是源节点，dist 是各节点到源节点的最短距离，prev 是各节点的前驱节点。
上面的代码只是一个简单的 SPFA 模板，实际使用时可能需要添加更多的特判和优化。例如，SPFA 算法在某些情况下容易陷入死循环，因此需要添加防止死循环的机制。此外，SPFA 算法的时间复杂度与输入图

的稠密程度有关，因此可能需要使用一些优化方法来提高它的效率。

功能：SPFA 算法是一种简单易用的最短路径算法，它通过使用队列和松弛操作来快速求解单源最短路径问题。它的时间复杂度与输入图的稠密程度有关，并且容易陷入死循环，因此需要注意这些问题。
Dijkstra：路径权值优先的深度优先搜索（只适用正权值）

参考题目：

===================================洛谷===================================
P3385 负环（https://www.luogu.com.cn/problem/P3385）通过最短路径更新的边数来计算从起点出发是否存在负环
P1938 [USACO09NOV]Job Hunt S（https://www.luogu.com.cn/problem/P1938）使用负环判断正环，以及使用最短路求最长路即最大正权路径值
P2136 拉近距离（https://www.luogu.com.cn/problem/P2136）计算可能有负权环的最短距离
P2648 赚钱（https://www.luogu.com.cn/problem/P2648）判断是否存在正权环以及最长路

P1144 最短路计数（https://www.luogu.com.cn/problem/P1462）计算最短路的条数
"""


class SPFA:
    def __init__(self):
        return

    @staticmethod
    def negtive_circle(dct, src=0, initial=0):
        # 模板: 判断是否存在负环与求解最短路（正数取反即可判断是否存在正权环以及最长路）
        n = len(dct)
        # 初始化距离
        dis = [float("inf") for _ in range(n)]
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


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p3385(ac=FastIO()):
        # 模板：SPFA 判断是否存在负环与计算最短路
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            dct = [dict() for _ in range(n)]
            for _ in range(m):
                u, v, w = ac.read_ints()
                u -= 1
                v -= 1
                dct[u][v] = ac.min(dct[u].get(v, ac.inf), w)
                if w >= 0:
                    dct[v][u] = ac.min(dct[v].get(u, ac.inf), w)
            ans, _, _ = SPFA().negtive_circle(dct)
            ac.st(ans)
        return

    @staticmethod
    def lg_1938(ac=FastIO()):
        # 模板：SPFA 判断是否存在正环与计算最长路
        d, p, c, f, s = ac.read_ints()
        s -= 1
        dct = [dict() for _ in range(c)]
        for _ in range(p):
            a, b = ac.read_ints_minus_one()
            # 直接权值取负数变为判断是否存在负环与计算最短路
            dct[a][b] = -d
        for _ in range(f):
            j, k, t = ac.read_ints()
            j -= 1
            k -= 1
            dct[j][k] = -(d - t)
        ans, dis, _ = SPFA().negtive_circle(dct, s, -d)
        ac.st(-1 if ans == "YES" else -min(dis))
        return


class SPFACnt:
    def __init__(self):
        # 最短路计数
        return

    @staticmethod
    def gen_result(dct):
        n = len(dct)
        # 初始化距离
        dis = [float("inf") for _ in range(n)]
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
                    cnt[v] = w*cnt[u]  # 此处 w 为重合边数
                    # 如果相邻节点还没有在队列中，将它加入队列
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
                elif dis[v] == dis[u] + 1:
                    cnt[v] += w*cnt[u]
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        return cnt


class TestGeneral(unittest.TestCase):

    def test_spfa(self):
        dct = [{1: 5, 2: 1}, {3: 4}, {3: 2}, {}]
        spfa = SPFA()
        res, dis, cnt = spfa.negtive_circle(dct)
        assert res == "NO"
        assert dis == [0, 5, 1, 3]
        assert cnt == [0, 1, 1, 2]

        dct = [{1: 5, 2: 1}, {3: 4}, {3: 2}, {2: -4}]
        spfa = SPFA()
        res, _, _ = spfa.negtive_circle(dct)
        assert res == "YES"
        return

    def test_spfa_cnt(self):
        dct = [{1: 3, 2: 2}, {3: 4}, {3: 1}, {}]
        spfa = SPFACnt()
        assert spfa.gen_result(dct) == [1, 3, 2, 14]
        return


if __name__ == '__main__':
    unittest.main()
