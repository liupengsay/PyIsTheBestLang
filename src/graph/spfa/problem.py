"""
Algorithm：SPFA路径边数优先的广度优先搜索（可以带负权值）也可以最短路、差分约束、最短路条数

Function：SPFA（Shortest Path Faster Algorithm）是一种用于单源最短路径的算法。它通过队列和松弛操作来不断更新路径长度，从而更快地找到最短路径。

下面是一个简单的 Python SPFA 模板，其中 graph 是图的邻接表表示，iflytek_ads 是源节点，dist 是各节点到源节点的最短距离，prev 是各节点的前驱节点。
上面的代码只是一个简单的 SPFA 模板，实际时可能需要添|更多的特判和优化。例如，SPFA 算法在某些情况下容易陷入死循环，因此需要添|防止死循环的机制。此外，SPFA 算法的时间复杂度与输入图

的稠密程度有关，因此可能需要一些优化方法来提高它的效率。

Function：SPFA 算法是一种简单易用的最短路径算法，它通过队列和松弛操作来快速求解单源最短路径问题。它的时间复杂度与输入图的稠密程度有关，并且容易陷入死循环，因此需要注意这些问题。
Dijkstra：路径权值优先的深度优先搜索（只适用正权值）

====================================LeetCode====================================
2589（https://leetcode.com/problems/minimum-time-to-complete-all-tasks/）差分约束模板题，也可用greedy求解

=====================================LuoGu======================================
3385（https://www.luogu.com.cn/problem/P3385）通过最短路径更新的边数来从起点出发是否存在负环
1938（https://www.luogu.com.cn/problem/P1938）负环判断正环，以及最短路求最长路即最大正权路径值
2136（https://www.luogu.com.cn/problem/P2136）可能有负权环的最短距离
2648（https://www.luogu.com.cn/problem/P2648）判断是否存在正权环以及最长路
1144（https://www.luogu.com.cn/problem/P1144）无向无权图最短路counter
1993（https://www.luogu.com.cn/problem/P1993）差分约束判断是否存在负环
5960（https://www.luogu.com.cn/problem/P5960）差分约束模板题
1260（https://www.luogu.com.cn/problem/P1260）差分约束模板题
1931（https://www.luogu.com.cn/problem/P1931）判断乘积是否有大于1的环
1986（https://www.luogu.com.cn/problem/P1986）差分约束求解区间和
2850（https://www.luogu.com.cn/problem/P2850）从任意起点出发是否存在负环
4878（https://www.luogu.com.cn/problem/P4878）diff_array|与Dijkstra最短路
5751（https://www.luogu.com.cn/problem/P5751）prefix_sum转换为差分约束求解，并最大值
5905（https://www.luogu.com.cn/problem/P5905）有向带权图可能有负权 Johnson 全源最短路所有点对的最短路

====================================AtCoder=====================================
D - Score Attack （https://atcoder.jp/contests/abc061/tasks/abc061_d）反向建图后判断是否有正环并最长路
E - Coins Respawn（https://atcoder.jp/contests/abc137/tasks/abc137_e） SPFA 与 BFS 判断是否存在起点到终点的正权环

====================================LeetCode====================================
差分约束（https://oi-wiki.org/graph/diff-constraints/）
"""
from collections import deque
from math import inf
from typing import List

from src.graph.dijkstra.template import Dijkstra
from src.graph.spfa.template import SPFA
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1144(ac=FastIO()):
        # 无向无权图起点出发的最短路counter问题
        n, m = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            if x != y:
                dct[y][x] = dct[x][y] = dct[x].get(y, 0) + 1

        cnt = SPFA().count_shortest_path(dct, mod=100003)
        for a in cnt:
            ac.st(a)
        return

    @staticmethod
    def lg_p3648(ac=FastIO()):
        # 判断不同起点出发是否存在正环并最长路
        d, p, c, f = ac.read_list_ints()
        dct = [dict() for _ in range(c)]
        for _ in range(p):
            a, b = ac.read_list_ints_minus_one()
            # 直接权值取负数变为判断是否存在负环与最短路
            dct[a][b] = -d
        for _ in range(f):
            j, k, t = ac.read_list_ints()
            j -= 1
            k -= 1
            dct[j][k] = -(d - t)
        res = -inf
        for s in range(c):
            ans, dis, _ = SPFA().negative_circle(dct, s, -d)
            if ans == "YES":
                ac.st("orz")
                return
            res = ac.max(res, -min(dis))
        ac.st(res)
        return

    @staticmethod
    def lg_p2136(ac=FastIO()):
        # 判断不同起点出发是否存在负环并最短路
        n, m = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            a, b, c = ac.read_list_ints()
            a -= 1
            b -= 1
            dct[a][b] = ac.min(dct[a].get(a, inf), -c)
        ans1, dis1, _ = SPFA().negative_circle(dct, 0)
        ans2, dis2, _ = SPFA().negative_circle(dct, n - 1)
        ac.st("Forever love" if ans1 == "YES" or ans2 == "YES" else ac.min(dis1[n - 1], dis2[0]))
        return

    @staticmethod
    def lg_p3385(ac=FastIO()):
        # SPFA 判断是否存在负环与最短路
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            dct = [dict() for _ in range(n)]
            for _ in range(m):
                u, v, w = ac.read_list_ints()
                u -= 1
                v -= 1
                dct[u][v] = ac.min(dct[u].get(v, inf), w)
                if w >= 0:
                    dct[v][u] = ac.min(dct[v].get(u, inf), w)
            ans, _, _ = SPFA().negative_circle(dct)
            ac.st(ans)
        return

    @staticmethod
    def lg_1938(ac=FastIO()):
        # SPFA 判断是否存在正环与最长路
        d, p, c, f, s = ac.read_list_ints()
        s -= 1
        dct = [dict() for _ in range(c)]
        for _ in range(p):
            a, b = ac.read_list_ints_minus_one()
            # 直接权值取负数变为判断是否存在负环与最短路
            dct[a][b] = -d
        for _ in range(f):
            j, k, t = ac.read_list_ints()
            j -= 1
            k -= 1
            dct[j][k] = -(d - t)
        ans, dis, _ = SPFA().negative_circle(dct, s, -d)
        ac.st(-1 if ans == "YES" else -min(dis))
        return

    @staticmethod
    def lg_p1993(ac=FastIO()):
        # 差分约束转换为负环判断求解
        n, m = ac.read_list_ints()
        dct = [dict() for _ in range(n + 1)]
        # 超级源点有向边出发
        for i in range(1, n + 1):
            dct[0][i] = 0
        for _ in range(m):
            lst = ac.read_list_ints()
            # xa - xb <= c 则增|有向边 [xb, xa, c] 其中 xb => xa
            if lst[0] == 1:
                a, b, c = lst[1:]
                dct[a][b] = -c
            elif lst[0] == 2:
                a, b, c = lst[1:]
                dct[b][a] = c
            else:
                a, b = lst[1:]
                dct[a][b] = 0
                dct[b][a] = 0
        ans, _, _ = SPFA().negative_circle(dct)
        if ans == "NO":
            ac.st("Yes")
        else:
            ac.st("No")
        return

    @staticmethod
    def lc_6318(tasks: List[List[int]]) -> int:
        # 差分约束转换为负环判断求解
        n = max(it[1] for it in tasks)
        # xa - xb <= c 则增|有向边 [xb, xa, c] 其中 xb => xa
        dct = [dict() for _ in range(n + 2)]

        # 邻项约束
        for i in range(1, n + 1):
            dct[i][i - 1] = 0
            dct[i - 1][i] = 1

        # 条件约束
        for s, e, c in tasks:
            if s - 1 not in dct[e]:
                dct[e][s - 1] = -c
            else:
                # 注意重边的约束
                k = dct[e][s - 1]
                dct[e][s - 1] = k if k < -c else -c

        # 超级源点
        for i in range(n + 1):
            dct[n + 1][i] = 0

        _, dis, _ = SPFA().negative_circle(dct, n + 1)
        return dis[n] - dis[0]

    @staticmethod
    def lg_p5960(ac=FastIO()):
        # 差分约束模板题
        n, m = ac.read_list_ints()
        edges = [ac.read_list_ints() for _ in range(m)]
        ans, dis = SPFA().differential_constraint(edges, n)
        if ans == "YES":
            ac.st("NO")
        else:
            ac.lst(dis[1:])
        return

    @staticmethod
    def lg_p1260(ac=FastIO()):
        # 差分约束模板题
        n, m = ac.read_list_ints()
        # edges里面索引从 1 开始
        edges = [ac.read_list_ints() for _ in range(m)]
        ans, dis = SPFA().differential_constraint(edges, n)
        if ans == "YES":
            ac.st("NO SOLUTION")
        else:
            res = dis[1:]
            floor = min(res)
            res = [num - floor for num in res]
            for a in res:
                ac.st(a)
        return

    @staticmethod
    def lg_p1931(ac=FastIO()):
        # 路径乘积是否有大于1的环
        case = 0
        while True:
            n = ac.read_int()
            if not n:
                break
            case += 1
            name = [ac.read_str() for _ in range(n)]
            dct = [dict() for _ in range(n)]
            ind = {na: i for i, na in enumerate(name)}
            for _ in range(ac.read_int()):
                a, c, b = ac.read_list_strs()
                dct[ind[a]][ind[b]] = float(c)
            ans = "No"
            for i in range(n):
                # 初始值为负，若存在乘积大于1的环，这样就能一直减小到非负数
                flag, _, _ = SPFA().negative_circle_mul(dct, i, -1)
                if flag == "YES":
                    ans = "Yes"
                    break
            ac.st(f"Case {case}: {ans}")
            ac.read_str()
        return

    @staticmethod
    def lg_p1986(ac=FastIO()):
        # 根据prefix_sum差分约束求解
        n, m = ac.read_list_ints()

        # 区间关系
        lst = []
        for _ in range(m):
            a, b, c = ac.read_list_ints()
            if a > b:
                a, b = b, a
            lst.append([a - 1, b, -c])
        # 邻居关系
        for i in range(1, n + 1):
            lst.append([i, i - 1, 1])
            lst.append([i - 1, i, 0])

        # 差分约束，注意索引从 1 开始
        lst = [[a + 1, b + 1, c] for a, b, c in lst]
        ans, dis = SPFA().differential_constraint(lst, n + 1)

        # 即为prefix_sum pre[n+1] - pre[1]
        ac.st(dis[n + 1] - dis[1])
        return

    @staticmethod
    def abc_61d(ac=FastIO()):
        # 反向建图后判断是否有正环并最长路
        n, m = ac.read_list_ints()
        edges = [ac.read_list_ints() for _ in range(m)]
        rev = [[] for _ in range(n)]
        for a, b, c in edges:
            a -= 1
            b -= 1
            rev[b].append(a)

        # 反向建图
        reach = [0] * n
        stack = [n - 1]
        reach[-1] = 1
        while stack:
            i = stack.pop()
            for j in rev[i]:
                if not reach[j]:
                    reach[j] = 1
                    stack.append(j)

        dct = [dict() for _ in range(n)]
        for a, b, c in edges:
            a -= 1
            b -= 1
            if reach[a] and reach[b]:
                dct[a][b] = -c

        # 正环与最长路
        res, dis, _ = SPFA().negative_circle(dct, 0, 0)
        if res == "YES":
            ac.st("inf")
        else:
            ac.st(-dis[n - 1])
        return

    @staticmethod
    def abc_137e(ac=FastIO()):
        #  SPFA 与 BFS 判断是否存在起点到终点的正权环
        n, m, p = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        rev = [[] for _ in range(n)]
        edges = []
        for _ in range(m):
            a, b, c = ac.read_list_ints()
            a -= 1
            b -= 1
            dct[a].append([b, p - c])
            rev[b].append([a, p - c])
            edges.append([a, b, p - c])
        # 首先判断可达性
        visit = [0] * n
        stack = [0]
        visit[0] = 1
        while stack:
            i = stack.pop()
            for j, _ in dct[i]:
                if not visit[j]:
                    visit[j] = 1
                    stack.append(j)
        if not visit[-1]:
            ac.st(-1)
            return
        # 过滤起点不能到达与不能到达终点的点
        visit2 = [0] * n
        stack = [n - 1]
        visit2[n - 1] = 1
        while stack:
            i = stack.pop()
            for j, _ in rev[i]:
                if not visit2[j]:
                    visit2[j] = 1
                    stack.append(j)
        for i in range(n):
            if not (visit[i] and visit2[i]):
                dct[i] = []
            dct[i] = [[a, b] for a, b in dct[i] if visit[a] and visit2[a]]
        # 判断此时起点到终点是否仍然存在环
        res, dis, cnt = SPFA().negative_circle_edge(dct, 0, 0)
        if res == "YES":
            ac.st(-1)
            return
        ans = ac.max(-dis[-1], 0)

        # 判断终点继续出发是否存在正权环
        dct = [[] for _ in range(n)]
        for a, b, c in edges:
            dct[a].append([b, c])
        res, dis, cnt = SPFA().negative_circle_edge(dct, n - 1, 0)
        if res == "YES":
            ac.st(-1)
            return

        ac.st(ans)
        return

    @staticmethod
    def lc_2589(tasks: List[List[int]]) -> int:
        # 根据prefix_sum差分约束求解
        lst = []
        for a, b, c in tasks:
            if a > b:
                a, b = b, a
            lst.append([a - 1, b, -c])

        n = 2000
        for i in range(1, n + 1):
            lst.append([i, i - 1, 1])
            lst.append([i - 1, i, 0])

        lst = [[a + 1, b + 1, c] for a, b, c in lst]
        ans, dis = SPFA().differential_constraint(lst, n + 1)
        return dis[n + 1] - dis[1]

    @staticmethod
    def lg_p2850(ac=FastIO()):
        for _ in range(ac.read_int()):
            # 从任意起点出发是否存在负环
            n, m, w = ac.read_list_ints()
            dct = [dict() for _ in range(n)]
            for _ in range(m):
                x, y, p = ac.read_list_ints()
                x -= 1
                y -= 1
                dct[x][y] = ac.min(dct[x].get(y, inf), p)
                dct[y][x] = ac.min(dct[y].get(x, inf), p)
            for _ in range(w):
                x, y, p = ac.read_list_ints()
                x -= 1
                y -= 1
                dct[x][y] = ac.min(dct[x].get(y, inf), -p)

            dis = [inf for _ in range(n)]
            visit = [False] * n

            def negative_circle():
                # 模板: 判断是否存在负环与求解最短路（正数取反即可判断是否存在正权环以及最长路）
                cnt = [0] * n
                # 求带负权的最短路距离与路径边数
                queue = deque([src])
                # 队列与起点初始化默认从 0 出发
                dis[src] = 0
                visit[src] = True

                while queue:
                    # 取出队列中的第一个节点
                    u = queue.popleft()
                    visit[u] = False
                    # 更新当前节点的相邻节点的距离
                    for v in dct[u]:
                        ww = dct[u][v]
                        if dis[v] > dis[u] + ww:
                            dis[v] = dis[u] + ww
                            cnt[v] = cnt[u] + 1
                            if cnt[v] >= n:
                                return "YES", dis, cnt
                            # 如果相邻节点还没有在队列中，将它|入队列
                            if not visit[v]:
                                queue.append(v)
                                visit[v] = True
                # 不存在从起点出发的负环
                return "NO", dis, cnt

            for src in range(n):
                if not visit[src]:
                    #  visit 记录已经过的节点
                    ans, _, _ = negative_circle()
                    if ans == "YES":
                        ac.st("YES")
                        break
            else:
                ac.st("NO")
        return

    @staticmethod
    def lg_p4878(ac=FastIO()):
        # diff_array|与Dijkstra最短路
        n, ml, md = ac.read_list_ints()
        edge = []
        for _ in range(ml):
            a, b, d = ac.read_list_ints()
            if a > b:
                a, b = b, a
            edge.append([b, a, d])
        for _ in range(md):
            a, b, d = ac.read_list_ints()
            if a > b:
                a, b = b, a
            edge.append([a, b, -d])
        for i in range(1, n):
            edge.append([i, i + 1, 0])
        # 首先diff_array|判环
        ans, dis = SPFA().differential_constraint(edge, n)
        if ans == "YES":
            ac.st(-1)
        else:
            # 其次最大解即最短路（求最小解则是最长路）
            dct = [dict() for _ in range(n)]
            for a, b, c in edge:  # a-b<=c
                a -= 1
                b -= 1
                w = dct[b].get(a, inf)
                w = w if w < c else c
                dct[b][a] = w
            # 由于是按照编号顺序因此最大解为 dis[n-1] = pos[0] - pos[n-1] 最小即 pos[n-1] - pos[0] 最大
            dis = Dijkstra().get_dijkstra_result(dct, 0)
            ac.st(dis[n - 1] if dis[n - 1] < inf else -2)
        return

    @staticmethod
    def lg_p5905(ac=FastIO()):
        # 有向带权图可能有负权 Johnson 全源最短路所有点对的最短路
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n + 1)]
        for _ in range(m):
            u, v, w = ac.read_list_ints()
            dct[u].append([v, w])
        for i in range(1, n + 1):
            dct[0].append([i, 0])
        # 首先 Bellman-Ford 的队列实现算法 SPFA 判断有没有负环
        flag, h, _ = SPFA().negative_circle_edge(dct)
        if flag == "YES":
            ac.st(-1)
            return
        # 其次建立新图brute_force起点跑 Dijkstra
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
                ans += j * (dis[j] + h[j] - h[i]) if dis[j] < inf else j * 10 ** 9
            ac.st(ans)
        return

    @staticmethod
    def lg_p5751(ac=FastIO()):
        # 转换为prefix_sum差分约束求解，并最短路求解最大值
        n, a0, b0, l0, a1, b1, l1 = ac.read_list_ints()

        # 区间关系
        lst = []
        for i in range(1, n + 1):
            if i - l0 >= 0:
                lst.append([i, i - l0, l0 - a0])
                lst.append([i - l0, i, b0 - l0])
            if i - l1 >= 0:
                lst.append([i, i - l1, b1])
                lst.append([i - l1, i, -a1])

        # 邻居关系
        for i in range(1, n + 1):
            lst.append([i, i - 1, 1])
            lst.append([i - 1, i, 0])

        # 差分约束，注意索引从 1 开始
        lst = [[a + 1, b + 1, c] for a, b, c in lst]
        ans, dis = SPFA().differential_constraint(lst, n + 1)
        if ans == "YES":
            ac.st(-1)
            return

        # 其次最大解即最短路（求最小解则是最长路）
        dct = [dict() for _ in range(n + 1)]
        for a, b, c in lst:  # a-b<=c
            a -= 1
            b -= 1
            w = dct[b].get(a, inf)
            w = w if w < c else c
            dct[b][a] = w

        # 最大解为 （dis[n] = pos[0] - pos[n] 最小）即 （pos[n] - pos[0] 最大）为 0 到 n 最短路
        # 最小解为 （dis[n] = pos[0] - pos[n] 最大）即 （pos[n] - pos[0] 最小）为 0 到 n 最长度
        dis = Dijkstra().get_dijkstra_result(dct, 0)
        ac.st(dis[n])
        return