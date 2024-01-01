"""
Algorithm：spfa|negative_weight|shortest_path|differential_constraint|number_of_shortest_path

Description：shortest_path_faster_algorithm|single_source|deque

====================================LeetCode====================================
2589（https://leetcode.cn/problems/minimum-time-to-complete-all-tasks/）differential_constraint|greedy|classical

=====================================LuoGu======================================
P3385（https://www.luogu.com.cn/problem/P3385）shortest_path|negative_circle
P1938（https://www.luogu.com.cn/problem/P1938）negative_circle|positive_circle|shortest_path|longest_path
P2136（https://www.luogu.com.cn/problem/P2136）negative_circle|shortest_path
P2648（https://www.luogu.com.cn/problem/P2648）positive_circle|longest_path|classical
P1993（https://www.luogu.com.cn/problem/P1993）differential_constraint|negative_circle
P5960（https://www.luogu.com.cn/problem/P5960）differential_constraint
P1260（https://www.luogu.com.cn/problem/P1260）differential_constraint
P1931（https://www.luogu.com.cn/problem/P1931）positive_circle|mul
P1986（https://www.luogu.com.cn/problem/P1986）differential_constraint
P2850（https://www.luogu.com.cn/problem/P2850）negative_circle|several_source|classical
P4878（https://www.luogu.com.cn/problem/P4878）diff_array|dijkstra|shortest_path
P5751（https://www.luogu.com.cn/problem/P5751）prefix_sum|differential_constraint
P5905（https://www.luogu.com.cn/problem/P5905）johnson_shortest_path|several_source|shortest_path

====================================AtCoder=====================================
ABC061D（https://atcoder.jp/contests/abc061/tasks/abc061_d）reverse_graph|positive_circle|longest_path
ABC137E（https://atcoder.jp/contests/abc137/tasks/abc137_e）spfa|positive_circle

====================================LeetCode====================================
differential_constraint（https://oi-wiki.org/graph/diff-constraints/）
"""
from collections import deque
from typing import List

from src.graph.dijkstra.template import Dijkstra
from src.graph.spfa.template import SPFA
from src.utils.fast_io import FastIO, ac_max
from src.utils.fast_io import inf


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p2648(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2648
        tag: positive_circle|longest_path|classical
        """
        d, p, c, f = ac.read_list_ints()
        dct = [[] for _ in range(c)]
        for _ in range(p):
            a, b = ac.read_list_ints()
            dct[a - 1].append((b - 1, d))
        for _ in range(f):
            j, k, t = ac.read_list_ints()
            dct[j - 1].append((k - 1, d - t))
        res = 0
        for s in range(c):
            ans, dis, _ = SPFA().positive_circle_edge(dct, s, d)
            if ans:
                ac.st("orz")
                return
            res = ac.max(res, max(dis))
        ac.st(res)
        return

    @staticmethod
    def lg_p2136(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2136
        tag: negative_circle|shortest_path
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            a, b, c = ac.read_list_ints()
            dct[a - 1].append((b - 1, -c))
        ans1, dis1, _ = SPFA().negative_circle_edge(dct, 0)
        if ans1:
            ac.st("Forever love")
            return
        ans2, dis2, _ = SPFA().negative_circle_edge(dct, n - 1)
        if ans2:
            ac.st("Forever love")
            return
        ac.st(ac.min(dis1[n - 1], dis2[0]))
        return

    @staticmethod
    def lg_p3385(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3385
        tag: shortest_path|negative_circle
        """
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            dct = [[] for _ in range(n)]
            for _ in range(m):
                u, v, w = ac.read_list_ints()
                dct[u - 1].append((v - 1, w))
                if w >= 0:
                    dct[v - 1].append((u - 1, w))
            ans, _, _ = SPFA().negative_circle_edge(dct)
            ac.st("YES" if ans else "NO")
        return

    @staticmethod
    def lg_p1938(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1938
        tag: negative_circle|positive_circle|shortest_path|longest_path
        """
        d, p, c, f, s = ac.read_list_ints()
        s -= 1
        dct = [[] for _ in range(c)]
        for _ in range(p):
            a, b = ac.read_list_ints()
            dct[a - 1].append((b - 1, d))
        for _ in range(f):
            j, k, t = ac.read_list_ints()
            dct[j - 1].append((k - 1, d - t))
        ans, dis, _ = SPFA().positive_circle_edge(dct, s, d)
        ac.st(-1 if ans else max(dis))
        return

    @staticmethod
    def lg_p1993(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1993
        tag: differential_constraint|negative_circle
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n + 1)]
        for i in range(1, n + 1):
            dct[0].append((i, 0))
        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                a, b, c = lst[1:]
                dct[a].append((b, -c))
            elif lst[0] == 2:
                a, b, c = lst[1:]
                dct[b].append((a, c))
            else:
                a, b = lst[1:]
                dct[a].append((b, 0))
                dct[b].append((a, 0))
        ans, _, _ = SPFA().negative_circle_edge(dct)
        ac.st("Yes" if not ans else "No")
        return

    @staticmethod
    def lg_p5960(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5960
        tag: differential_constraint
        """

        n, m = ac.read_list_ints()
        edge = [[] for _ in range(n + 1)]
        for _ in range(m):
            a, b, c = ac.read_list_ints()
            edge[b].append((a, c))
        for i in range(1, n + 1):
            edge[0].append((i, 0))
        ans, dis, _ = SPFA().negative_circle_edge(edge, 0, 0)
        if ans:
            ac.st("NO")
        else:
            ac.lst(dis[1:])
        return

    @staticmethod
    def lg_p1260(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1260
        tag: differential_constraint
        """
        n, m = ac.read_list_ints()
        edge = [[] for _ in range(n + 1)]
        for _ in range(m):
            a, b, c = ac.read_list_ints()
            edge[b].append((a, c))
        for i in range(1, n + 1):
            edge[0].append((i, 0))
        ans, dis, _ = SPFA().negative_circle_edge(edge, 0, 0)
        if ans:
            ac.st("NO SOLUTION")
        else:
            low = min(dis[1:])
            for x in dis[1:]:
                ac.st(x - low)
        return

    @staticmethod
    def lg_p1931(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1931
        tag: positive_circle|mul
        """
        case = 0
        while True:
            n = ac.read_int()
            if not n:
                break
            case += 1
            name = [ac.read_str() for _ in range(n)]
            dct = [[] for _ in range(n)]
            ind = {na: i for i, na in enumerate(name)}
            for _ in range(ac.read_int()):
                a, c, b = ac.read_list_strs()
                dct[ind[a]].append((ind[b], float(c)))
            ans = "No"
            for i in range(n):
                flag, _, _ = SPFA().positive_circle_mul(dct, i, 1)
                if flag:
                    ans = "Yes"
                    break
            ac.st(f"Case {case}: {ans}")
            ac.read_str()
        return

    @staticmethod
    def lg_p1986(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1986
        tag: differential_constraint|minimum|longest_path|prefix_sum|classical|can_not_be_dijkstra
        """

        n, m = ac.read_list_ints()
        edge = [[] for _ in range(n + 1)]
        for _ in range(m):
            # x2 - x1 >= w is edge[x1].append((x2, w))
            a, b, c = ac.read_list_ints()
            if a > b:
                a, b = b, a
            edge[a - 1].append((b, c))

        for i in range(1, n + 1):
            # xi - 0 >= 0
            edge[0].append((i, 0))
            if i > 1:
                # (i) - (i-1) >= 0
                edge[i - 1].append((i, 0))
                # (i-1) - (i) >= -1
                edge[i].append((i - 1, -1))
        ans, dis, _ = SPFA().positive_circle_edge(edge, 0)
        ac.st(dis[n])
        return

    @staticmethod
    def abc_61d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc061/tasks/abc061_d
        tag: reverse_graph|positive_circle|longest_path|classical|reachable
        """
        n, m = ac.read_list_ints()
        edges = [ac.read_list_ints() for _ in range(m)]
        rev = [[] for _ in range(n)]
        for a, b, c in edges:
            a -= 1
            b -= 1
            rev[b].append(a)

        reach = [0] * n  # important
        stack = [n - 1]
        reach[-1] = 1
        while stack:
            i = stack.pop()
            for j in rev[i]:
                if not reach[j]:
                    reach[j] = 1
                    stack.append(j)
        dct = [[] for _ in range(n)]
        for a, b, c in edges:
            a -= 1
            b -= 1
            if reach[a] and reach[b]:
                dct[a].append((b, c))

        ans, dis, _ = SPFA().positive_circle_edge(dct, 0, 0)
        ac.st("inf" if ans else dis[n - 1])
        return

    @staticmethod
    def abc_137e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc137/tasks/abc137_e
        tag: spfa|positive_circle
        """
        # inf = 1 << 64
        n, m, p = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        rev = [[] for _ in range(n)]
        edges = [ac.read_list_ints() for _ in range(m)]
        for a, b, c in edges:
            dct[a - 1].append((b - 1, c - p))
            rev[b - 1].append((a - 1, c - p))

        visit = [0] * n
        stack = [n - 1]
        visit[n - 1] = 1
        while stack:
            i = stack.pop()
            for j, _ in rev[i]:
                if not visit[j]:
                    visit[j] = 1
                    stack.append(j)
        for i in range(n):
            dct[i] = [(a, b) for a, b in dct[i] if visit[a]]

        res, dis, cnt = SPFA().positive_circle_edge(dct, 0, 0)
        if res or dis[-1] == -inf:
            ac.st(-1)
            return
        ac.st(ac.max(dis[-1], 0))
        return

    @staticmethod
    def lc_2589(tasks: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-time-to-complete-all-tasks/
        tag: differential_constraint|greedy|classical|minimum|longest_path
        """

        n = max(ac_max(a, b) for a, b, _ in tasks)
        edge = [[] for _ in range(n + 1)]
        for a, b, c in tasks:
            if a > b:
                a, b = b, a
            edge[a - 1].append((b, c))

        for i in range(1, n + 1):
            edge[0].append((i, 0))
            if i > 1:
                edge[i - 1].append((i, 0))
                edge[i].append((i - 1, -1))
        ans, dis, _ = SPFA().positive_circle_edge(edge, 0, 0)
        return dis[n]

    @staticmethod
    def lg_p2850(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2850
        tag: negative_circle|several_source|classical
        """
        for _ in range(ac.read_int()):
            n, m, w = ac.read_list_ints()
            dct = [[] for _ in range(n + 1)]
            for _ in range(m):
                x, y, p = ac.read_list_ints()
                dct[x].append((y, p))
                dct[y].append((x, p))
            for _ in range(w):
                x, y, p = ac.read_list_ints()
                dct[x].append((y, -p))
            for i in range(1, n + 1):
                dct[0].append((i, 0))

            ans, _, _ = SPFA().negative_circle_edge(dct, 0, 0)
            ac.st("YES" if ans else "NO")
        return

    @staticmethod
    def lg_p4878(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4878
        tag: diff_array|dijkstra|shortest_path
        """
        n, ml, md = ac.read_list_ints()
        edge = [[] for _ in range(n + 1)]
        for _ in range(ml):
            # x1 - x2 <= w is edge[x2].append((x1, w))
            a, b, d = ac.read_list_ints()
            edge[a].append((b, d))

        for _ in range(md):
            a, b, d = ac.read_list_ints()
            edge[b].append((a, -d))

        for i in range(1, n + 1):
            # xi <= 0 is edge[0].append((i, 0))
            edge[0].append((i, 0))  # super source for solution check
            if i > 1:
                edge[i].append((i - 1, 0))

        ans, dis, _ = SPFA().negative_circle_edge(edge, 0, 0)
        if ans:
            ac.st(-1)
        else:
            ans, dis, _ = SPFA().negative_circle_edge(edge, 1, 0)
            if ans:
                ac.st(-1)
            elif dis[n] == inf:
                ac.st(-2)
            else:
                ac.st(dis[n])
        return

    @staticmethod
    def lg_p5905(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5905
        tag: johnson_shortest_path|several_source|shortest_path|classical
        """
        n, m = ac.read_list_ints()
        dct = [dict() for _ in range(n + 1)]
        for _ in range(m):
            u, v, w = ac.read_list_ints()
            if u != v:
                dct[u][v] = ac.min(dct[u].get(v, inf), w)
        dct = [[(x, d[x]) for x in d] for d in dct]
        for i in range(1, n + 1):
            dct[0].append([i, 0])
        flag, h, _ = SPFA().negative_circle_edge(dct)
        if flag:
            ac.st(-1)
            return

        for i in range(n + 1):
            dct[i] = [(j, w + h[i] - h[j]) for j, w in dct[i]]

        ceil = 10 ** 9
        dj = Dijkstra()
        for i in range(1, n + 1):
            ans = 0
            dis = dj.get_shortest_path(dct, i)
            for j in range(1, n + 1):
                ans += j * (dis[j] + h[j] - h[i]) if dis[j] < inf else j * ceil
            ac.st(ans)
        return

    @staticmethod
    def lg_p5751(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5751
        tag: prefix_sum|differential_constraint|negative_circle_edge|maximum|shortest_path
        """
        n, a0, b0, l0, a1, b1, l1 = ac.read_list_ints()
        # maximum is shortest_path
        edge = [[] for _ in range(n + 1)]  # node is the prefix sum
        for i in range(n):
            # a - b <= c is edge[b].append((a, c))
            if i - l0 + 1 >= 0:
                edge[i - l0 + 1].append((i + 1, l0 - a0))
                edge[i + 1].append((i - l0 + 1, b0 - l0))
            if i - l1 + 1 >= 0:
                edge[i - l1 + 1].append((i + 1, b1))
                edge[i + 1].append((i - l1 + 1, -a1))

        for i in range(1, n + 1):
            # xi - x0 >= 0 is edge edge[i].append((0, 0))
            edge[i].append((0, 0))
            if i > 1:
                edge[i].append((i - 1, 0))
                edge[i - 1].append((i, 1))

        ans, dis, _ = SPFA().negative_circle_edge(edge, 0, 0)
        if ans:
            ac.st(-1)
            return

        ac.st(dis[n])
        return
