"""

Algorithm：dinic_max_flow_min_cut|dinic_max_flow_min_cost|dinic_max_flow_max_cost|bipartite_matching
Description：dinic_max_flow_min_cut|dinic_max_flow_min_cost|dinic_max_flow_max_cost|bipartite_matching

====================================LeetCode====================================
1947（https://leetcode.cn/problems/maximum-compatibility-score-sum/）bipartite_graph|maximum_weight_match|state_compress
1066（https://leetcode.cn/problems/campus-bikes-ii/）bipartite_graph|minimum_weight_match|km
100401（https://leetcode.cn/problems/find-the-power-of-k-size-subarrays-ii/）max_flow_min_cost|classical
3276（https://leetcode.cn/problems/select-cells-in-grid-with-maximum-score/）dinic_max_flow_min_cost|state_dp|classical
1601（https://leetcode.cn/problems/maximum-number-of-achievable-transfer-requests/description/）fake_source|build_graph|brain_teaser

=====================================LuoGu======================================
P3376（https://www.luogu.com.cn/problem/P3376）dinic_max_flow
P1343（https://www.luogu.com.cn/problem/P1343）dinic_max_flow
P2740（https://www.luogu.com.cn/problem/P2740）dinic_max_flow
P1361（https://www.luogu.com.cn/problem/P1361）dinic_max_flow|min_cut
P2057（https://www.luogu.com.cn/problem/P2057）dinic_max_flow|min_cut
P1344（https://www.luogu.com.cn/problem/P1344）dinic_max_flow|min_cut
P1345（https://www.luogu.com.cn/problem/P1345）dinic_max_flow|min_cut
P2762（https://www.luogu.com.cn/problem/P2762）dinic_max_flow|min_cut|specific_plan
P3381（https://www.luogu.com.cn/problem/P3381）dinic_max_flow|min_cost
P4452（https://www.luogu.com.cn/problem/P4452）dinic_max_flow|min_cost
P2153（https://www.luogu.com.cn/problem/P2153）dinic_max_flow|min_cost
P2053（https://www.luogu.com.cn/problem/P2053）dinic_max_flow|min_cost
P2050（https://www.luogu.com.cn/problem/P2050）dinic_max_flow|min_cost
P4722（https://www.luogu.com.cn/problem/P4722）dinic_max_flow

===================================CodeForces===================================
2026E（https://codeforces.com/contest/2026/problem/E）max_flow_min_cut|network_flow|build_graph|classical
1082G（https://codeforces.com/problemset/problem/1082/G）max_flow_min_cut|brain_teaser|build_graph|classical

===================================AtCoder===================================
ABC247G（https://atcoder.jp/contests/abc247/tasks/abc247_g）max_flow|max_cost|dynamic_graph|brain_teaser|network_flow|classical
ABC241G（https://atcoder.jp/contests/abc241/tasks/abc241_g）network_flow|brain_teaser|brute_force|greedy|implemention|classical
ABC239E（https://atcoder.jp/contests/abc239/tasks/abc239_g）specific_plan|network_flow|max_flow|min_cut|greedy|implemention
ABC205F（https://atcoder.jp/contests/abc205/tasks/abc205_f）max_flow_min_cut|matrix|build_graph
ABC326G（https://atcoder.jp/contests/abc326/tasks/abc326_g）max_flow_min_cut|brain_teaser|build_graph

"""
import math
from collections import defaultdict
from typing import List

from src.graph.network_flow.template import DinicMaxflowMinCut, DinicMaxflowMinCost
from src.util.fast_io import FastIO


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
        flow = DinicMaxflowMinCut(n)
        graph = [defaultdict(int) for _ in range(n)]
        for _ in range(m):
            u, v, w = ac.read_list_ints()
            graph[u - 1][v - 1] += w
        for u in range(n):
            for v in graph[u]:
                flow.add_edge(u + 1, v + 1, graph[u][v])
        ans = flow.max_flow_min_cut(s, t)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1343(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1343
        tag: dinic_max_flow
        """
        n, m, x = ac.read_list_ints()
        s, t = 1, n
        flow = DinicMaxflowMinCut(n)
        graph = [defaultdict(int) for _ in range(n)]
        for _ in range(m):
            u, v, w = ac.read_list_ints()
            graph[u - 1][v - 1] += w
        for u in range(n):
            for v in graph[u]:
                flow.add_edge(u + 1, v + 1, graph[u][v])
        ans = flow.max_flow_min_cut(s, t)
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
        s, t = 1, n
        flow = DinicMaxflowMinCut(n)
        graph = [defaultdict(int) for _ in range(n)]
        for _ in range(m):
            u, v, w = ac.read_list_ints()
            graph[u - 1][v - 1] += w
        for u in range(n):
            for v in graph[u]:
                flow.add_edge(u + 1, v + 1, graph[u][v])
        ans = flow.max_flow_min_cut(s, t)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1361(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1361
        tag: dinic_max_flow|min_cut
        """
        n = ac.read_int()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        s, t = 1, n + 2
        m = ac.read_int()
        flow = DinicMaxflowMinCut(n + 2 + m * 2)
        for i in range(n):
            flow.add_edge(s, i + 2, a[i])
            flow.add_edge(i + 2, t, b[i])
        ans = sum(a) + sum(b)
        for i in range(m):
            nums = ac.read_list_ints()
            c1, c2 = nums[1], nums[2]
            ans += c1 + c2
            flow.add_edge(s, n + 1 + i * 2 + 1 + 1, c1)
            flow.add_edge(n + 1 + i * 2 + 2 + 1, t, c2)
            for j in nums[3:]:
                flow.add_edge(n + 1 + i * 2 + 1 + 1, j + 1, math.inf)
                flow.add_edge(j + 1, n + 1 + i * 2 + 2 + 1, math.inf)
        ac.st(ans - flow.max_flow_min_cut(s, t))
        return

    @staticmethod
    def lg_p2057(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2057
        tag: dinic_max_flow|min_cut
        """
        n, m = ac.read_list_ints()
        s, t = 1, n + 2
        a = ac.read_list_ints()
        flow = DinicMaxflowMinCut(n + 2)
        for i in range(n):
            if a[i]:
                flow.add_edge(s, i + 2, 1)
            else:
                flow.add_edge(i + 2, t, 1)

        for _ in range(m):
            x, y = ac.read_list_ints()
            flow.add_edge(x + 1, y + 1, 1)
            flow.add_edge(y + 1, x + 1, 1)
        ac.st(flow.max_flow_min_cut(s, t))
        return

    @staticmethod
    def lg_p1344(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1344
        tag: dinic_max_flow|min_cut
        """
        n, m = ac.read_list_ints()
        flow = DinicMaxflowMinCut(n)
        mod = 2023
        for _ in range(m):
            s, e, c = ac.read_list_ints()
            flow.add_edge(s, e, c * mod + 1)
        ans = flow.max_flow_min_cut(1, n)
        ac.lst([ans // mod, ans % mod])
        return

    @staticmethod
    def lg_p1345(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1345
        tag: dinic_max_flow|min_cut
        """
        n, m, s, t = ac.read_list_ints()
        flow = DinicMaxflowMinCut(n * 2)
        for i in range(1, n + 1):
            flow.add_edge(i * 2 - 1, i * 2, 1)
        for _ in range(m):
            x, y = ac.read_list_ints()
            flow.add_edge(x * 2, y * 2 - 1, math.inf)
            flow.add_edge(y * 2, x * 2 - 1, math.inf)
        ans = flow.max_flow_min_cut(2 * s, 2 * t - 1)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2762(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2762
        tag: dinic_max_flow|min_cut|specific_plan
        """
        m, n = ac.read_list_ints()
        flow = DinicMaxflowMinCut(m + n + 2)
        ans = 0
        for i in range(1, m + 1):
            nums = ac.read_list_ints()
            ans += nums[0]
            flow.add_edge(1, i + 1, nums[0])
            for j in nums[1:]:
                flow.add_edge(i + 1, m + 1 + j, math.inf)
        cost = ac.read_list_ints()
        for j in range(1, n + 1):
            flow.add_edge(m + j + 1, m + n + 2, cost[j - 1])
        ans -= flow.max_flow_min_cut(1, m + n + 2)
        ac.lst([i - 1 for i in range(2, m + 2) if flow.depth[i] != -1])
        ac.lst([j - m - 1 for j in range(m + 2, m + n + 2) if flow.depth[j] != -1])
        ac.st(ans)
        return

    @staticmethod
    def lg_p3381(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3381
        tag: dinic_max_flow|min_cost
        """
        n, m, s, t = ac.read_list_ints()
        flow = DinicMaxflowMinCost(n)
        for i in range(1, m + 1):
            u, v, w, c = ac.read_list_ints()
            flow.add_edge(u, v, w, c)
        ac.lst(flow.max_flow_min_cost(s, t))
        return

    @staticmethod
    def lg_p4452(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4452
        tag: dinic_max_flow|min_cost
        """
        n, m, k, end = ac.read_list_ints()
        tt = [ac.read_list_ints() for _ in range(n)]
        f = [ac.read_list_ints() for _ in range(n)]
        queries = [ac.read_list_ints() for _ in range(m)]
        source, target, original = m * 2 + 1, m * 2 + 2, m * 2 + 3
        flow = DinicMaxflowMinCost(m * 2 + 3)
        flow.add_edge(original, source, k, 0)
        for i in range(1, m + 1):
            a, b, s, t, c = queries[i - 1]
            flow.add_edge(2 * i - 1, 2 * i, 1, -c)
            if tt[0][a] <= s:
                flow.add_edge(source, 2 * i - 1, math.inf, f[0][a])
            if t + tt[b][0] <= end:
                flow.add_edge(2 * i, target, math.inf, f[b][0])
        for i in range(1, m + 1):
            a1, b1, s1, t1, c1 = queries[i - 1]
            for j in range(1, m + 1):
                a2, b2, s2, t2, c2 = queries[j - 1]
                if t1 + tt[b1][a2] <= s2:
                    flow.add_edge(2 * i, 2 * j - 1, math.inf, f[b1][a2])
        _, min_cost = flow.max_flow_min_cost(original, target)
        ac.st(-min_cost)
        return

    @staticmethod
    def lg_p2153(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2153
        tag: dinic_max_flow|min_cost
        """
        n, m = ac.read_list_ints()
        flow = DinicMaxflowMinCost(n * 2)
        for _ in range(m):
            a, b, c = ac.read_list_ints()
            if a == 1 and b == n:
                flow.add_edge(a * 2, b * 2 - 1, 1, c)
            else:
                flow.add_edge(a * 2, b * 2 - 1, math.inf, c)
        for i in range(2, n):
            flow.add_edge(i * 2 - 1, i * 2, 1, 0)
        ac.lst(flow.max_flow_min_cost(2, 2 * n - 1))
        return

    @staticmethod
    def lg_p2053(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2053
        tag: dinic_max_flow|min_cost
        """
        m, n = ac.read_list_ints()
        flow = DinicMaxflowMinCost(m * n + n + 2)
        for i in range(1, n + 1):
            flow.add_edge(m * n + n + 1, m * n + i, 1, 0)
            cost = ac.read_list_ints()
            for j in range(1, m + 1):
                for k in range(1, n + 1):
                    flow.add_edge(m * n + i, (j - 1) * n + k, 1, k * cost[j - 1])
        for j in range(1, m + 1):
            for k in range(1, n + 1):
                flow.add_edge((j - 1) * n + k, m * n + n + 2, 1, 0)

        max_flow, min_cost = flow.max_flow_min_cost(m * n + n + 1, m * n + n + 2)
        assert max_flow == n
        ac.st("%.2f" % (min_cost / n))
        return

    @staticmethod
    def lg_p2050(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2050
        tag: dinic_max_flow|min_cost
        """
        n, m = ac.read_list_ints()
        p = ac.read_list_ints()
        np = sum(p)
        flow = DinicMaxflowMinCost(m * np + n + 2)

        for i in range(1, n + 1):
            flow.add_edge(m * np + n + 1, m * np + i, p[i - 1], 0)
            cost = ac.read_list_ints()
            for j in range(1, m + 1):
                for k in range(1, np + 1):
                    flow.add_edge(m * np + i, (j - 1) * np + k, 1, k * cost[j - 1])
        for j in range(1, m + 1):
            for k in range(1, np + 1):
                flow.add_edge((j - 1) * np + k, m * np + n + 2, 1, 0)

        max_flow, min_cost = flow.max_flow_min_cost(m * np + n + 1, m * np + n + 2)
        assert max_flow == np
        ac.st(min_cost)
        return

    @staticmethod
    def lc_1947(students: List[List[int]], mentors: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-compatibility-score-sum/
        tag: bipartite_graph|maximum_weight_match|state_compress|max_flow_max_cost
        """
        m = len(students)
        flow = DinicMaxflowMinCost(2 * m + 2 * m + 2)
        for i in range(1, m + 1):
            flow.add_edge(2 * i - 1, 2 * i, 1, 0)
            flow.add_edge(2 * m + 2 * m + 1, 2 * i - 1, 1, 0)
            flow.add_edge(2 * m + 2 * i - 1, 2 * m + 2 * i, 1, 0)
            flow.add_edge(2 * m + 2 * i, 2 * m + 2 * m + 2, 1, 0)
        for i in range(m):
            for j in range(m):
                score = sum(x == y for x, y in zip(students[i], mentors[j]))
                flow.add_edge(2 * (i + 1), 2 * m + 2 * (j + 1) - 1, 1, -score)
        ans = flow.max_flow_min_cost(2 * m + 2 * m + 1, 2 * m + 2 * m + 2)
        return -ans[1]

    @staticmethod
    def lc_1066(workers: List[List[int]], bikes: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/campus-bikes-ii/
        tag: bipartite_graph|minimum_weight_match|km
        """
        m, n = len(workers), len(bikes)
        flow = DinicMaxflowMinCost(2 * m + 2 * n + 2)
        for i in range(1, m + 1):
            flow.add_edge(2 * i - 1, 2 * i, 1, 0)
            flow.add_edge(2 * m + 2 * n + 1, 2 * i - 1, 1, 0)
        for i in range(1, n + 1):
            flow.add_edge(2 * m + 2 * i - 1, 2 * m + 2 * i, 1, 0)
            flow.add_edge(2 * m + 2 * i, 2 * m + 2 * n + 2, 1, 0)
        for i in range(m):
            for j in range(n):
                score = abs(workers[i][0] - bikes[j][0]) + abs(workers[i][1] - bikes[j][1])
                flow.add_edge(2 * (i + 1), 2 * m + 2 * (j + 1) - 1, 1, score)
        ans = flow.max_flow_min_cost(2 * m + 2 * n + 1, 2 * m + 2 * n + 2)
        return ans[1]

    @staticmethod
    def abc_247g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc247/tasks/abc247_g
        tag: max_flow|max_cost|dynamic_graph|brain_teaser|network_flow|classical
        """
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        dct = defaultdict(int)
        for a, b, c in nums:
            dct[(a, b)] = max(dct[(a, b)], c)
        aa = {a for a, _, _ in nums}
        bb = {b for _, b, _ in nums}
        ans = []
        flow = DinicMaxflowMinCost(304)
        for a, b in dct:
            c = dct[(a, b)]
            flow.add_edge(a, b + 150, 1, -c)
        for a in aa:
            flow.add_edge(302, a, 1, 0)
        for b in bb:
            flow.add_edge(b + 150, 303, 1, 0)
        for k in range(1, n + 1):
            flow.add_edge(301, 302, 1, 0)
            flow.add_edge(303, 304, 1, 0)
            max_flow, min_cost = flow.max_flow_min_cost(301, 304)
            if max_flow < k:
                break
            assert max_flow == k
            ans.append(-min_cost)
        ac.st(len(ans))
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def abc_241g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc241/tasks/abc241_g
        tag: network_flow|brain_teaser|brute_force|greedy|implemention|classical
        """
        n, m = ac.read_list_ints()
        lose = [0] * (n + 1)
        visit = [[0] * (n + 1) for _ in range(n + 1)]
        for _ in range(m):
            ww, ll = ac.read_list_ints()
            lose[ll] += 1
            visit[ww][ll] = 1
            visit[ll][ww] = 0

        ans = []
        e = n * (n - 1) // 2
        for i in range(1, n + 1):
            ceil = n - 1 - lose[i]
            if ceil == 0:
                continue
            s = e + n + 1
            t = e + n + 2
            ind = 0
            flow = DinicMaxflowMinCut(e + n + 2)
            for a in range(1, n + 1):
                for b in range(a + 1, n + 1):
                    ind += 1
                    flow.add_edge(s, ind, 1)
                    if visit[a][b]:
                        flow.add_edge(ind, a + e, 1)
                        continue
                    if visit[b][a]:
                        flow.add_edge(ind, b + e, 1)
                        continue
                    flow.add_edge(ind, a + e, 1)
                    flow.add_edge(ind, b + e, 1)
                if a != i:
                    flow.add_edge(a + e, t, min(ceil - 1, n - 1 - lose[a]))
                else:
                    flow.add_edge(a + e, t, ceil)
            assert ind == e
            if flow.max_flow_min_cut(s, t) == e:
                ans.append(i)
        ac.lst(ans)
        return

    @staticmethod
    def abc_239e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc239/tasks/abc239_g
        tag: specific_plan|network_flow|max_flow|min_cut|greedy|implemention
        """
        n, m = ac.read_list_ints()
        s = 1
        t = n
        edges = [ac.read_list_ints() for _ in range(m)]
        c = ac.read_list_ints()
        c[0] = c[-1] = math.inf

        flow = DinicMaxflowMinCut(n * 2)
        for i in range(1, n + 1):
            flow.add_edge(i * 2 - 1, i * 2, c[i - 1])
        for x, y in edges:
            flow.add_edge(x * 2, y * 2 - 1, math.inf)
            flow.add_edge(y * 2, x * 2 - 1, math.inf)
        min_cut = flow.max_flow_min_cut(2 * s, 2 * t - 1)
        ac.st(min_cut)
        ans = [i for i in range(2, n) if flow.depth[i * 2 - 1] != -1 and flow.depth[i * 2] == -1]
        ac.st(len(ans))
        ac.lst(ans)
        return

    @staticmethod
    def lc_100401(board: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-value-sum-by-placing-three-rooks-ii/description/
        tag: max_flow_min_cost|classical
        """
        m, n = len(board), len(board[0])
        flow = DinicMaxflowMinCost(m + n + 4)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                flow.add_edge(i, m + j, 1, -board[i - 1][j - 1])
        for i in range(1, m + 1):
            flow.add_edge(m + n + 1, i, 1, 0)
        for j in range(1, n + 1):
            flow.add_edge(m + j, m + n + 2, 1, 0)

        flow.add_edge(m + n + 3, m + n + 1, 3, 0)
        flow.add_edge(m + n + 2, m + n + 4, 3, 0)
        ans = flow.max_flow_min_cost(m + n + 3, m + n + 4)
        return -ans[1]

    @staticmethod
    def abc_205f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc205/tasks/abc205_f
        tag: max_flow_min_cut|matrix|build_graph
        """
        m, n, k = ac.read_list_ints()
        flow = DinicMaxflowMinCut(m + n + k + k + 2)
        for j in range(1, k + 1):
            a, b, c, d = ac.read_list_ints()
            for i in range(a, c + 1):
                flow.add_edge(i, m + n + j, 1)
            for i in range(b, d + 1):
                flow.add_edge(m + n + j + k, m + i, 1)
            flow.add_edge(m + n + j, m + n + j + k, 1)
        for i in range(1, m + 1):
            flow.add_edge(m + n + k + k + 1, i, 1)
        for i in range(1, n + 1):
            flow.add_edge(m + i, m + n + k + k + 2, 1)
        ans = flow.max_flow_min_cut(m + n + k + k + 1, m + n + k + k + 2)
        ac.st(ans)
        return

    @staticmethod
    def lc_3276(grid: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/select-cells-in-grid-with-maximum-score/submissions/
        tag: dinic_max_flow_min_cost|state_dp|classical
        """
        m, n = len(grid), len(grid[0])
        flow = DinicMaxflowMinCost(m + 202)
        start = m + 201
        end = m + 202
        for i in range(m):
            flow.add_edge(start, i + 1, 1, 0)
        vals = set()
        edges = set()
        for i in range(m):
            for j in range(n):
                edges.add((i + 1, m + grid[i][j], 1, 0))
                vals.add(grid[i][j])
                edges.add((m + 100 + grid[i][j], end, 1, 0))
        for a, b, c, d in edges:
            flow.add_edge(a, b, c, d)
        for va in vals:
            flow.add_edge(m + va, m + va + 100, 1, -va)
        ans = flow.max_flow_min_cost(start, end)
        return -ans[1]

    @staticmethod
    def cf_2026e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/2026/problem/E
        tag: max_flow_min_cut|network_flow|build_graph|classical
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            s = n + 60 + 1
            t = n + 60 + 2
            flow = DinicMaxflowMinCut(t)
            for i in range(n):
                for j in range(60):
                    if (nums[i] >> j) & 1:
                        flow.add_edge(i + 1, n + j + 1, math.inf)
                flow.add_edge(s, i + 1, 1)
            for j in range(60):
                flow.add_edge(n + j + 1, t, 1)
            ans = n - flow.max_flow_min_cut(s, t)
            ac.st(ans)
        return

    @staticmethod
    def abc_326g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc326/tasks/abc326_g
        tag: max_flow_min_cut|brain_teaser|build_graph
        """
        n, m = ac.read_list_ints()
        c = ac.read_list_ints()
        a = ac.read_list_ints()
        s = 6 * n + m + 1
        t = 6 * n + m + 2
        graph = DinicMaxflowMinCut(t)
        for i in range(n):
            graph.add_edge(s, i * 6 + 6, math.inf)
            for j in range(6, 1, -1):
                graph.add_edge(i * 6 + j, i * 6 + j - 1, c[i] * (j - 2))
        for j in range(m):
            ll = ac.read_list_ints()
            for i in range(n):
                graph.add_edge(i * 6 + ll[i], 6 * n + j + 1, math.inf)
            graph.add_edge(6 * n + j + 1, t, a[j])
        ans = sum(a) - graph.max_flow_min_cut(s, t)
        ac.st(ans)
        return

    @staticmethod
    def cf_1082g(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1082/G
        tag: max_flow_min_cut|brain_teaser|build_graph|classical
        """
        n, m = ac.read_list_ints()
        a = ac.read_list_ints()
        s = n + m + 1
        t = n + m + 2
        graph = DinicMaxflowMinCut(t)
        for i in range(n):
            graph.add_edge(m + i + 1, t, a[i])
        ans = 0
        for j in range(m):
            u, v, w = ac.read_list_ints()
            graph.add_edge(s, j + 1, w)
            graph.add_edge(j + 1, u + m, w)
            graph.add_edge(j + 1, v + m, w)
            ans += w
        ans -= graph.max_flow_min_cut(s, t)
        ac.st(ans)
        return

    @staticmethod
    def lc_1601(n: int, requests: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-number-of-achievable-transfer-requests/description/
        tag: fake_source|build_graph|brain_teaser
        """
        s = n + 1
        t = n + 2
        graph = DinicMaxflowMinCost(t)
        degree = [0] * n
        for i, j in requests:
            graph.add_edge(i + 1, j + 1, 1, 1)
            degree[j] += 1
            degree[i] -= 1
        for i in range(n):
            if degree[i] < 0:
                graph.add_edge(s, i + 1, -degree[i], 0)
            elif degree[i] > 0:
                graph.add_edge(i + 1, t, degree[i], 0)
        ans = len(requests) - graph.max_flow_min_cost(s, t)[1]
        return ans
