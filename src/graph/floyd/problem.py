"""

Algorithm：floyd|several_source_shortest_path|undirected_graph|directed_graph|pos_weight|neg_weight|negative_circle|shortest_circle
Description：shortest_path|longest_path|necessary_point_on_shortest_path|necessary_point_on_longest_path|necessary_edge
specific_plan： floyd need dp[i][j] where pre[i][j] = k, and bellman-ford dijkstra need pre[v] = u

====================================LeetCode====================================
2642（https://leetcode.cn/problems/design-graph-with-shortest-path-calculator/）floyd|dynamic_graph|shortest_path
1462（https://leetcode.cn/problems/course-schedule-iv/）transitive_closure|floyd

=====================================LuoGu======================================
P1119（https://www.luogu.com.cn/problem/P1119）offline_query|floyd|dynamic_graph
P1476（https://www.luogu.com.cn/problem/P1476）floyd|longest_path|specific_plan
P3906（https://www.luogu.com.cn/problem/P3906）floyd|shortest_path|specific_plan

P2009（https://www.luogu.com.cn/problem/P2009）floyd|shortest_path
P2419（https://www.luogu.com.cn/problem/P2419）floyd|topological_sort
P2910（https://www.luogu.com.cn/problem/P2910）shortest_path|floyd
P6464（https://www.luogu.com.cn/problem/P6464）brute_force|floyd|dynamic_graph
P6175（https://www.luogu.com.cn/problem/P6175）floyd|brute_force|O(n^3)|bfs|dijkstra
B3611（https://www.luogu.com.cn/problem/B3611）transitive_closure|floyd
P1613（https://www.luogu.com.cn/problem/P1613）floyd|several_floyd|shortest_path
P8312（https://www.luogu.com.cn/problem/P8312）limited_floyd|shortest_path|several_floyd
P8794（https://www.luogu.com.cn/problem/P8794）binary_search|floyd


===================================CodeForces===================================
472D（https://codeforces.com/problemset/problem/472/D）floyd|construction|shortest_path
1205B（https://codeforces.com/problemset/problem/1205/B）data_range|floyd|undirected_shortest_circle
25C（https://codeforces.com/problemset/problem/25/C）floyd
543B（https://codeforces.com/problemset/problem/543/B）bfs|brute_force|observation|floyd|implemention


====================================AtCoder=====================================
ABC051D（https://atcoder.jp/contests/abc051/tasks/abc051_d）floyd|shortest_path|necessary_edge|classical
ARC083B（https://atcoder.jp/contests/abc074/tasks/arc083_b）shortest_path_spanning_tree|floyd|dynamic_graph
ABC143E（https://atcoder.jp/contests/abc143/tasks/abc143_e）floyd|build_graph|shortest_path|several_floyd
ABC286E（https://atcoder.jp/contests/abc286/tasks/abc286_e）floyd|classical
ABC243E（https://atcoder.jp/contests/abc243/tasks/abc243_e）get_cnt_of_shortest_path|undirected|dijkstra|floyd|classical
ABC208D（https://atcoder.jp/contests/abc208/tasks/abc208_d）floyd|shortest_path|classical
ABC369E（https://atcoder.jp/contests/abc369/tasks/abc369_e）floyd|permutation|brute_force
ABC375F（https://atcoder.jp/contests/abc375/tasks/abc375_f）floyd|add_undirected_edge
        
=====================================AcWing=====================================
4872（https://www.acwing.com/problem/content/submission/4875/）floyd|reverse_thinking|shortest_path|reverse_graph

"""
from collections import defaultdict
from heapq import heappop, heappush
from itertools import permutations
from typing import List

from src.basis.binary_search.template import BinarySearch
from src.graph.dijkstra.template import Dijkstra
from src.graph.floyd.template import Floyd, WeightedGraphForFloyd
from src.utils.fast_io import FastIO



class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1613(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1613
        tag: floyd|several_floyd|shortest_path
        """
        n, m = ac.read_list_ints()

        pre = [0] * n * n
        cur = [0] * n * n
        dis = [math.inf] * n * n
        for _ in range(m):
            u, v = ac.read_list_ints_minus_one()
            pre[u * n + v] = dis[u * n + v] = 1

        for x in range(1, 32):
            for k in range(n):
                for i in range(n):
                    if not pre[i * n + k]:
                        continue
                    for j in range(n):
                        if pre[i * n + k] and pre[k * n + j]:
                            cur[i * n + j] = dis[i * n + j] = 1

            for i in range(n * n):
                pre[i] = cur[i]
                cur[i] = 0

        for k in range(n):
            for i in range(n):
                if dis[i * n + k] == inf:
                    continue
                for j in range(n):
                    dis[i * n + j] = ac.min(dis[i * n + j], dis[i * n + k] + dis[k * n + j])
        ac.st(dis[n - 1])
        return

    @staticmethod
    def ac_4872(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4875/
        tag: floyd|reverse_thinking|shortest_path|reverse_graph
        """
        n = ac.read_int()
        dp = [ac.read_list_ints() for _ in range(n)]
        a = ac.read_list_ints_minus_one()
        node = []
        ans = []
        for ind in range(n - 1, -1, -1):
            x = a[ind]
            node.append(x)
            cur = 0
            for i in node:
                for j in node:
                    dp[i][x] = ac.min(dp[i][x], dp[i][j] + dp[j][x])
                    dp[x][i] = ac.min(dp[x][i], dp[x][j] + dp[j][i])

            for i in node:
                for j in node:
                    dp[i][j] = ac.min(dp[i][j], dp[i][x] + dp[x][j])
                    cur += dp[i][j]
            ans.append(cur)

        ac.lst(ans[::-1])
        return

    @staticmethod
    def lg_p1119(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1119
        tag: offline_query|floyd|dynamic_graph|undirected
        """
        n, m = ac.read_list_ints()
        repair = ac.read_list_ints()
        dis = [math.inf] * n * n
        for i in range(m):
            a, b, c = ac.read_list_ints()
            dis[a * n + b] = dis[b * n + a] = c
        for i in range(n):
            dis[i * n + i] = 0

        k = 0
        for _ in range(ac.read_int()):
            x, y, t = ac.read_list_ints()
            while k < n and repair[k] <= t:
                for a in range(n):
                    for b in range(a + 1, n):
                        dis[a * n + b] = dis[b * n + a] = ac.min(dis[a * n + k] + dis[k * n + b], dis[b * n + a])
                k += 1
            if dis[x * n + y] < inf and x < k and y < k:
                ac.st(dis[x * n + y])
            else:
                ac.st(-1)
        return

    @staticmethod
    def lg_p1476(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1476
        tag: floyd|longest_path|specific_plan|classical
        """
        n = ac.read_int() + 1
        m = ac.read_int()
        dp = [[-inf] * (n + 1) for _ in range(n + 1)]
        for _ in range(m):
            i, j, k = ac.read_list_ints()
            dp[i][j] = k
        for i in range(n + 1):
            dp[i][i] = 0
        for k in range(1, n + 1):
            for i in range(1, n + 1):
                if dp[i][k] == -inf:
                    continue
                for j in range(1, n + 1):
                    if dp[i][j] < dp[i][k] + dp[k][j]:
                        dp[i][j] = dp[i][k] + dp[k][j]

        length = dp[1][n]
        path = []
        for i in range(1, n + 1):
            if dp[1][i] + dp[i][n] == dp[1][n]:
                path.append(i)
        ac.st(length)
        ac.lst(path)
        return

    @staticmethod
    def lg_p3906(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3906
        tag: floyd|shortest_path|specific_plan|classical
        """
        n, m = ac.read_list_ints()
        dp = [math.inf] * n * n
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            dp[i * n + j] = dp[j * n + i] = 1
        for i in range(n):
            dp[i * n + i] = 0

        for k in range(n):
            for i in range(n):
                if dp[i * n + k] == inf:
                    continue
                for j in range(i + 1, n):
                    dp[j * n + i] = dp[i * n + j] = ac.min(dp[i * n + j], dp[i * n + k] + dp[k * n + j])

        for _ in range(ac.read_int()):
            u, v = ac.read_list_ints_minus_one()
            dis = dp[u * n + v]
            ac.lst([x + 1 for x in range(n) if dp[u * n + x] + dp[x * n + v] == dis])
        return

    @staticmethod
    def lg_b3611(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/B3611
        tag: transitive_closure|floyd
        """
        n = ac.read_int()
        dp = []
        for _ in range(n):
            dp.extend(ac.read_list_ints())
        for k in range(n):
            for i in range(n):
                if not dp[i * n + k]:
                    continue
                for j in range(n):
                    if dp[i * n + k] and dp[k * n + j]:
                        dp[i * n + j] = 1
        for i in range(n):
            ac.lst([dp[i * n + j] for j in range(n)])
        return

    @staticmethod
    def abc_51d_1(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc051/tasks/abc051_d
        tag: floyd|shortest_path|necessary_edge|classical|reverse_thinking
        """
        n, m = ac.read_list_ints()
        dp = [math.inf] * n * n
        for i in range(n):
            dp[i * n + i] = 0

        edges = [ac.read_list_ints() for _ in range(m)]
        for i, j, w in edges:
            i -= 1
            j -= 1
            dp[i * n + j] = dp[j * n + i] = w

        for k in range(n):
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = dp[i * n + j], dp[i * n + k] + dp[k * n + j]
                    dp[i * n + j] = dp[j * n + i] = a if a < b else b
        ans = 0
        for i, j, w in edges:
            i -= 1
            j -= 1
            if dp[i * n + j] < w:
                ans += 1
        ac.st(ans)
        return

    @staticmethod
    def abc_51d_2(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc051/tasks/abc051_d
        tag: floyd|shortest_path|necessary_edge|classical|reverse_thinking
        """
        n, m = ac.read_list_ints()
        edges = [ac.read_list_ints() for _ in range(m)]
        dct = [[] for _ in range(n)]
        for i, j, w in edges:
            i -= 1
            j -= 1
            dct[i].append([j, w])
            dct[j].append([i, w])
        dis = []
        for i in range(n):
            dis.append(Dijkstra().get_shortest_path(dct, i))
        ans = 0
        for i, j, w in edges:
            i -= 1
            j -= 1
            if dis[i][j] < w:
                ans += 1
        ac.st(ans)
        return

    @staticmethod
    def abc_74d(ac=FastIO()):
        # shortest_path生成图，Floyd维护最小生成图
        n = ac.read_int()
        grid = [ac.read_list_ints() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if grid[i][j] != grid[j][i]:
                    ac.st(-1)
                    return
            if grid[i][i]:
                ac.st(-1)
                return

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append([i, j, grid[i][j]])
        edges.sort(key=lambda it: it[2])
        ans = 0
        dis = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            dis[i][i] = 0
        # 逐渐更新最短距离
        for i, j, w in edges:
            if dis[i][j] < grid[i][j]:
                ac.st(-1)
                return
            if dis[i][j] == w:
                continue
            ans += w
            for x in range(n):
                for y in range(x + 1, n):
                    a, b = dis[x][y], dis[x][i] + w + dis[j][y]
                    a = a if a < b else b
                    b = dis[x][j] + w + dis[i][y]
                    a = a if a < b else b
                    dis[x][y] = dis[y][x] = a
        ac.st(ans)
        return

    @staticmethod
    def abc_143e(ac=FastIO()):
        # Floydbuild_graph|shortest_path，两种shortest_path，建两次图
        n, m, ll = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        dis = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            dis[i][i] = 0
        for _ in range(m):
            x, y, z = ac.read_list_ints()
            x -= 1
            y -= 1
            dct[x].append([y, z])
            dct[y].append([x, z])
            a, b = dis[x][y], z
            dis[x][y] = dis[y][x] = a if a < b else b

        for k in range(n):
            for i in range(n):
                for j in range(i + 1, n):
                    cur = dis[i][k] + dis[k][j]
                    if cur < dis[i][j]:
                        dis[i][j] = dis[j][i] = cur

        dp = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 0
            for j in range(i + 1, n):
                if dis[i][j] <= ll:
                    dp[i][j] = dp[j][i] = 0

        for k in range(n):
            for i in range(n):
                for j in range(i + 1, n):
                    cur = dp[i][k] + dp[k][j] + 1
                    if cur < dp[i][j]:
                        dp[i][j] = dp[j][i] = cur

        for _ in range(ac.read_int()):
            x, y = ac.read_list_ints_minus_one()
            ans = dp[x][y]
            ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def cf_472d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/472/D
        tag: floyd|construction|shortest_path
        """

        n = ac.read_int()
        grid = [ac.read_list_ints() for _ in range(n)]
        for i in range(n):
            if grid[i][i]:
                ac.no()
                return
            for j in range(i + 1, n):
                if grid[i][j] != grid[j][i] or not grid[i][j]:
                    ac.no()
                    return
        if n == 1:
            ac.yes()
            return
        for i in range(n):
            j = 1 if not i else 0
            for r in range(n):
                if grid[i][r] < grid[i][j] and i != r:
                    j = r
            for k in range(n):
                if abs(grid[i][k] - grid[j][k]) != grid[i][j]:
                    ac.no()
                    return
        ac.yes()
        return

    @staticmethod
    def lg_p1613(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1613
        tag: floyd|several_floyd|shortest_path
        """
        # 建立新图Floydshortest_path
        n, m = ac.read_list_ints()

        # 表示节点i与j之间距离为2^k的路径是否存在
        dp = [[[0] * 32 for _ in range(n)] for _ in range(n)]
        for _ in range(m):
            u, v = ac.read_list_ints_minus_one()
            dp[u][v][0] = 1

        # 结合倍增思想Floyd建新图
        for x in range(1, 32):
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        if dp[i][k][x - 1] and dp[k][j][x - 1]:
                            dp[i][j][x] = 1

        dis = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for x in range(32):
                    if dp[i][j][x]:
                        dis[i][j] = 1
                        break

        # Floyd新图shortest_path
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dis[i][j] = ac.min(dis[i][j], dis[i][k] + dis[k][j])
        ac.st(dis[0][n - 1])
        return

    @staticmethod
    def lg_p8312(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8312
        tag: limited_floyd|shortest_path|several_floyd
        """
        # 最多k条边的shortest_path跑k遍Floyd
        n, m = ac.read_list_ints()
        dis = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            dis[i][i] = 0

        for _ in range(m):
            a, b, c = ac.read_list_ints_minus_one()
            c += 1
            dis[a][b] = ac.min(dis[a][b], c)

        dct = [d[:] for d in dis]
        k, q = ac.read_list_ints()
        nums = [ac.read_list_ints_minus_one() for _ in range(q)]
        k = ac.min(k, n)
        for _ in range(k - 1):
            cur = [d[:] for d in dis]
            for p in range(n):
                for i in range(n):
                    for j in range(n):
                        cur[i][j] = ac.min(cur[i][j], dis[i][p] + dct[p][j])
            dis = [d[:] for d in cur]

        for c, d in nums:
            res = dis[c][d]
            ac.st(res if res < inf else -1)
        return

    @staticmethod
    def lg_p8794(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8794
        tag: binary_search|floyd
        """
        # binary_search|Floyd

        def get_dijkstra_result_mat(mat: List[List[int]], src: int) -> List[float]:
            # 模板: Dijkstra求shortest_path，变成负数求可以求最长路（还是正权值）
            len(mat)
            dis = [math.inf] * n
            stack = [[0, src]]
            dis[src] = 0
            visit = set(list(range(n)))
            while stack:
                d, ii = heappop(stack)
                if dis[ii] < d:
                    continue
                visit.discard(ii)
                for j in visit:
                    dj = mat[ii][j] + d
                    if dj < dis[j]:
                        dis[j] = dj
                        heappush(stack, [dj, j])
            return dis

        n, q = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(n)]
        lower = [ac.read_list_ints() for _ in range(n)]
        ans = 0
        for i in range(n):
            ans += sum(get_dijkstra_result_mat(lower, i))
        if ans > q:
            ac.st(-1)
            return

        ans = 0
        for i in range(n):
            ans += sum(get_dijkstra_result_mat(grid, i))
        if ans <= q:
            ac.st(0)
            return

        def check(x):
            cnt = [x // n] * n
            for y in range(x % n):
                cnt[y] += 1
            cur = [[0] * n for _ in range(n)]
            for a in range(n):
                for b in range(n):
                    cur[a][b] = ac.max(lower[a][b], grid[a][b] - cnt[a] - cnt[b])
            dis = 0
            for y in range(n):
                dis += sum(get_dijkstra_result_mat(cur, y))
            return dis <= q

        def check2(x):
            cnt = [x // n] * n
            for y in range(x % n):
                cnt[y] += 1
            cur = [[0] * n for _ in range(n)]
            for a in range(n):
                for b in range(n):
                    cur[a][b] = ac.max(lower[a][b], grid[a][b] - cnt[a] - cnt[b])
            # Floyd全源shortest_path
            for k in range(n):
                for a in range(n):
                    for b in range(a + 1, n):
                        cur[a][b] = cur[b][a] = ac.min(cur[a][b], cur[a][k] + cur[k][b])
            return sum(sum(c) for c in cur) <= q

        low = 1
        high = n * 10 ** 5
        BinarySearch().find_int_left(low, high, check)
        ans = BinarySearch().find_int_left(low, high, check2)
        ac.st(ans)
        return

    @staticmethod
    def lc_2642():

        class Graph:
            def __init__(self, n: int, edges: List[List[int]]):
                d = [[math.inf] * n for _ in range(n)]
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

        return Graph

    @staticmethod
    def abc_286e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc286/tasks/abc286_e
        tag: floyd|classical
        """
        n = ac.read_int()
        a = ac.read_list_ints()
        s = [ac.read_str() for _ in range(n)]

        dis = [math.inf] * n * n
        gain = [0] * n * n
        for i in range(n):
            dis[i * n + i] = 0
            gain[i * n + i] = a[i]

        for i in range(n):
            for j in range(n):
                if s[i][j] == "Y":
                    dis[i * n + j] = 1
                    gain[i * n + j] = a[i] + a[j]

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    cur = dis[i * n + k] + dis[k * n + j]
                    g = gain[i * n + k] + gain[k * n + j] - a[k]
                    if cur < dis[i * n + j] or (cur == dis[i * n + j] and g > gain[i * n + j]):
                        gain[i * n + j] = g
                        dis[i * n + j] = cur

        for _ in range(ac.read_int()):
            u, v = ac.read_list_ints_minus_one()
            if dis[u * n + v] == inf:
                ac.st("Impossible")
            else:
                ac.lst([dis[u * n + v], gain[u * n + v]])
        return

    @staticmethod
    def abc_243e_1(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc243/tasks/abc243_e
        tag: get_cnt_of_shortest_path|undirected|dijkstra|floyd|classical
        """
        n, m = ac.read_list_ints()
        edges = []
        dct = [[] for _ in range(n)]
        for _ in range(m):
            x, y, w = ac.read_list_ints_minus_one()
            edges.append((x, y, w + 1))
            dct[x].append((y, w + 1))
            dct[y].append((x, w + 1))
        dis = []
        cnt = []
        for i in range(n):
            cur_cnt, cur_dis = Dijkstra().get_cnt_of_shortest_path(dct, i)
            dis.append(cur_dis)
            cnt.append(cur_cnt)
        ans = sum(cnt[x][y] > 1 or dis[x][y] < w for x, y, w in edges)
        ac.st(ans)
        return

    @staticmethod
    def abc_243e_2(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc243/tasks/abc243_e
        tag: get_cnt_of_shortest_path|undirected|dijkstra|floyd|classical
        """
        n, m = ac.read_list_ints()
        edges = []
        for _ in range(m):
            x, y, w = ac.read_list_ints_minus_one()
            edges.append((x, y, w + 1))
        cnt, dis = Floyd().get_cnt_of_shortest_path(edges, n)
        ans = sum(cnt[x][y] > 1 or dis[x][y] < w for x, y, w in edges)
        ac.st(ans)
        return

    @staticmethod
    def abc_208d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc208/tasks/abc208_d
        tag: floyd|shortest_path|classical
        """
        n, m = ac.read_list_ints()
        dp = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 0
        for _ in range(m):
            i, j, c = ac.read_list_ints_minus_one()
            c += 1
            dp[i][j] = c
        ans = 0
        tot = sum(sum(x if x < inf else 0 for x in dp[i]) for i in range(n))
        for i in range(n):
            for a in range(n):
                if dp[a][i] < inf:
                    for b in range(n):
                        if dp[a][i] + dp[i][b] < dp[a][b]:
                            if dp[a][b] < inf:
                                tot -= dp[a][b]
                            dp[a][b] = dp[a][i] + dp[i][b]
                            tot += dp[a][b]
            ans += tot
        ac.st(ans)
        return

    @staticmethod
    def cf_1205b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1205/B
        tag: data_range|floyd|undirected_shortest_circle
        """
        ac.read_int()
        nums = ac.read_list_ints()
        nums = [num for num in nums if num]
        n = len(nums)
        if n <= 2:
            ac.st(-1)
            return
        dct = [[] for _ in range(64)]
        for i, num in enumerate(nums):
            for x in range(64):
                if (num >> x) & 1:
                    dct[x].append(i)
        if any(len(dct[w]) >= 3 for w in range(64)):
            ac.st(3)
            return

        assert n <= 200
        edge = [math.inf] * n * n
        dis = [math.inf] * n * n
        for x in range(64):
            if len(dct[x]) == 2:
                i, j = dct[x][0], dct[x][1]
                dis[i * n + j] = dis[j * n + i] = 1
                edge[i * n + j] = edge[j * n + i] = 1
        ans = inf
        for k in range(n):
            for i in range(n):
                if dis[i * n + k] == inf:
                    continue
                for j in range(i + 1, n):
                    ans = min(ans, dis[i * n + j] + edge[i * n + k] + edge[k * n + j])  # classical
            for i in range(n):
                if dis[i * n + k] == inf:
                    continue
                for j in range(i + 1, n):
                    dis[j * n + i] = dis[i * n + j] = min(dis[i * n + j], dis[i * n + k] + dis[k * n + j])
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def abc_369e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc369/tasks/abc369_e
        tag: floyd|permutation|brute_force
        """
        n, m = ac.read_list_ints()
        dp = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 0
        edges = []
        for _ in range(m):
            i, j, t = ac.read_list_ints_minus_one()
            t += 1
            dp[i][j] = dp[j][i] = min(dp[i][j], t)
            edges.append((i, j, t))

        for k in range(n):
            for i in range(n):
                for j in range(i, n):
                    dp[i][j] = dp[j][i] = min(dp[i][j], dp[i][k] + dp[k][j])

        for _ in range(ac.read_int()):
            k = ac.read_int()
            lst = ac.read_list_ints_minus_one()
            ans = inf
            cost = sum(edges[x][-1] for x in lst)

            for item in permutations(lst, k):
                pre = defaultdict(lambda: inf)
                pre[0] = cost
                for x in item:
                    cur = defaultdict(lambda: inf)
                    for p in pre:
                        cur[edges[x][0]] = min(cur[edges[x][0]], pre[p] + dp[p][edges[x][1]])
                        cur[edges[x][1]] = min(cur[edges[x][1]], pre[p] + dp[p][edges[x][0]])
                    pre = cur

                ans = min(ans, min(pre[p] + dp[p][-1] for p in pre))
            ac.st(ans)
        return

    @staticmethod
    def abc_375f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc375/tasks/abc375_f
        tag: floyd|add_undirected_edge
        """
        n, m, q = ac.read_list_ints()
        inf = 2 * 10 ** 15
        graph = WeightedGraphForFloyd(n, inf)
        edges = [ac.read_list_ints_minus_one() for _ in range(m)]
        visit = [1] * m
        queries = [ac.read_list_ints_minus_one() for _ in range(q)]
        for lst in queries:
            if lst[0] == 0:
                visit[lst[1]] = 0

        for x in range(m):
            if visit[x]:
                i, j, w = edges[x]
                w += 1
                graph.add_undirected_edge_initial(i, j, w)
        graph.initialize_undirected()

        res = []
        for x in range(q - 1, -1, -1):
            lst = queries[x]
            if lst[0] == 0:
                ind = lst[1]
                i, j, w = edges[ind]
                w += 1
                graph.add_undirected_edge(i, j, w)
            else:
                i, j = lst[1], lst[2]
                res.append(graph.dis[i * n + j])
        res.reverse()
        for x in res:
            ac.st(x if x < inf else -1)
        return
