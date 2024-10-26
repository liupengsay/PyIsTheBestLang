"""
Algorithm：bfs|deque_bfs|discretization_bfs|bound_bfs|coloring_method|odd_circle
Description：multi_source_bfs|bilateral_bfs|0-1bfs|bilateral_bfs|a-star|heuristic_search

====================================LeetCode====================================
1036（https://leetcode.cn/problems/escape-a-large-maze/）bound_bfs|discretization_bfs
2493（https://leetcode.cn/problems/divide-nodes-into-the-maximum-number-of-groups/）union_find|bfs|brute_force|specific_plan|coloring_method|bipartite_graph
2290（https://leetcode.cn/problems/minimum-obstacle-removal-to-reach-corner/）0-1bfs|deque_bfs
1368（https://leetcode.cn/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/）0-1bfs|deque_bfs
2258（https://leetcode.cn/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/）binary_search|bfs|implemention
2092（https://leetcode.cn/problems/find-all-people-with-secret/）bfs
2608（https://leetcode.cn/contest/biweekly-contest-101/problems/shortest-cycle-in-a-graph/）bfs|undirected_smallest_circle|brute_force|shortest_path
1197（https://leetcode.cn/problems/minimum-knight-moves/?envType=study-plan-v2&id=premium-algo-100）bilateral_bfs
1654（https://leetcode.cn/problems/minimum-jumps-to-reach-home/）bfs|implemention
1926（https://leetcode.cn/problems/nearest-exit-from-entrance-in-maze/）deque_bfs|in_place_hash
909（https://leetcode.cn/problems/snakes-and-ladders/）01-bfs|implemention
1210（https://leetcode.cn/problems/minimum-moves-to-reach-target-with-rotations/description/）01-bfs|implemention
1298（https://leetcode.cn/problems/maximum-candies-you-can-get-from-boxes/）bfs
928（https://leetcode.cn/problems/minimize-malware-spread-ii/description/）brute_force|bfs
994（https://leetcode.cn/problems/rotting-oranges/description/）deque_bfs|implemention

=====================================LuoGu======================================
P1144（https://www.luogu.com.cn/problem/P1144）number_of_shortest_path
P1747（https://www.luogu.com.cn/problem/P1747）bilateral_bfs|shortest_path
P5507（https://www.luogu.com.cn/problem/P5507）bilateral_bfs|shortest_path
P2040（https://www.luogu.com.cn/problem/P2040）bfs
P2335（https://www.luogu.com.cn/problem/P2335）bfs
P2385（https://www.luogu.com.cn/problem/P2385）bfs|shortest_path|shortest_path
P2630（https://www.luogu.com.cn/problem/P2630）bfs|implemention|lexicographical_order
P1332（https://www.luogu.com.cn/problem/P1332）bfs
P1330（https://www.luogu.com.cn/problem/P1330）bfs|level_wise|coloring_method|union_find|odd_circle
P1215（https://www.luogu.com.cn/problem/P1215）bfs|implemention|hash
P1037（https://www.luogu.com.cn/problem/P1037）bfs|implemention|brute_force
P2853（https://www.luogu.com.cn/problem/P2853）bfs|counter
P2881（https://www.luogu.com.cn/problem/P2881）bfs|complexity
P2895（https://www.luogu.com.cn/problem/P2895）bfs|implemention
P2960（https://www.luogu.com.cn/problem/P2960）bfs
P2298（https://www.luogu.com.cn/problem/P2298）bfs
P3139（https://www.luogu.com.cn/problem/P3139）bfs|memory_search
P3183（https://www.luogu.com.cn/problem/P3183）bfs|counter|number_of_path|dfs|dp
P4017（https://www.luogu.com.cn/problem/P4017）bfs|counter|number_of_path|dfs|dp
P3395（https://www.luogu.com.cn/problem/P3395）bfs|implemention
P3416（https://www.luogu.com.cn/problem/P3416）bfs|memory_search
P3916（https://www.luogu.com.cn/problem/P3916）reverse_thinking|reverse_graph|reverse_order
P3958（https://www.luogu.com.cn/problem/P3958）build_graph|bfs
P4328（https://www.luogu.com.cn/problem/P4328）bfs|implemention
P4961（https://www.luogu.com.cn/problem/P4961）brute_force|implemention|counter
P6207（https://www.luogu.com.cn/problem/P6207）bfs|shortest_path|specific_plan
P6582（https://www.luogu.com.cn/problem/P6582）bfs|comb|counter|fast_power
P7243（https://www.luogu.com.cn/problem/P7243）bfs|gcd
P3496（https://www.luogu.com.cn/problem/P3496）brain_teaser|bfs|coloring_method|level_wise
P1432（https://www.luogu.com.cn/problem/P1432）memory_search|bfs
P1379（https://www.luogu.com.cn/problem/P1379）bilateral_bfs
P5507（https://www.luogu.com.cn/problem/P5507）bilateral_bfs|a-star|heuristic_search
P5908（https://www.luogu.com.cn/problem/P5908）bfs
P1038（https://www.luogu.com.cn/problem/P1038）topological_sorting
P1126（https://www.luogu.com.cn/problem/P1126）bfs
P1213（https://www.luogu.com.cn/problem/P1213）state_compression|01-bfs
P1902（https://www.luogu.com.cn/problem/P1902）binary_search|bfs|in_place_hash
P2199（https://www.luogu.com.cn/problem/P2199）deque_bfs|01-bfs
P2226（https://www.luogu.com.cn/problem/P2226）bfs
P2296（https://www.luogu.com.cn/problem/P2296）reverse_graph|bfs
P2919（https://www.luogu.com.cn/problem/P2919）bfs
P2937（https://www.luogu.com.cn/problem/P2937）01-bfs|monotonic_queue
P3456（https://www.luogu.com.cn/problem/P3456）bfs
P3496（https://www.luogu.com.cn/problem/P3496）brain_teaser|bfs
P3818（https://www.luogu.com.cn/problem/P3818）01-bfs|deque_bfs
P3855（https://www.luogu.com.cn/problem/P3855）bfs|md_state
P3869（https://www.luogu.com.cn/problem/P3869）bfs|state_compression
P4554（https://www.luogu.com.cn/problem/P4554）classical|01-bfs|implemention
P4667（https://www.luogu.com.cn/problem/P4667）01-bfs|implemention
P5096（https://www.luogu.com.cn/problem/P5096）state_compression|bfs|implemention
P5099（https://www.luogu.com.cn/problem/P5099）01-bfs|implemention
P5195（https://www.luogu.com.cn/problem/P5195）bfs
P6131（https://www.luogu.com.cn/problem/P6131）bfs|union_find
P6909（https://www.luogu.com.cn/problem/P6909）preprocess|bfs
P8628（https://www.luogu.com.cn/problem/P8628）01-bfs
P8673（https://www.luogu.com.cn/problem/P8673）01-bfs|implemention
P8674（https://www.luogu.com.cn/problem/P8674）preprocess|build_graph|bfs|implemention
P9065（https://www.luogu.com.cn/problem/P9065）brain_teaser|bfs|brute_force
P1099（https://www.luogu.com.cn/problem/P1099）tree_diameter|two_pointers|brute_force|monotonic_queue
P1363（https://www.luogu.com.cn/problem/P1363）classical|brain_teaser|observation|bfs
P2130（https://www.luogu.com.cn/problem/P2130）bfs|data_range
P6909（https://www.luogu.com.cn/problem/P6909）01-bfs|preprocess|classical

===================================CodeForces===================================
1594D（https://codeforces.com/contest/1594/problem/D）build_graph|coloring_method|bfs|bipartite_graph
1272E（https://codeforces.com/problemset/problem/1272/E）reverse_graph|multi_source_bfs
1572A（https://codeforces.com/problemset/problem/1572/A）brain_teaser|build_graph|bfs|circle_judge|dag_dp|classical
1037D（https://codeforces.com/problemset/problem/1037/D）01-bfs|implemention|classical
1176E（https://codeforces.com/contest/1176/problem/E）bds|color_method|classical
1520G（https://codeforces.com/contest/1520/problem/G）brain_teaser|bfs|classical
1611E2（https://codeforces.com/contest/1611/problem/E2）brain_teaser|bfs|implemention|classical
1607F（https://codeforces.com/contest/1607/problem/F）classical|topological_sort
1593E（https://codeforces.com/contest/1593/problem/E）classical|topological_sort|undirected
1702E（https://codeforces.com/contest/1702/problem/E）color_method|odd_circle_check
1674G（https://codeforces.com/contest/1674/problem/G）classical|brain_teaser|dag_dp|topologic_sort
1790F（https://codeforces.com/contest/1790/problem/F）classical|data_range|limit_operation
1840F（https://codeforces.com/contest/1840/problem/F）bfs|classical
796D（https://codeforces.com/problemset/problem/796/D）bfs
1063B（https://codeforces.com/problemset/problem/1063/B）bfs|observation|classical
1344B（https://codeforces.com/contest/1344/problem/B）bfs|observation
877D（https://codeforces.com/problemset/problem/877/D）bfs|observation|brain_teaser|union_find
987D（https://codeforces.com/contest/987/problem/D）several_source|bfs|brute_force
82C（https://codeforces.com/problemset/problem/82/C）implemention|bfs
1093D（https://codeforces.com/problemset/problem/1093/D）bfs|color_method|classical
1349C（https://codeforces.com/problemset/problem/1349/C）bfs|observation|implemention
1214D（https://codeforces.com/problemset/problem/1214/D）bfs|greed|classical
1276B（https://codeforces.com/problemset/problem/1276/B）bfs|unweighted_graph|multiplication_method

====================================AtCoder=====================================
ARC090B（https://atcoder.jp/contests/abc087/tasks/arc090_b）bfs|differential_constraint|O(n^2)
ABC133E（https://atcoder.jp/contests/abc133/tasks/abc133_e）bfs|coloring_method|counter
ABC070D（https://atcoder.jp/contests/abc070/tasks/abc070_d）classical|lca|offline_lca
ABC336F（https://atcoder.jp/contests/abc336/tasks/abc336_f）bilateral_bfs|classical|matrix_rotate
ABC339B（https://atcoder.jp/contests/abc339/tasks/abc339_b）bfs|visit
ABC339D（https://atcoder.jp/contests/abc339/tasks/abc339_d）bfs
ABC335E（https://atcoder.jp/contests/abc335/tasks/abc335_e）bfs|union_find|linear_dp
ABC329E（https://atcoder.jp/contests/abc329/tasks/abc329_e）bfs|matrix_dp|classical
ABC327D（https://atcoder.jp/contests/abc327/tasks/abc327_d）bfs|color_method|classical
ABC320D（https://atcoder.jp/contests/abc320/tasks/abc320_d）bfs
ABC317E（https://atcoder.jp/contests/abc317/tasks/abc317_e）bfs
ABC315E（https://atcoder.jp/contests/abc315/tasks/abc315_e）bfs|dfs|classical
ABC315D（https://atcoder.jp/contests/abc315/tasks/abc315_d）bfs|classical|implemention
ABC311D（https://atcoder.jp/contests/abc311/tasks/abc311_d）bfs
ABC302F（https://atcoder.jp/contests/abc302/tasks/abc302_f）build_graph|bfs|brain_teaser
ABC289E（https://atcoder.jp/contests/abc289/tasks/abc289_e）bfs
ABC282D（https://atcoder.jp/contests/abc282/tasks/abc282_d）color_method|bipartite_graph|bfs|classical
ABC280F（https://atcoder.jp/contests/abc280/tasks/abc280_f）bfs|negative_circle|positive_circle|brain_teaser|classical
ABC277F（https://atcoder.jp/contests/abc277/tasks/abc277_e）bfs
ABC277C（https://atcoder.jp/contests/abc277/tasks/abc277_c）bfs
ABC276E（https://atcoder.jp/contests/abc276/tasks/abc276_e）bfs
ABC244F（https://atcoder.jp/contests/abc244/tasks/abc244_f）bfs|bit_operation|brain_teaser
ABC246E（https://atcoder.jp/contests/abc246/tasks/abc246_e）bfs|union_find|brain_teaser|prune|classical
ABC241F（https://atcoder.jp/contests/abc241/tasks/abc241_f）bfs|implemention
ABC226C（https://atcoder.jp/contests/abc226/tasks/abc226_c）reverse_graph|bfs
ABC224D（https://atcoder.jp/contests/abc224/tasks/abc224_d）bfs|classical
ABC218F（https://atcoder.jp/contests/abc218/tasks/abc218_f）shortest_path|bfs|brute_force|brain_teaser
ABC216D（https://atcoder.jp/contests/abc216/tasks/abc216_d）topological_sort
ABC211E（https://atcoder.jp/contests/abc211/tasks/abc211_e）bfs|classical|not_dfs_back_trace
ABC209E（https://atcoder.jp/contests/abc209/tasks/abc209_e）build_graph|reverse_graph|brain_teaser|game_dp
ABC361D（https://atcoder.jp/contests/abc361/tasks/abc361_d）bfs|classical
ABC197F（https://atcoder.jp/contests/abc197/tasks/abc197_f）bfs|classical

=====================================AcWing=====================================
175（https://www.acwing.com/problem/content/175/）multi_source_bfs|classical
177（https://www.acwing.com/problem/content/177/）monotonic_queue|bfs
179（https://www.acwing.com/problem/content/179/）multi_source_bfs|bilateral_bfs
4418（https://www.acwing.com/problem/content/description/4418）bfs|coloring_method|odd_circle|specific_plan|counter
4484（https://www.acwing.com/problem/content/description/4484/）01-bfs

=====================================CodeChef=====================================
1（https://www.codechef.com/problems/PRISON）01-bfs


"""
import bisect
import math
from collections import deque, defaultdict
from heapq import heappush, heappop
from typing import List

from src.graph.dijkstra.template import WeightedGraphForDijkstra
from src.graph.union_find.template import UnionFind
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_2608_1(n: int, edges: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/contest/biweekly-contest-101/problems/shortest-cycle-in-a-graph/
        tag: bfs|undirected_smallest_circle|brute_force|shortest_path
        """
        graph = [[] for _ in range(n)]
        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)

        ans = math.inf
        for i in range(n):
            dist = [math.inf] * n
            par = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                x = q.popleft()
                for child in graph[x]:
                    if dist[x] > ans:
                        break
                    if dist[child] == math.inf:
                        dist[child] = 1 + dist[x]
                        par[child] = x
                        q.append(child)
                    elif par[x] != child and par[child] != x:
                        cur = dist[x] + dist[child] + 1
                        ans = ans if ans < cur else cur
        return ans if ans != math.inf else -1

    @staticmethod
    def lc_2608_2(n: int, edges: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/contest/biweekly-contest-101/problems/shortest-cycle-in-a-graph/
        tag: bfs|undirected_smallest_circle|brute_force|shortest_path
        """

        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        ans = math.inf
        for i in range(n):
            q = deque([(i, -1, 1)])
            visited = {(i, -1)}
            while q:
                u, parent, dist = q.popleft()
                if dist > ans:
                    break
                for v in graph[u]:
                    if v == parent:
                        continue
                    if v == i:
                        ans = ans if ans < dist else dist
                        break
                    if (v, u) not in visited:
                        visited.add((v, u))
                        q.append((v, u, dist + 1))
        return ans if ans < math.inf else -1

    @staticmethod
    def lc_2608_3(n: int, edges: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/contest/biweekly-contest-101/problems/shortest-cycle-in-a-graph/
        tag: bfs|undirected_smallest_circle|brute_force|shortest_path
        """

        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        def bfs(start: int) -> int:
            dis = [-1] * n
            dis[start] = 0
            q = deque([(start, -1)])
            res = math.inf
            while q:
                x, fa = q.popleft()
                for y in g[x]:
                    if dis[y] < 0:
                        dis[y] = dis[x] + 1
                        q.append((y, x))
                    elif y != fa:
                        res = res if res < dis[x] + dis[y] + 1 else dis[x] + dis[y] + 1
            return res

        ans = min(bfs(i) for i in range(n))
        return ans if ans < math.inf else -1

    @staticmethod
    def lc_2608_4(n: int, edges: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/contest/biweekly-contest-101/problems/shortest-cycle-in-a-graph/
        tag: bfs|undirected_smallest_circle|brute_force|shortest_path
        """

        graph = [set() for _ in range(n)]
        for x, y in edges:
            graph[x].add(y)
            graph[y].add(x)

        ans = math.inf
        for x, y in edges:
            graph[x].discard(y)
            graph[y].discard(x)
            dis = [math.inf] * n
            dis[x] = 0
            stack = deque([x])
            while stack:
                m = len(stack)
                for _ in range(m):
                    i = stack.popleft()
                    for j in graph[i]:
                        if dis[j] == math.inf:
                            dis[j] = dis[i] + 1
                            stack.append(j)
            ans = ans if ans < dis[y] else dis[y]
            graph[x].add(y)
            graph[y].add(x)
        return ans + 1 if ans < math.inf else -1

    @staticmethod
    def cf_1272e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1272/E
        tag: reverse_graph|multi_source_bfs
        """

        n = ac.read_int()
        nums = ac.read_list_ints()
        ans = [-1] * n

        edge = [[] for _ in range(n)]
        for i in range(n):
            for x in [i + nums[i], i - nums[i]]:
                if 0 <= x < n:
                    edge[x].append(i)

        for x in [0, 1]:
            stack = [i for i in range(n) if nums[i] % 2 == x]
            visit = set(stack)
            step = 1
            while stack:
                nex = []
                for i in stack:
                    for j in edge[i]:
                        if j not in visit:
                            ans[j] = step
                            nex.append(j)
                            visit.add(j)
                step += 1
                stack = nex
        ac.lst(ans)
        return

    @staticmethod
    def lg_p3183(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3183
        tag: bfs|counter|number_of_path|dfs|dag_dp
        """

        n, m = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        degree = [0] * n
        out_degree = [0] * n
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            edge[i].append(j)
            degree[j] += 1
            out_degree[i] += 1
        ind = [i for i in range(n) if degree[i] and not out_degree[i]]
        cnt = [0] * n
        stack = [i for i in range(n) if not degree[i]]
        for x in stack:
            cnt[x] = 1
        while stack:
            nex = []
            for i in stack:
                for j in edge[i]:
                    degree[j] -= 1
                    cnt[j] += cnt[i]
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        ans = sum(cnt[i] for i in ind)
        return ans

    @staticmethod
    def lg_p1747(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1747
        tag: bilateral_bfs|shortest_path
        """

        x0, y0 = ac.read_list_ints()
        x2, y2 = ac.read_list_ints()

        def check(x1, y1):
            if (x1, y1) == (1, 1):
                return 0

            visit1 = {(x1, y1): 0}
            visit2 = {(1, 1): 0}
            directions = [[1, 2], [1, -2], [-1, 2], [-1, -2],
                          [2, 1], [2, -1], [-2, 1], [-2, -1]]
            directions.extend([[2, 2], [2, -2], [-2, 2], [-2, -2]])
            stack1 = [[x1, y1]]
            stack2 = [[1, 1]]
            step = 1

            while True:
                nex1 = []
                for i, j in stack1:
                    for a, b in directions:
                        if 0 < i + a <= 20 and 0 < j + b <= 20 and (i + a, j + b) not in visit1:
                            visit1[(i + a, j + b)] = step
                            nex1.append([i + a, j + b])
                            if (i + a, j + b) in visit2:
                                return step + visit2[(i + a, j + b)]

                stack1 = nex1

                nex2 = []
                for i, j in stack2:
                    for a, b in directions:
                        if 0 < i + a <= 20 and 0 < j + b <= 20 and (i + a, j + b) not in visit2:
                            visit2[(i + a, j + b)] = step
                            nex2.append([i + a, j + b])
                            if (i + a, j + b) in visit1:
                                return step + visit1[(i + a, j + b)]

                stack2 = nex2
                step += 1

        ac.st(check(x0, y0))
        ac.st(check(x2, y2))
        return

    @staticmethod
    def lc_2290(grid: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-obstacle-removal-to-reach-corner/
        tag: 0-1bfs|deque_bfs|limited_shortest_path
        """
        m, n = len(grid), len(grid[0])
        visit = [[0] * n for _ in range(m)]
        q = deque([(0, 0, 0)])
        while q:
            d, x, y = q.popleft()
            for nx, ny in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
                if 0 <= nx < m and 0 <= ny < n and not visit[nx][ny]:
                    if [nx, ny] == [m - 1, n - 1]:
                        return d + grid[nx][ny]
                    visit[nx][ny] = 1
                    if not grid[nx][ny]:
                        q.appendleft((d, nx, ny))
                    else:
                        q.append((d + 1, nx, ny))
        return -1

    @staticmethod
    def lc_2493(n: int, edges: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/divide-nodes-into-the-maximum-number-of-groups/
        tag: union_find|bfs|brute_force|specific_plan|coloring_method|bipartite_graph
        """
        dct = [[] for _ in range(n)]
        uf = UnionFind(n)
        for i, j in edges:
            uf.union(i - 1, j - 1)
            dct[i - 1].append(j - 1)
            dct[j - 1].append(i - 1)

        group = uf.get_root_part()
        ans = 0
        for g in group:
            cur = -math.inf
            lst = group[g]
            edge = [[i - 1, j - 1] for i, j in edges if uf.find(i - 1) == uf.find(j - 1) == g]
            for i in lst:
                stack = [i]
                visit = {i: 1}
                deep = 1
                while stack:
                    nex = []
                    for x in stack:
                        for y in dct[x]:
                            if y not in visit:
                                visit[y] = visit[x] + 1
                                deep = visit[x] + 1
                                nex.append(y)
                    stack = nex[:]
                if all(abs(visit[x] - visit[y]) == 1 for x, y in edge):
                    if deep > cur:
                        cur = deep
            if cur == -math.inf:
                return -1
            ans += cur
        return ans

    @staticmethod
    def lc_1368(grid: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/
        tag: 0-1bfs|deque_bfs|limited_shortest_path
        """

        m, n = len(grid), len(grid[0])
        ceil = int(1e9)
        dist = [0] + [ceil] * (m * n - 1)
        seen = set()
        q = deque([(0, 0)])

        while q:
            x, y = q.popleft()
            if (x, y) in seen:
                continue
            seen.add((x, y))
            cur_pos = x * n + y
            for i, (nx, ny) in enumerate([(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)]):
                new_pos = nx * n + ny
                new_dis = dist[cur_pos] + (1 if grid[x][y] != i + 1 else 0)
                if 0 <= nx < m and 0 <= ny < n and new_dis < dist[new_pos]:  # important!!!
                    dist[new_pos] = new_dis  # O(mn)
                    if grid[x][y] == i + 1:
                        q.appendleft((nx, ny))
                    else:
                        q.append((nx, ny))
        return dist[m * n - 1]

    @staticmethod
    def lc_1926(maze: List[List[str]], entrance: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/nearest-exit-from-entrance-in-maze/
        tag: deque_bfs|in_place_hash
        """
        m, n = len(maze), len(maze[0])
        x0, y0 = entrance[:]
        stack = deque([[x0, y0, 0]])
        maze[x0][y0] = "+"
        while stack:
            i, j, d = stack.popleft()
            for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                if 0 <= x < m and 0 <= y < n and maze[x][y] == ".":
                    if x in [0, m - 1] or y in [0, n - 1]:
                        return d + 1
                    stack.append([x, y, d + 1])
                    maze[x][y] = "+"
        return -1

    @staticmethod
    def cf_1572a(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1572/A
        tag: brain_teaser|build_graph|bfs|circle_judge|dag_dp|classical|longest_path
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            dct = [dict() for _ in range(n)]
            degree = [0] * n
            for i in range(n):
                lst = ac.read_list_ints_minus_one()[1:]
                for j in lst:
                    dct[j][i] = 0 if i > j else 1
                degree[i] = len(lst)

            visit = [0] * n
            stack = [i for i in range(n) if not degree[i]]
            while stack:
                nex = []
                for i in stack:
                    for j in dct[i]:
                        degree[j] -= 1
                        if visit[i] + dct[i][j] > visit[j]:
                            visit[j] = visit[i] + dct[i][j]
                        if not degree[j]:
                            nex.append(j)
                stack = nex
            if max(degree) == 0:
                ac.st(max(visit) + 1)
            else:
                ac.st(-1)
        return

    @staticmethod
    def cf_1037d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1037/D
        tag: 01-bfs|implemention|classical
        """
        n = ac.read_int()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            edge[i].append(j)
            edge[j].append(i)

        dct = [set() for _ in range(n)]
        stack = [(0, -1)]
        parent = [-1] * n
        while stack:
            nex = []
            for i, fa in stack:
                for j in edge[i]:
                    if j != fa:
                        nex.append((j, i))
                        dct[i].add(j)
                        parent[j] = i
            stack = nex[:]

        nums = ac.read_list_ints_minus_one()
        stack = deque([{0}])
        for num in nums:
            if not stack or num not in stack[0]:
                ac.no()
                return
            stack[0].discard(num)
            if not stack[0]:
                stack.popleft()
            if dct[num]:
                stack.append(dct[num])
        ac.yes()
        return

    @staticmethod
    def lg_p1099(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1099
        tag: tree_diameter|two_pointers|brute_force|monotonic_queue
        """
        n, s = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(n - 1):
            i, j, w = ac.read_list_ints()
            dct[i - 1][j - 1] = w
            dct[j - 1][i - 1] = w

        def bfs_diameter(src):
            res, node = 0, src
            stack = [[src, 0]]
            parent = [-1] * n
            while stack:
                u, dis = stack.pop()
                if dis > res:
                    res = dis
                    node = u
                for v in dct[u]:
                    if v != parent[u]:
                        parent[v] = u
                        stack.append([v, dis + dct[u][v]])
            pa = [node]
            while parent[pa[-1]] != -1:
                pa.append(parent[pa[-1]])
            pa.reverse()
            return node, pa

        start, _ = bfs_diameter(0)
        end, path = bfs_diameter(start)

        def bfs_distance(src):
            dis = [0] * n
            stack = [[src, -1, 1]]
            while stack:
                u, fa, state = stack.pop()
                if state:
                    stack.append([u, fa, 0])
                    for v in dct[u]:
                        if v != fa:
                            stack.append([v, u, 1])
                else:
                    x = 0
                    for v in dct[u]:
                        if v != fa:
                            x = max(x, dct[u][v] + dis[v])
                    dis[u] = x
            return dis

        dis1 = bfs_distance(start)  # start -> end
        dis2 = bfs_distance(end)  # end -> start

        def bfs_node(src):
            stack = [[src, -1, 0]]
            res = 0
            while stack:
                u, fa, dis = stack.pop()
                res = max(res, dis)
                for v in dct[u]:
                    if v != fa and v not in diameter:
                        stack.append([v, u, dis + dct[u][v]])
            diameter[src] = res
            return

        diameter = {node: 0 for node in path}
        for node in diameter:
            bfs_node(node)

        m = len(path)
        ans = math.inf
        gap = 0
        j = 0
        q = deque()
        q.append([diameter[path[0]], 0])
        for i in range(m):
            while q and q[0][1] < i:
                q.popleft()
            if i:
                gap -= dct[path[i - 1]][path[i]]
            while j + 1 < m and gap + dct[path[j]][path[j + 1]] <= s:
                gap += dct[path[j]][path[j + 1]]
                while q and q[-1][0] < diameter[path[j + 1]]:
                    q.pop()
                q.append([diameter[path[j + 1]], j + 1])
                j += 1

            ans = min(ans, max(dis2[path[i]], dis1[path[j]], q[0][0]))
        ac.st(ans)
        return

    @staticmethod
    def abc_133e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc133/tasks/abc133_e
        tag: bfs|coloring_method|counter
        """

        n, k = ac.read_list_ints()
        mod = 1000000007
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
            degree[i] += 1
            degree[j] += 1
        if n == 1:
            ac.st(k)
            return
        root = [i for i in range(n) if degree[i] == 1][0]
        stack = [[root, -1, 0, 0]]
        ans = 1
        while stack:
            i, fa, pre, c = stack.pop()
            if pre == 0:
                ans *= (k - c)
            elif pre == 1:
                ans *= (k - 1 - c)
            else:
                ans *= (k - 2 - c)
            ans %= mod
            cnt = 0
            for j in dct[i]:
                if j != fa:
                    cnt += 1
                    stack.append([j, i, pre + 1, cnt - 1])
        ac.st(ans)
        return

    @staticmethod
    def ac_175(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/175/
        tag: multi_source_bfs|classical
        """

        m, n = ac.read_list_ints()
        grid = [ac.read_list_str() for _ in range(m)]
        stack = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    grid[i][j] = 0
                    stack.append([i, j])
                else:
                    grid[i][j] = math.inf
        while stack:
            nex = []
            for i, j in stack:
                for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                    if 0 <= x < m and 0 <= y < n and grid[x][y] == math.inf:
                        nex.append([x, y])
                        grid[x][y] = grid[i][j] + 1
            stack = nex[:]
        for g in grid:
            ac.lst(g)
        return

    @staticmethod
    def ac_177(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/177/
        tag: monotonic_queue|bfs
        """
        for _ in range(ac.read_int()):
            m, n = ac.read_list_ints()
            grid = [ac.read_str() for _ in range(m)]
            dct = [dict() for _ in range((m + 1) * (n + 1))]
            for i in range(m):
                for j in range(n):
                    x1, x2, x3, x4 = i * (n + 1) + j, i * (n + 1) + j + 1, \
                                     (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1
                    if grid[i][j] == "/":
                        dct[x2][x3] = dct[x3][x2] = 0
                        dct[x1][x4] = dct[x4][x1] = 1
                    else:
                        dct[x2][x3] = dct[x3][x2] = 1
                        dct[x1][x4] = dct[x4][x1] = 0
            visit = [math.inf] * ((m + 1) * (n + 1))
            visit[0] = 0
            stack = deque([0])
            while stack and visit[-1] == math.inf:
                i = stack.popleft()
                d = visit[i]
                for j in dct[i]:
                    dd = d + dct[i][j]
                    if dd < visit[j]:
                        visit[j] = dd
                        if dd == d + 1:
                            stack.append(j)
                        else:
                            stack.appendleft(j)
            ac.st(visit[-1] if visit[-1] < math.inf else "NO SOLUTION")
        return

    @staticmethod
    def ac_179(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/179/
        tag: multi_source_bfs|bilateral_bfs
        """
        for _ in range(ac.read_int()):
            m, n = ac.read_list_ints()
            grid = [ac.read_str() for _ in range(m)]
            ghost = []
            boy = []
            girl = []
            for i in range(m):
                for j in range(n):
                    w = grid[i][j]
                    if w == "M":
                        boy = [i, j]
                    elif w == "G":
                        girl = [i, j]
                    elif w == "Z":
                        ghost.append([i, j])

            dis_boy = [[math.inf] * n for _ in range(m)]
            stack_boy = [boy]
            for i, j in stack_boy:
                dis_boy[i][j] = 0

            dis_girl = [[math.inf] * n for _ in range(m)]
            stack_girl = [girl]
            for i, j in stack_girl:
                dis_girl[i][j] = 0

            dis_ghost = [[math.inf] * n for _ in range(m)]
            stack_ghost = ghost[:]
            for i, j in stack_ghost:
                dis_ghost[i][j] = 0
            pre = 0

            ans = math.inf
            while ans == math.inf and stack_girl and stack_boy:
                pre += 1
                for _ in range(2):
                    nex_ghost = []
                    for i, j in stack_ghost:
                        for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                            if 0 <= x < m and 0 <= y < n and dis_ghost[x][y] == math.inf:
                                dis_ghost[x][y] = pre
                                nex_ghost.append([x, y])
                    stack_ghost = nex_ghost[:]

                for _ in range(3):
                    nex_boy = []
                    for i, j in stack_boy:
                        if dis_ghost[i][j] == math.inf:
                            for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                                if 0 <= x < m and 0 <= y < n and dis_boy[x][y] == math.inf and grid[x][y] != "X" and \
                                        dis_ghost[x][y] == math.inf:
                                    dis_boy[x][y] = pre
                                    nex_boy.append([x, y])
                                    if dis_girl[x][y] < math.inf:
                                        ans = pre
                    stack_boy = nex_boy[:]

                for _ in range(1):
                    nex = []
                    for i, j in stack_girl:
                        if dis_ghost[i][j] == math.inf:
                            for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                                if 0 <= x < m and 0 <= y < n and dis_girl[x][y] == math.inf and grid[x][y] != "X" and \
                                        dis_ghost[x][y] == math.inf:
                                    dis_girl[x][y] = pre
                                    if dis_boy[x][y] < math.inf:
                                        ans = pre
                                    nex.append([x, y])
                    stack_girl = nex[:]

            ac.st(ans if ans < math.inf else -1)
        return

    @staticmethod
    def lg_p1213(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1213
        tag: state_compression|01-bfs
        """

        nex = {0: 1, 1: 2, 2: 3, 3: 0}
        lst = "ABDE,ABC,BCEF,ADG,BDEFH,CFI,DEGH,GHI,EFHI".split(",")
        ind = dict()
        for i, st in enumerate(lst):
            ind[i + 1] = [ord(w) - ord("A") for w in st]

        grid = []
        for _ in range(3):
            grid.extend([(num - 3) // 3 for num in ac.read_list_ints()])

        def list_to_num(ls):
            res = 0
            for num in ls:
                res *= 4
                res += num
            return res

        def num_to_list(num):
            res = []
            while num:
                res.append(num % 4)
                num //= 4
            while len(res) < 9:
                res.append(0)
            return res[::-1]

        ans = ""
        start = list_to_num(grid)
        target = list_to_num([3] * 9)

        stack = deque([start])
        visit = dict()
        visit[start] = ""
        if start == target:
            ac.st("")
            return

        while stack:
            state = stack.popleft()
            pre = visit[state]
            if ans and len(pre) > len(ans):
                continue
            if state == target:
                if len(pre) < len(ans) or (len(pre) == len(ans) and pre < ans) or not ans:
                    ans = pre
                continue

            state = num_to_list(state)
            for i in range(9):
                tmp = state[:]
                for w in ind[i + 1]:
                    tmp[w] = nex[tmp[w]]
                cur = list_to_num(tmp)
                if cur not in visit:
                    visit[cur] = pre + str(i + 1)
                    stack.append(cur)
        ac.lst(list(ans))
        return

    @staticmethod
    def lg_p1902(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1902
        tag: binary_search|bfs|in_place_hash
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        for jj in range(n):
            grid[0][jj] = -grid[0][jj] - 1
        dct = dict()

        def check(x):
            stack = [(0, _) for _ in range(n)]
            cnt = 0
            while stack and cnt < n:
                i, j = stack.pop()
                cnt += 1 if i == m - 1 else 0
                if i + 1 < m:
                    a, b = i + 1, j
                    w = grid[a][b]
                    if x >= w >= 0:
                        stack.append((a, b))
                        grid[a][b] = -w - 1
                if i - 1 >= 0:
                    a, b = i - 1, j
                    w = grid[a][b]
                    if x >= w >= 0:
                        stack.append((a, b))
                        grid[a][b] = -w - 1
                if j + 1 < n:
                    a, b = i, j + 1
                    w = grid[a][b]
                    if x >= w >= 0:
                        stack.append((a, b))
                        grid[a][b] = -w - 1
                if j - 1 >= 0:
                    a, b = i, j - 1
                    w = grid[a][b]
                    if x >= w >= 0:
                        stack.append((a, b))
                        grid[a][b] = -w - 1
            for i in range(1, m):
                for j in range(n):
                    w = grid[i][j]
                    if w < 0:
                        grid[i][j] = -w - 1
            return cnt == n

        low = 0
        high = 1000
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                high = mid
                dct[mid] = True
            else:
                low = mid
                dct[mid] = False

        if low in dct:
            ac.st(low if dct[low] else high)
        elif high in dct and not dct[high]:
            ac.st(low)
        else:
            ac.st(low if check(low) else high)
        return

    @staticmethod
    def lg_p2199(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2199
        tag: deque_bfs|01-bfs
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_list_str() for _ in range(m)]
        ind = [[0, 1], [0, -1], [1, 0], [-1, 0],
               [1, 1], [1, -1], [-1, 1], [-1, -1]]
        while True:
            lst = ac.read_list_ints_minus_one()
            if lst == [-1, -1, -1, -1]:
                break
            end = [lst[0], lst[1]]
            start = [lst[2], lst[3]]

            seen = set()
            i, j = end
            seen.add((i, j))
            for a, b in ind:
                x, y = i, j
                while 0 <= x < m and 0 <= y < n and grid[x][y] != "X":
                    seen.add((x, y))
                    x += a
                    y += b
            if (start[0], start[1]) in seen:
                ac.st(0)
                continue

            visit = [[math.inf] * n for _ in range(m)]
            stack = deque([[0, start[0], start[1]]])
            ans = -1
            visit[start[0]][start[1]] = 0
            while stack and ans == -1:
                d, i, j = stack.popleft()
                for a, b in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                    if 0 <= a < m and 0 <= b < n and grid[a][b] != "X" and visit[a][b] == math.inf:
                        visit[a][b] = d + 1
                        stack.append([d + 1, a, b])
                        if (a, b) in seen:
                            ans = d + 1
                            break
            ac.st(ans if ans != -1 else "Poor Harry")
        return

    @staticmethod
    def lg_p2226(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2226
        tag: bfs
        """
        m, n = ac.read_list_ints()
        s1, s2, e1, e2 = ac.read_list_ints_minus_one()
        grid = [ac.read_list_ints() for _ in range(m)]
        ind = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        for t in range(1, 11):
            stack = deque([[s1, s2, -1, 0]])
            visit = [[[0 for _ in range(4)] for _ in range(n)] for _ in range(m)]
            ans = -1
            while stack and ans == -1:
                i, j, d, total = stack.popleft()
                pre = visit[i][j][d] if d != -1 else math.inf
                for dd in range(4):
                    x, y = i + ind[dd][0], j + ind[dd][1]
                    if 0 <= x < m and 0 <= y < n and grid[x][y] == 1 and (dd == d or pre >= t):
                        nex = pre + 1 if d == dd else 1
                        if visit[x][y][dd] < nex:
                            visit[x][y][dd] = nex
                            stack.append([x, y, dd, total + 1])
                            if (x, y) == (e1, e2):
                                ans = total + 1
                                break
            if ans != -1:
                ac.lst([t, ans])
        return

    @staticmethod
    def lg_p2296(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2296
        tag: reverse_graph|bfs
        """
        n, m = ac.read_list_ints()
        dct = [set() for _ in range(n)]
        rev = [set() for _ in range(n)]
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            if x != y:
                dct[x].add(y)
                rev[y].add(x)
        s, t = ac.read_list_ints_minus_one()

        reach = [0] * n
        reach[t] = 1
        stack = [t]
        while stack:
            i = stack.pop()
            for j in rev[i]:
                if not reach[j]:
                    reach[j] = 1
                    stack.append(j)
        if not all(reach[x] for x in dct[s]):
            ac.st(-1)
            return

        visit = [math.inf] * n
        visit[s] = 0
        stack = deque([s])
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                if all(reach[k] for k in dct[j]) and visit[j] == math.inf:
                    visit[j] = visit[i] + 1
                    stack.append(j)
        ac.st(visit[t] if visit[t] < math.inf else -1)
        return

    @staticmethod
    def lg_p2919(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2919
        tag: bfs
        """
        m, n = ac.read_list_ints()
        grid = []
        for _ in range(m):
            grid.append(ac.read_list_ints())
        nodes = []
        for i in range(m):
            for j in range(n):
                nodes.append([i, j])
        nodes = deque(sorted(nodes, reverse=True, key=lambda it: grid[it[0]][it[1]]))

        ans = 0
        while nodes:
            i, j = nodes.popleft()
            if grid[i][j] == -1:
                continue
            ans += 1
            stack = [[grid[i][j], i, j]]
            grid[i][j] = -1
            while stack:
                val, i, j = stack.pop()
                for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1],
                             [i - 1, j - 1], [i - 1, j + 1], [i + 1, j - 1], [i + 1, j + 1]]:
                    if 0 <= x < m and 0 <= y < n and -1 < grid[x][y] <= val:
                        stack.append([grid[x][y], x, y])
                        grid[x][y] = -1
        ac.st(ans)
        return

    @staticmethod
    def lg_p2937(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2937
        tag: 01-bfs|monotonic_queue
        """
        n, m = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]
        visit = [[[math.inf] * 4 for _ in range(n)] for _ in range(m)]
        res = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "C":
                    res.append([i, j])
        start, end = res[0], res[1]
        ind = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        stack = deque([[d, start[0], start[1]] for d in range(4)])
        visit[start[0]][start[1]] = [0, 0, 0, 0]
        while stack:
            d, i, j = stack.popleft()
            x, y = i + ind[d][0], j + ind[d][1]
            if 0 <= x < m and 0 <= y < n and grid[x][y] != "*" and visit[x][y][d] > visit[i][j][d]:
                visit[x][y][d] = visit[i][j][d]
                stack.appendleft([d, x, y])
            for dd in [d - 1, d + 1]:
                dd %= 4
                x, y = i + ind[dd][0], j + ind[dd][1]
                if 0 <= x < m and 0 <= y < n and grid[x][y] != "*" and visit[x][y][dd] > visit[i][j][d] + 1:
                    visit[x][y][dd] = visit[i][j][d] + 1
                    stack.append([dd, x, y])
        ac.st(min(visit[end[0]][end[1]]))
        return

    @staticmethod
    def lg_p3456(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3456
        tag: bfs
        """
        n = ac.read_int()
        grid = [ac.read_list_ints() for _ in range(n)]
        visit = [[0] * n for _ in range(n)]
        ceil = floor = 0
        for x in range(n):
            for y in range(n):
                if visit[x][y]:
                    continue
                visit[x][y] = 1
                stack = [[x, y]]
                big = small = False
                while stack:
                    i, j = stack.pop()
                    for a, b in ((i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1),
                                 (i, j + 1), (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)):
                        if 0 <= a < n and 0 <= b < n:
                            if grid[a][b] > grid[x][y]:
                                big = True
                            elif grid[a][b] < grid[x][y]:
                                small = True
                            else:
                                if not visit[a][b]:
                                    stack.append([a, b])
                                    visit[a][b] = 1
                if small and big:
                    continue
                else:
                    if big:
                        ceil += 1
                    elif small:
                        floor += 1
                    else:
                        ceil += 1
                        floor += 1
        ac.lst([ceil, floor][::-1])
        return

    @staticmethod
    def lg_p3818(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3818
        tag: 01-bfs|deque_bfs
        """
        m, n, d, r = ac.read_list_ints()
        grid = []
        for _ in range(m):
            grid.append(ac.read_str())
        visit = [[[None] * 2 for _ in range(n)] for _ in range(m)]
        visit[0][0][0] = 0
        stack = deque([[0, 0, 0]])
        while stack:
            i, j, s = stack.popleft()
            if i == m - 1 and j == n - 1:
                ac.st(visit[i][j][s])
                return
            if (s == 0 and 0 <= i + d < m and 0 <= j + r < n and
                    not visit[i + d][j + r][1] and grid[i + d][j + r] != "#"):
                visit[i + d][j + r][1] = visit[i][j][s] + 1
                stack.append([i + d, j + r, 1])
            for x, y in [[i - 1, j], [i, j + 1], [i + 1, j], [i, j - 1]]:
                if 0 <= x < m and 0 <= y < n and not visit[x][y][s] and grid[x][y] != "#":
                    visit[x][y][s] = visit[i][j][s] + 1
                    stack.append([x, y, s])
        ac.st(-1)
        return

    @staticmethod
    def lg_p3855(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3855
        tag: bfs|md_state
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]
        gg = [-1, -1]
        mm = [-1, -1]
        tt = [-1, -1]
        for i in range(m):
            for j in range(n):
                w = grid[i][j]
                if w == "G":
                    gg = [i, j]
                elif w == "M":
                    mm = [i, j]
                elif w == "T":
                    tt = [i, j]

        visit = [[[[-1 for _ in range(n)] for _ in range(m)] for _ in range(n)] for _ in range(m)]
        stack = deque([[gg[0], gg[1], mm[0], mm[1]]])
        visit[gg[0]][gg[1]][mm[0]][mm[1]] = 0
        while stack:
            a, b, c, d = stack.popleft()
            ind = [[1, 0, 1, 0], [-1, 0, -1, 0], [0, 1, 0, -1], [0, -1, 0, 1]]
            for a0, b0, c0, d0 in ind:
                if 0 <= a + a0 < m and 0 <= b + b0 < n and grid[a + a0][b + b0] == "X":
                    continue
                if 0 <= c + c0 < m and 0 <= d + d0 < n and grid[c + c0][d + d0] == "X":
                    continue
                if 0 <= a + a0 < m and 0 <= b + b0 < n and grid[a + a0][b + b0] != "#":
                    x, y = a + a0, b + b0
                else:
                    x, y = a, b
                if 0 <= c + c0 < m and 0 <= d + d0 < n and grid[c + c0][d + d0] != "#":
                    p, q = c + c0, d + d0
                else:
                    p, q = c, d

                if visit[x][y][p][q] == -1:
                    visit[x][y][p][q] = visit[a][b][c][d] + 1
                    stack.append([x, y, p, q])
                    if [x, y] == [p, q] == tt:
                        ac.st(visit[x][y][p][q])
                        return
        ac.no()
        return

    @staticmethod
    def lg_p3869(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3869
        tag: bfs|state_compression
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]
        k = ac.read_int()
        pos = dict()
        ind = dict()
        for i in range(k):
            a, b, c, d = ac.read_list_ints_minus_one()
            if (c, d) not in ind:
                ind[(c, d)] = len(ind)
            if (a, b) not in pos:
                pos[(a, b)] = []
            pos[(a, b)].append((c, d))
        k = len(ind)

        visit = [[[math.inf] * (1 << k) for _ in range(n)] for _ in range(m)]
        ss = [-1, -1]
        tt = [-1, -1]
        for i in range(m):
            for j in range(n):
                w = grid[i][j]
                if w == "S":
                    ss = [i, j]
                elif w == "T":
                    tt = [i, j]
        stack = deque([[ss[0], ss[1], 0]])
        visit[ss[0]][ss[1]][0] = 0
        while stack:
            i, j, state = stack.popleft()
            if [i, j] == tt:
                break
            for a, b in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                if 0 <= a < m and 0 <= b < n:
                    if (a, b) not in ind:
                        if grid[a][b] != "#":
                            cur_state = state
                            if (a, b) in pos:
                                for (c, d) in pos[(a, b)]:
                                    cur_state ^= (1 << ind[(c, d)])
                            if visit[a][b][cur_state] == math.inf:
                                stack.append([a, b, cur_state])
                                visit[a][b][cur_state] = visit[i][j][state] + 1
                    else:
                        if (grid[a][b] != "#") == (not state & (1 << ind[(a, b)])):
                            cur_state = state
                            if (a, b) in pos:
                                for (c, d) in pos[(a, b)]:
                                    cur_state ^= (1 << ind[(c, d)])
                            if visit[a][b][cur_state] == math.inf:
                                stack.append([a, b, cur_state])
                                visit[a][b][cur_state] = visit[i][j][state] + 1
        ac.st(min(visit[tt[0]][tt[1]]))
        return

    @staticmethod
    def lg_p4554(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4554
        tag: classical|01-bfs|implemention
        """
        while True:
            lst = ac.read_list_ints()
            if lst == [0, 0]:
                break
            m, n = lst
            grid = [ac.read_str() for _ in range(m)]
            x1, y1, x2, y2 = ac.read_list_ints()
            visit = [[math.inf] * n for _ in range(m)]
            stack = deque([[x1, y1]])
            visit[x1][y1] = 0
            while stack and visit[x2][y2] == math.inf:
                x, y = stack.popleft()
                w = grid[x][y]
                d = visit[x][y]
                for a, b in [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]:
                    if 0 <= a < m and 0 <= b < n:
                        cost = d if grid[a][b] == w else d + 1
                        if visit[a][b] > cost:
                            visit[a][b] = cost
                            if cost == d:
                                stack.appendleft([a, b])
                            else:
                                stack.append([a, b])
            ac.st(visit[x2][y2])
        return

    @staticmethod
    def lg_p4667(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4667
        tag: 01-bfs|implemention
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]
        dct = [dict() for _ in range((m + 1) * (n + 1))]
        for i in range(m):
            for j in range(n):
                x1, x2, x3, x4 = i * (n + 1) + j, i * (n + 1) + j + 1, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1
                if grid[i][j] == "/":
                    dct[x2][x3] = dct[x3][x2] = 0
                    dct[x1][x4] = dct[x4][x1] = 1
                else:
                    dct[x2][x3] = dct[x3][x2] = 1
                    dct[x1][x4] = dct[x4][x1] = 0
        visit = [math.inf] * ((m + 1) * (n + 1))
        visit[0] = 0
        stack = deque([[0, 0]])
        while stack and visit[-1] == math.inf:
            i, d = stack.popleft()
            if visit[i] < d:
                continue
            for j in dct[i]:
                dd = d + dct[i][j]
                if dd < visit[j]:
                    visit[j] = dd
                    if dd == d + 1:
                        stack.append([j, dd])
                    else:
                        stack.appendleft([j, dd])
        ac.st(visit[-1] if visit[-1] < math.inf else "NO SOLUTION")
        return

    @staticmethod
    def lg_p5096(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5096
        tag: state_compression|bfs|implemention
        """
        n, m, k = ac.read_list_ints()  # TLE
        dct = [dict() for _ in range(n)]
        cao = dict()
        for i in range(k):
            cao[ac.read_int() - 1] = i

        for _ in range(m):
            a, b, c = ac.read_list_ints_minus_one()
            dct[a][b] = dct[b][a] = c + 1
        visit = [[0] * (1 << k) for _ in range(n)]
        cnt = [bin(x).count("1") for x in range(1 << k)]
        visit[0][0] = 1
        stack = [[0, 0]]
        while stack:
            i, state = stack.pop()
            for j in dct[i]:
                w = dct[i][j]
                if cnt[state] > w:
                    continue
                nex = state
                if j in cao:
                    nex |= 1 << cao[j]
                if not visit[j][nex]:
                    visit[j][nex] = 1
                    stack.append([j, nex])
                if not visit[j][state]:
                    visit[j][state] = 1
                    stack.append([j, state])
        ans = 0
        for x in range(1 << k):
            if visit[0][x]:
                ans = max(ans, cnt[x])
        ac.st(ans)
        return

    @staticmethod
    def lg_p5099(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5099
        tag: 01-bfs|implemention
        """
        n, t = ac.read_list_ints()
        dct = dict()
        for i in range(n):
            x, z = ac.read_list_ints()
            dct[(x, z)] = i
        visit = [math.inf] * n
        stack = deque([[0, 0, -1]])
        ans = math.inf
        while stack:
            i, j, ind = stack.popleft()
            d = 0 if ind == -1 else visit[ind]
            if j == t:
                ans = d
                break
            for a in range(-2, 3, 1):
                for b in range(-2, 3, 1):
                    if (i + a, j + b) in dct and visit[dct[(i + a, j + b)]] > d + 1:
                        visit[dct[(i + a, j + b)]] = d + 1
                        stack.append([i + a, j + b, dct[(i + a, j + b)]])
        ac.st(ans if ans < math.inf else -1)
        return

    @staticmethod
    def lg_p5195(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5195
        tag: bfs
        """
        n, m = ac.read_list_ints()
        lst = []
        while len(lst) < m * n:
            lst.extend(ac.read_list_ints())
        grid = [lst[i * n: i * n + n] for i in range(m)]
        del lst
        pos_2 = [-1, -1]
        wood = []
        for i in range(m):
            for j in range(n):
                w = grid[i][j]
                if w == 2:
                    pos_2 = [i, j]
                elif w == 4:
                    wood.append([i, j])

        visit = [[[math.inf, math.inf] for _ in range(n)] for _ in range(m)]
        stack = deque([pos_2 + [0]])
        visit[pos_2[0]][pos_2[1]][0] = 0
        ans = math.inf
        while stack:
            i, j, state = stack.popleft()
            d = visit[i][j][state]
            if grid[i][j] == 3 and state == 1:
                ans = d
                break
            for a, b in [[i + 1, j], [i - 1, j], [i, j - 1], [i, j + 1]]:
                if 0 <= a < m and 0 <= b < n:
                    if state and grid[a][b] != 1 and visit[a][b][state] > d + 1:
                        visit[a][b][state] = d + 1
                        stack.append([a, b, state])
                    if not state and grid[a][b] not in [1, 3]:
                        if grid[a][b] == 4:
                            cur = 1
                        else:
                            cur = 0
                        if visit[a][b][cur] > d + 1:
                            visit[a][b][cur] = d + 1
                            stack.append([a, b, cur])

        ac.st(ans if ans < math.inf else -1)
        return

    @staticmethod
    def lg_p6131(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6131
        tag: bfs|union_find
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_list_str() for _ in range(m)]

        color = 0
        dct = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "X":
                    stack = [[i, j]]
                    grid[i][j] = str(color)
                    cur = []
                    while stack:
                        a, b = stack.pop()
                        cur.append([a, b])
                        for x, y in [[a - 1, b], [a + 1, b], [a, b - 1], [a, b + 1]]:
                            if 0 <= x < m and 0 <= y < n and grid[x][y] == "X":
                                stack.append([x, y])
                                grid[x][y] = str(color)
                    color += 1
                    dct.append(cur)

        dis = [[0] * n for _ in range(m)]
        for c in range(3):
            stack = deque(dct[c])
            cur = [[math.inf] * n for _ in range(m)]
            for i, j in stack:
                cur[i][j] = 0
            while stack:
                a, b = stack.popleft()
                for x, y in [[a - 1, b], [a + 1, b], [a, b - 1], [a, b + 1]]:
                    if 0 <= x < m and 0 <= y < n:
                        if grid[x][y] != ".":
                            if cur[x][y] > cur[a][b]:
                                cur[x][y] = cur[a][b]
                                stack.append([x, y])
                        else:
                            if cur[x][y] > cur[a][b] + 1:
                                cur[x][y] = cur[a][b] + 1
                                stack.append([x, y])
            for i in range(m):
                for j in range(n):
                    dis[i][j] += cur[i][j]

        ans = math.inf
        for i in range(m):
            for j in range(n):
                if grid[i][j] == ".":
                    ans = min(ans, dis[i][j] - 2)
                else:
                    ans = min(ans, dis[i][j])
        ac.st(ans)
        return

    @staticmethod
    def lg_p6909(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6909
        tag: preprocess|bfs
        """

        m, n = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]

        up = [[-1] * n for _ in range(m)]
        for j in range(n):
            for i in range(1, m):
                if grid[i][j] == grid[i - 1][j]:
                    up[i][j] = up[i - 1][j]
                else:
                    up[i][j] = i - 1

        down = [[-1] * n for _ in range(m)]
        for j in range(n):
            for i in range(m - 2, -1, -1):
                if grid[i][j] == grid[i + 1][j]:
                    down[i][j] = down[i + 1][j]
                else:
                    down[i][j] = i + 1

        left = [[-1] * n for _ in range(m)]
        for i in range(m):
            for j in range(1, n):
                if grid[i][j] == grid[i][j - 1]:
                    left[i][j] = left[i][j - 1]
                else:
                    left[i][j] = j - 1

        right = [[-1] * n for _ in range(m)]
        for i in range(m):
            for j in range(n - 2, -1, -1):
                if grid[i][j] == grid[i][j + 1]:
                    right[i][j] = right[i][j + 1]
                else:
                    right[i][j] = j + 1

        s = ac.read_str() + "*"
        k = len(s)
        visit = [[-1] * n for _ in range(m)]
        visit[0][0] = 0
        stack = deque([[0, 0, 0, 0]])
        ans = -1
        while stack and ans == -1:
            d, ind, i, j = stack.popleft()
            if s[ind] == grid[i][j]:
                if ind + 1 > visit[i][j]:
                    stack.append([d + 1, ind + 1, i, j])
                    visit[i][j] = ind + 1
                    if ind + 1 == k:
                        ans = d + 1
                        break
            if up[i][j] != -1:
                x, y = up[i][j], j
                if visit[x][y] < ind:
                    visit[x][y] = ind
                    stack.append([d + 1, ind, x, y])
            if down[i][j] != -1:
                x, y = down[i][j], j
                if visit[x][y] < ind:
                    visit[x][y] = ind
                    stack.append([d + 1, ind, x, y])
            if left[i][j] != -1:
                x, y = i, left[i][j]
                if visit[x][y] < ind:
                    visit[x][y] = ind
                    stack.append([d + 1, ind, x, y])
            if right[i][j] != -1:
                x, y = i, right[i][j]
                if visit[x][y] < ind:
                    visit[x][y] = ind
                    stack.append([d + 1, ind, x, y])
        ac.st(ans)
        return

    @staticmethod
    def lg_p9065(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P9065
        tag: brain_teaser|bfs|brute_force
        """
        m, n, k = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        pos = set(tuple(ac.read_list_ints_minus_one()) for _ in range(k))

        def bfs(s1, s2):
            stack = deque()
            dis = [[math.inf] * n for _ in range(m)]
            dis[s1][s2] = 0
            stack.append([0, s1, s2])
            while stack:
                d, ii, jj = stack.popleft()
                for x, y in [[ii - 1, jj], [ii + 1, jj], [ii, jj - 1], [ii, jj + 1]]:
                    if 0 <= x < m and 0 <= y < n and d + 1 < dis[x][y] and grid[x][y]:
                        dis[x][y] = d + 1
                        stack.append([d + 1, x, y])
            return dis

        dis1 = bfs(0, 0)
        dis2 = bfs(m - 1, n - 1)
        ans = dis1[m - 1][n - 1]
        if pos:
            pre = defaultdict(lambda: math.inf)
            for i, j in pos:
                pre[grid[i][j]] = min(pre[grid[i][j]], dis1[i][j])
            floor = min(pre.values())
            for i, j in pos:
                cur = min(pre[grid[i][j]], floor + 1) + dis2[i][j] + 1
                ans = min(ans, cur)
        ac.st(ans if ans < math.inf else -1)
        return

    @staticmethod
    def cf_1594d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1594/problem/D
        tag: build_graph|coloring_method|bfs|bipartite_graph
        """
        for _ in range(ac.read_int()):

            def check():
                n, m = ac.read_list_ints()
                dct = [[] for _ in range(n)]
                for _ in range(m):
                    x, y, w = ac.read_list_strs()
                    x = int(x) - 1
                    y = int(y) - 1
                    if w == "imposter":
                        z = 1
                    else:
                        z = 0
                    dct[x].append([y, z])
                    dct[y].append([x, z])
                color = [-1] * n
                ans = 0
                for i in range(n):
                    if color[i] == -1:
                        stack = [i]
                        color[i] = 0
                        cnt = [0, 0]
                        cnt[0] = 1
                        while stack:
                            x = stack.pop()
                            for y, z in dct[x]:
                                ww = color[x] ^ z
                                if color[y] == -1:
                                    color[y] = ww
                                    stack.append(y)
                                    cnt[ww] += 1
                                else:
                                    if color[y] != ww:
                                        ac.st(-1)
                                        return
                        ans += max(cnt)
                ac.st(ans)
                return

            check()
        return

    @staticmethod
    def lc_909(board: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/snakes-and-ladders/
        tag: 01-bfs|implemention
        """

        def position(num):
            i = (num - 1) // n
            j = (num - 1) % n
            if i % 2:
                return [n - 1 - i, n - 1 - j]
            return [n - 1 - i, j]

        n = len(board)
        visit = {1}
        stack = deque([[1, 0]])
        while stack:
            pre, ans = stack.popleft()
            if pre == n * n:
                return ans
            for nex in range(pre + 1, pre + 7):
                a, b = position(nex)
                if board[a][b] != -1:
                    nex = board[a][b]
                if nex == n * n:
                    return ans + 1
                if nex not in visit:
                    visit.add(nex)
                    stack.append([nex, ans + 1])
        return -1

    @staticmethod
    def lc_994(grid: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/rotting-oranges/description/
        tag: deque_bfs|implemention
        """

        m, n = len(grid), len(grid[0])
        stack = deque()
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    stack.append([i, j, 0])
        ans = 0
        while stack:
            i, j, d = stack.popleft()
            ans = d
            for x, y in [[i + 1, j], [i - 1, j], [i, j + 1], [i, j - 1]]:
                if 0 <= x < m and 0 <= y < n and grid[x][y] == 1:
                    grid[x][y] = 2
                    stack.append([x, y, d + 1])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    return -1
        return ans

    @staticmethod
    def lc_1036_1(blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        """
        url: https://leetcode.cn/problems/escape-a-large-maze/
        tag: bound_bfs|discretization_bfs
        """

        def check(node):
            stack = [node]
            visit = {tuple(node)}
            while stack:
                nex = []
                for i, j in stack:
                    for x, y in [[i - 1, j], [i + 1, j], [i, j + 1], [i, j - 1]]:
                        if 0 <= x < n and 0 <= y < n and (x, y) not in visit and (x, y) not in block:
                            nex.append([x, y])
                            visit.add((x, y))
                stack = nex
                if len(visit) >= ceil:
                    break
            return visit

        n = 10 ** 6
        block = set(tuple(b) for b in blocked)
        m = len(block)
        ceil = m * m
        visit_s = check(source)
        visit_t = check(target)
        return len(visit_s.intersection(visit_t)) > 0 or (len(visit_s) >= ceil and len(visit_t) >= ceil)

    @staticmethod
    def lc_1036_2(blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        """
        url: https://leetcode.cn/problems/escape-a-large-maze/
        tag: bound_bfs|discretization_bfs
        """

        nodes_r = {0, 10 ** 6 - 1}
        nodes_c = {0, 10 ** 6 - 1}
        for a, b in blocked + [source] + [target]:
            nodes_r.add(a)
            nodes_c.add(b)

        nodes_r = sorted(list(nodes_r))
        m = len(nodes_r)
        ind_r = dict()
        x = 0
        ind_r[nodes_r[0]] = x
        for i in range(1, m):
            if nodes_r[i] == nodes_r[i - 1] + 1:
                x += 1
            else:
                x += 2
            ind_r[nodes_r[i]] = x
        r_id = x

        nodes_c = sorted(list(nodes_c))
        m = len(nodes_c)
        ind_c = dict()
        x = 0
        ind_c[nodes_c[0]] = x
        for i in range(1, m):
            if nodes_c[i] == nodes_c[i - 1] + 1:
                x += 1
            else:
                x += 2
            ind_c[nodes_c[i]] = x
        c_id = x

        blocked = set((ind_r[b[0]], ind_c[b[1]]) for b in blocked)
        source = (ind_r[source[0]], ind_c[source[1]])
        target = (ind_r[target[0]], ind_c[target[1]])
        stack = deque([source])
        visit = {source}
        while stack:
            i, j = stack.popleft()
            for x, y in [[i - 1, j], [i + 1, j], [i, j + 1], [i, j - 1]]:
                if 0 <= x <= r_id and 0 <= y <= c_id and (x, y) not in visit and (x, y) not in blocked:
                    stack.append((x, y))
                    if (x, y) == target:
                        return True
                    visit.add((x, y))
        return False

    @staticmethod
    def ac_4415(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4418
        tag: bfs|coloring_method|odd_circle|specific_plan|counter
        """
        mod = 998244353

        def check():
            n, m = ac.read_list_ints()
            dct = [[] for _ in range(n)]
            for _ in range(m):
                u, v = ac.read_list_ints_minus_one()
                dct[u].append(v)
                dct[v].append(u)

            visit = [-1] * n
            ans = 1
            for i in range(n):
                if visit[i] == -1:
                    stack = [i]
                    color = 0
                    visit[i] = color
                    cnt = [1, 0]
                    while stack:
                        color = 1 - color
                        nex = []
                        for x in stack:
                            for y in dct[x]:
                                if visit[y] == -1:
                                    visit[y] = color
                                    cnt[color] += 1
                                    nex.append(y)
                                elif visit[y] != color:
                                    ac.st(0)
                                    return
                        stack = nex
                    res = pow(2, cnt[0], mod) + pow(2, cnt[1], mod)
                    ans *= res
                    ans %= mod
            ac.st(ans)
            return

        for _ in range(ac.read_int()):
            check()
        return

    @staticmethod
    def lg_p1330(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1330
        tag: bfs|level_wise|coloring_method|union_find|odd_circle
        """
        n, m = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        for _ in range(m):
            u, v = ac.read_list_ints_minus_one()
            edge[u].append(v)
            edge[v].append(u)

        visit = [-1] * n
        ans = 0
        for i in range(n):
            if visit[i] == -1:
                stack = [i]
                color = 0
                visit[i] = color
                cnt = [1, 0]
                while stack:
                    color = 1 - color
                    nex = []
                    for x in stack:
                        for y in edge[x]:
                            if visit[y] == -1:
                                visit[y] = color
                                cnt[color] += 1
                                nex.append(y)
                            elif visit[y] != color:
                                # 奇数环
                                ac.st("Impossible")
                                return
                    stack = nex
                ans += cnt[0] if cnt[0] < cnt[1] else cnt[1]
        ac.st(ans)
        return

    @staticmethod
    def ac_4484(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4484/
        tag: 01-bfs
        """

        m, n = ac.read_list_ints()
        r, c = ac.read_list_ints_minus_one()
        x, y = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]

        visit = [[0] * n for _ in range(m)]
        visit[r][c] = 1
        stack = deque([[0, 0, r, c]])
        while stack:
            a, b, x1, y1 = stack.popleft()
            for c, d in [[x1 - 1, y1], [x1 + 1, y1]]:
                if 0 <= c < m and 0 <= d < n and grid[c][d] == "." and not visit[c][d]:
                    visit[c][d] = 1
                    stack.appendleft([a, b, c, d])

            for c, d in [[x1, y1 + 1]]:
                if 0 <= c < m and 0 <= d < n and grid[c][d] == "." and b + \
                        1 <= y and not visit[c][d]:
                    visit[c][d] = 1
                    stack.append([a, b + 1, c, d])

            for c, d in [[x1, y1 - 1]]:
                if 0 <= c < m and 0 <= d < n and grid[c][d] == "." and a + \
                        1 <= x and not visit[c][d]:
                    visit[c][d] = 1
                    stack.append([a + 1, b, c, d])

        ans = 0
        for i in range(m):
            for j in range(n):
                if visit[i][j]:
                    ans += 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p1144(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1144
        tag: number_of_shortest_path|bfs
        """
        mod = 100003
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            dct[x].append(y)
            dct[y].append(x)
        n = len(dct)
        dis = [math.inf for _ in range(n)]
        cnt = [0] * n
        queue = deque([0])
        dis[0] = 0
        cnt[0] = 1
        while queue:
            u = queue.popleft()
            for v in dct[u]:
                if dis[v] > dis[u] + 1:
                    dis[v] = dis[u] + 1
                    cnt[v] = cnt[u]
                    cnt[v] %= mod
                    queue.append(v)
                elif dis[v] == dis[u] + 1:
                    cnt[v] += cnt[u]
                    cnt[v] %= mod
        for x in cnt:
            ac.st(x)
        return

    @staticmethod
    def abc_070d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc070/tasks/abc070_d
        tag: classical|bfs|tree_dis
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            a, b, c = ac.read_list_ints()
            dct[a - 1].append((b - 1, c))
            dct[b - 1].append((a - 1, c))

        q, k = ac.read_list_ints()
        k -= 1
        dis = [0] * n
        stack = [[k, -1]]
        while stack:
            i, fa = stack.pop()
            for j, w in dct[i]:
                if j != fa:
                    stack.append([j, i])
                    dis[j] = dis[i] + w
        for _ in range(q):
            a, b = ac.read_list_ints_minus_one()
            ac.st(dis[a] + dis[b])
        return

    @staticmethod
    def abc_336f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc336/tasks/abc336_f
        tag: bilateral_bfs|classical|matrix_rotate
        """
        m, n = ac.read_list_ints()
        grid = []
        for _ in range(m):
            grid.extend(ac.read_list_ints())

        def check():
            visit = {tuple(grid): 0}
            stack = [grid[:]]
            for s in range(10):
                nex = []
                for pre in stack:

                    for a, b, c, d in [[0, 0, m - 2, n - 2], [0, 1, m - 2, n - 1], [1, 0, m - 1, n - 2],
                                       [1, 1, m - 1, n - 1]]:
                        tmp = pre[:]
                        for i in range(m):
                            for j in range(n):
                                if a <= i <= c and b <= j <= d:
                                    new_i = c - (i - a)
                                    new_j = d - (j - b)
                                else:
                                    new_i = i
                                    new_j = j
                                tmp[new_i * n + new_j] = pre[i * n + j]
                        if tuple(tmp) not in visit:
                            visit[tuple(tmp)] = s + 1
                            nex.append(tmp[:])
                stack = [ls[:] for ls in nex]
            return visit

        visit1 = check()
        grid = list(range(1, m * n + 1))
        visit2 = check()
        ans = math.inf
        for k in visit1:
            if k in visit2:
                ans = min(ans, visit1[k] + visit2[k])
        ac.st(ans if ans < math.inf else -1)
        return

    @staticmethod
    def cf_1593e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1593/problem/E
        tag: classical|topological_sort|undirected
        """
        for _ in range(ac.read_int()):
            ac.read_str()
            n, k = ac.read_list_ints()
            degree = [0] * n
            dct = [[] for _ in range(n)]
            for _ in range(n - 1):
                u, v = ac.read_list_ints_minus_one()
                dct[u].append(v)
                dct[v].append(u)
                degree[u] += 1
                degree[v] += 1
            stack = [i for i in range(n) if degree[i] == 1]
            for _ in range(k):
                if not stack:
                    break
                nex = []
                for i in stack:
                    degree[i] = 0
                for i in stack:
                    for j in dct[i]:
                        if degree[j] > 1:
                            degree[j] -= 1
                            if degree[j] == 1:
                                nex.append(j)
                stack = nex[:]
            ans = sum(x >= 1 for x in degree)
            ac.st(ans)
        return

    @staticmethod
    def cf_1674g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1674/problem/G
        tag: classical|brain_teaser|dag_dp|topologic_sort
        """
        n, m = ac.read_list_ints()
        in_degree = [0] * n
        out_degree = [0] * n
        dp = [1] * n
        dct = [[] for _ in range(n)]
        for _ in range(m):
            u, v = ac.read_list_ints_minus_one()
            in_degree[v] += 1
            out_degree[u] += 1
            dct[u].append(v)
        stack = deque([x for x in range(n) if in_degree[x] == 0])
        degree = in_degree[:]
        while stack:
            x = stack.popleft()
            for y in dct[x]:
                if out_degree[x] > 1 and in_degree[y] > 1:
                    dp[y] = max(dp[y], dp[x] + 1)
                degree[y] -= 1
                if not degree[y]:
                    stack.append(y)
        ac.st(max(dp))
        return

    @staticmethod
    def abc_302f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc302/tasks/abc302_f
        tag: build_graph|bfs|brain_teaser
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(m + n)]
        for i in range(n):
            ac.read_int()
            lst = ac.read_list_ints_minus_one()
            for x in lst:
                dct[x].append(m + i)
                dct[m + i].append(x)
        dis = [math.inf] * (m + n)
        dis[0] = 0
        stack = [0]
        while stack:
            nex = []
            for i in stack:
                if i == m - 1:
                    ac.st((dis[m - 1] - 2) // 2)
                    return
                for j in dct[i]:
                    if dis[j] == math.inf:
                        dis[j] = dis[i] + 1
                        nex.append(j)
            stack = nex
        ac.st(-1)
        return

    @staticmethod
    def abc_282d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc282/tasks/abc282_d
        tag: color_method|bipartite_graph|bfs|classical
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        color = [-1] * n
        ans = pre = 0
        for x in range(n):
            if color[x] == -1:
                stack = [x]
                color[x] = 0
                zero = 1
                tot = 1
                edge = 0
                while stack:
                    nex = []
                    for i in stack:
                        c = color[i]
                        for j in dct[i]:
                            edge += 1
                            if color[j] != -1:
                                if color[j] != 1 - c:
                                    ac.st(0)
                                    return
                            else:
                                color[j] = 1 - c
                                nex.append(j)
                                tot += 1
                                if c:
                                    zero += 1
                    stack = nex[:]
                ans += zero * (tot - zero) - edge // 2
                ans += pre * tot
                pre += tot
        ac.st(ans)
        return

    @staticmethod
    def abc_280f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc280/tasks/abc280_f
        tag: bfs|negative_circle|positive_circle|brain_teaser|classical
        """
        n, m, q = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        dis = [math.inf] * n
        for _ in range(m):
            a, b, c = ac.read_list_ints_minus_one()
            c += 1
            dct[a].append((b, c))
            dct[b].append((a, -c))
        visit = [0] * n
        for i in range(n):
            if not visit[i]:
                visit[i] = 1
                dis[i] = 0
                stack = [i]
                lst = [i]
                circle = 0
                while stack:
                    x = stack.pop()
                    for y, w in dct[x]:
                        if dis[y] != math.inf and dis[y] != dis[x] + w:
                            circle = 1
                        dis[y] = dis[x] + w
                        if not visit[y]:
                            visit[y] = 1
                            lst.append(y)
                            stack.append(y)
                            dis[y] = dis[x] + w
                if circle:
                    for x in lst:
                        dis[x] = math.inf
                for x in lst:
                    visit[x] = lst[0] + 1
        for _ in range(q):
            x, y = ac.read_list_ints_minus_one()
            if visit[x] != visit[y]:
                ac.st("nan")
            elif dis[x] == math.inf:
                ac.st("math.inf")
            else:
                ac.st(dis[y] - dis[x])
        return

    @staticmethod
    def abc_246e_1(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc246/tasks/abc246_e
        tag: bfs|union_find|brain_teaser|prune|classical
        """
        n = ac.read_int()
        sx, sy = ac.read_list_ints_minus_one()
        tx, ty = ac.read_list_ints_minus_one()

        grid = [ac.read_str() for _ in range(n)]
        visit = [[math.inf] * n for _ in range(n)]
        ind = [[-1, 1], [-1, -1], [1, 1], [1, -1]]
        visit[sx][sy] = 0
        stack = [(sx, sy)]
        while stack:
            nex = []
            for x, y in stack:
                d = visit[x][y] + 1
                for dx, dy in ind:
                    px, py = x, y
                    while 0 <= px + dx < n and 0 <= py + dy < n and grid[px + dx][py + dy] != "#" and visit[px + dx][
                        py + dy] >= d:
                        px, py = px + dx, py + dy
                        if visit[px][py] == math.inf:
                            visit[px][py] = d
                            nex.append((px, py))

            stack = nex
        ans = visit[tx][ty]
        ac.st(ans if ans < math.inf else -1)
        return

    @staticmethod
    def abc_246e_2(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc246/tasks/abc246_e
        tag: bfs|union_find|brain_teaser|prune|classical
        """
        n = ac.read_int()
        sx, sy = ac.read_list_ints_minus_one()
        tx, ty = ac.read_list_ints_minus_one()
        visit = [[math.inf] * n for _ in range(n)]
        grid = [ac.read_str() for _ in range(n)]
        stack = [(sx, sy)]
        visit[sx][sy] = 0
        uf1 = UnionFind(n * n)
        uf2 = UnionFind(n * n)
        uf3 = UnionFind(n * n)
        uf4 = UnionFind(n * n)

        while stack:
            nex = []
            for x, y in stack:
                cx, cy = x, y
                while True:
                    root = uf1.find(cx * n + cy)
                    if root == cx * n + cy:
                        root = (cx - 1) * n + cy + 1
                    if not (0 <= (cx - 1) < n and 0 <= cy + 1 < n and 0 <= root < n * n):
                        break
                    px, py = root // n, root % n
                    if 0 <= px < n and 0 <= py < n:
                        uf1.union_right(cx * n + cy, root)
                        if grid[px][py] != '#':
                            if visit[px][py] == math.inf:
                                nex.append((px, py))
                                visit[px][py] = visit[x][y] + 1
                            cx, cy = px, py
                        else:
                            break
                    else:
                        break

                cx, cy = x, y
                while True:
                    root = uf2.find(cx * n + cy)
                    if root == cx * n + cy:
                        root = (cx - 1) * n + cy - 1
                    if not (0 <= (cx - 1) < n and 0 <= cy - 1 < n and 0 <= root < n * n):
                        break
                    px, py = root // n, root % n
                    if 0 <= px < n and 0 <= py < n:
                        uf2.union_right(cx * n + cy, root)
                        if grid[px][py] != '#':
                            if visit[px][py] == math.inf:
                                nex.append((px, py))
                                visit[px][py] = visit[x][y] + 1
                            cx, cy = px, py
                        else:
                            break
                    else:
                        break

                cx, cy = x, y
                while True:
                    root = uf3.find(cx * n + cy)
                    if root == cx * n + cy:
                        root = (cx + 1) * n + cy - 1
                    if not (0 <= (cx + 1) < n and 0 <= cy - 1 < n and 0 <= root < n * n):
                        break
                    px, py = root // n, root % n

                    if 0 <= px < n and 0 <= py < n:
                        uf3.union_right(cx * n + cy, root)
                        if grid[px][py] != '#':
                            if visit[px][py] == math.inf:
                                nex.append((px, py))
                                visit[px][py] = visit[x][y] + 1
                            cx, cy = px, py
                        else:
                            break
                    else:
                        break

                cx, cy = x, y
                while True:
                    root = uf4.find(cx * n + cy)
                    if root == cx * n + cy:
                        root = (cx + 1) * n + cy + 1
                    if not (0 <= (cx + 1) < n and 0 <= cy + 1 < n and 0 <= root < n * n):
                        break
                    px, py = root // n, root % n

                    if 0 <= px < n and 0 <= py < n:
                        uf4.union_right(cx * n + cy, root)
                        if grid[px][py] != '#':
                            if visit[px][py] == math.inf:
                                nex.append((px, py))
                                visit[px][py] = visit[x][y] + 1
                            cx, cy = px, py
                        else:
                            break
                    else:
                        break

            stack = nex
        ac.st(visit[tx][ty] if visit[tx][ty] < math.inf else -1)
        return

    @staticmethod
    def abc_244f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc244/tasks/abc244_f
        tag: bfs|bit_operation|brain_teaser
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints()
            dct[i - 1].append(j - 1)
            dct[j - 1].append(i - 1)
        dis = [[math.inf] * (1 << n) for _ in range(n)]
        for i in range(n):
            dis[i][0] = 0
        stack = []
        for i in range(n):
            dis[i][1 << i] = 1
            stack.append((1 << i, i))
        while stack:
            nex = []
            for s, i in stack:
                d = dis[i][s]
                for j in dct[i]:
                    dj = d + 1
                    if dis[j][s ^ (1 << j)] == math.inf:
                        dis[j][s ^ (1 << j)] = dj
                        nex.append((s ^ (1 << j), j))
            stack = nex
        ac.st(sum(min(dis[i][j] for i in range(n)) for j in range(1 << n)))
        return

    @staticmethod
    def abc_241f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc241/tasks/abc241_f
        tag: bfs|implemention
        """
        m, n, k = ac.read_list_ints()
        sx, sy = ac.read_list_ints()
        gx, gy = ac.read_list_ints()
        points = [ac.read_list_ints() for _ in range(k)]
        dct_x = defaultdict(list)
        dct_y = defaultdict(list)
        obs = {(x, y) for x, y in points}
        for x, y in points:
            for a, b in [(0, 0)]:
                if 1 <= x + a <= m and 1 <= y + b <= n:
                    dct_x[x + a].append(y + b)
                    dct_y[y + b].append(x + a)

        for x in dct_x:
            dct_x[x] = sorted(set(dct_x[x]))
        for y in dct_y:
            dct_y[y] = sorted(set(dct_y[y]))

        dis = defaultdict(lambda: math.inf)
        dis[(sx, sy)] = 0
        stack = [(0, sx, sy)]

        while stack:
            d, i, j = heappop(stack)
            if dis[(i, j)] < d:
                continue
            ind = bisect.bisect_left(dct_y[j], i) - 1
            if 0 <= ind < len(dct_y[j]):
                ii = dct_y[j][ind] + 1

                if (ii, j) not in obs:
                    dj = d + 1
                    if dj < dis[(ii, j)]:
                        dis[(ii, j)] = dj
                        heappush(stack, (dj, ii, j))

            ind = bisect.bisect_right(dct_y[j], i)
            if 0 <= ind < len(dct_y[j]):
                ii = dct_y[j][ind] - 1

                if (ii, j) not in obs:
                    dj = d + 1
                    if dj < dis[(ii, j)]:
                        dis[(ii, j)] = dj
                        heappush(stack, (dj, ii, j))

            ind = bisect.bisect_left(dct_x[i], j) - 1
            if 0 <= ind < len(dct_x[i]):
                jj = dct_x[i][ind] + 1
                if (i, jj) not in obs:
                    dj = d + 1
                    if dj < dis[(i, jj)]:
                        dis[(i, jj)] = dj
                        heappush(stack, (dj, i, jj))

            ind = bisect.bisect_right(dct_x[i], j)
            if 0 <= ind < len(dct_x[i]):
                jj = dct_x[i][ind] - 1
                if (i, jj) not in obs:
                    dj = d + 1
                    if dj < dis[(i, jj)]:
                        dis[(i, jj)] = dj
                        heappush(stack, (dj, i, jj))

        ans = dis[(gx, gy)]
        ac.st(ans if ans < math.inf else -1)
        return

    @staticmethod
    def abc_226c(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc226/tasks/abc226_c
        tag: reverse_graph|bfs
        """
        n = ac.read_int()

        rev = [[] for _ in range(n)]
        ans = [0] * n
        for i in range(n):
            lst = ac.read_list_ints()
            for x in lst[2:]:
                rev[i].append(x - 1)
            ans[i] = lst[0]

        stack = [n - 1]
        res = ans[n - 1]
        ans[n - 1] = -1
        while stack:
            nex = []
            for i in stack:
                for j in rev[i]:
                    if ans[j] != -1:
                        nex.append(j)
                        res += ans[j]
                        ans[j] = -1
            stack = nex[:]
        ac.st(res)
        return

    @staticmethod
    def abc_218f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc218/tasks/abc218_f
        tag: shortest_path|bfs|brute_force|brain_teaser
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for x in range(m):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append((j, x))

        dis = [math.inf] * n
        dis[0] = 0
        stack = [0]
        parent = [-1] * n
        index = [-1] * n
        while stack:
            nex = []
            for i in stack:
                for j, y in dct[i]:
                    if dis[j] == math.inf:
                        dis[j] = dis[i] + 1
                        parent[j] = i
                        index[j] = y
                        nex.append(j)
            stack = nex

        if dis[-1] == math.inf:
            path = set()
        else:
            path = []
            node = n - 1
            while node != 0:
                path.append(index[node])
                node = parent[node]
            path = set(path)
        res = dis[-1] if dis[-1] < math.inf else -1
        for x in range(m):
            if x not in path:
                ac.st(res)
                continue
            dis = [math.inf] * n
            dis[0] = 0
            stack = [0]
            while stack:
                nex = []
                for i in stack:
                    for j, y in dct[i]:
                        if y != x and dis[j] == math.inf:
                            dis[j] = dis[i] + 1
                            nex.append(j)
                stack = nex
            ac.st(dis[-1] if dis[-1] < math.inf else -1)
        return

    @staticmethod
    def abc_211e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc211/tasks/abc211_e
        tag: bfs|classical|not_dfs_back_trace
        """
        n = ac.read_int()
        k = ac.read_int()
        grid = [ac.read_str() for _ in range(n)]
        pre = set()
        for i in range(n):
            for j in range(n):
                if grid[i][j] == ".":
                    pre.add((i * n + j,))
        for _ in range(k - 1):
            cur = set()
            for ls in pre:
                tmp = list(ls)
                dct = set(ls)
                for x in ls:
                    ii, jj = x // n, x % n
                    for a, b in ((ii - 1, jj), (ii + 1, jj), (ii, jj - 1), (ii, jj + 1)):
                        if 0 <= a < n and 0 <= b < n and grid[a][b] == "." and a * n + b not in dct:
                            cur.add(tuple(sorted(tmp + [a * n + b])))
            pre = cur
        ac.st(len(pre))
        return

    @staticmethod
    def abc_209e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc209/tasks/abc209_e
        tag: build_graph|reverse_graph|brain_teaser|game_dp
        """
        n = ac.read_int()
        words = [ac.read_str() for _ in range(n)]

        nodes = set()
        for i, word in enumerate(words):
            nodes.add(word[:3])
            nodes.add(word[-3:])
        nodes = list(nodes)
        k = len(nodes)
        rev = [set() for _ in range(k)]
        ind = {num: i for i, num in enumerate(nodes)}

        for i, word in enumerate(words):
            rev[ind[word[-3:]]].add(ind[word[:3]])

        degree = [0] * k
        for i in range(k):
            for j in rev[i]:
                degree[j] += 1

        stack = [i for i in range(k) if not degree[i]]
        dis = [0] * k
        for i in stack:
            dis[i] = 1

        while stack:
            nex = []
            for i in stack:
                for j in rev[i]:
                    degree[j] -= 1
                    if dis[j] == 0:
                        if dis[i] == 1:
                            dis[j] = -1
                            nex.append(j)
                        elif dis[i] == -1 and degree[j] == 0:
                            dis[j] = 1
                            nex.append(j)
            stack = nex

        for i in range(n):
            cur = dis[ind[words[i][-3:]]]
            if cur == 0:
                ac.st("Draw")
            elif cur == -1:
                ac.st("Aoki")
            else:
                ac.st("Takahashi")
        return

    @staticmethod
    def cf_1063b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1063/B
        tag: bfs|observation|classical
        """
        m, n = ac.read_list_ints()
        r, c = ac.read_list_ints_minus_one()
        x, y = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]

        stack = deque([(r, c, x, y)])
        visit = [-1] * n * m
        visit[r * n + c] = 0
        while stack:
            a, b, xx, yy = stack.pop()
            for i, j in [(a + 1, b), (a - 1, b), (a, b - 1), (a, b + 1)]:
                if 0 <= i < m and 0 <= j < n and grid[i][j] == ".":
                    dij = [xx, yy]
                    if j != b:
                        if j == b - 1:
                            dij[0] -= 1
                        else:
                            dij[1] -= 1
                    if dij[0] < 0 or dij[1] < 0:
                        continue
                    if visit[i * n + j] < dij[0]:
                        visit[i * n + j] = dij[0]
                        if dij[0] == xx:
                            stack.appendleft((i, j, dij[0], dij[1]))
                        else:
                            stack.append((i, j, dij[0], dij[1]))
        ans = sum(x > -1 for x in visit)
        ac.st(ans)
        return

    @staticmethod
    def cf_1344b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1344/problem/B
        tag: bfs|observation
        """
        m, n = ac.read_list_ints()
        grid = [list(ac.read_str()) for _ in range(m)]
        flag1 = 0
        for i in range(m):
            pre = -1
            for j in range(n):
                if grid[i][j] == "#":
                    if pre != -1 and j - pre > 1:
                        ac.st(-1)
                        return
                    pre = j
            if pre == -1:
                flag1 = 1

        flag2 = 0
        for j in range(n):
            pre = -1
            for i in range(m):
                if grid[i][j] == "#":
                    if pre != -1 and i - pre > 1:
                        ac.st(-1)
                        return
                    pre = i
            if pre == -1:
                flag2 = 1
        if flag1 != flag2:
            ac.st(-1)
            return
        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "#":
                    ans += 1
                    stack = [(i, j)]
                    grid[i][j] = "."
                    while stack:
                        x, y = stack.pop()
                        for a, b in [(x - 1, y), (x, y + 1), (x, y - 1), (x + 1, y)]:
                            if 0 <= a < m and 0 <= b < n and grid[a][b] == "#":
                                stack.append((a, b))
                                grid[a][b] = "."
        ac.st(ans)
        return

    @staticmethod
    def cf_877d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/877/D
        tag: bfs|observation|brain_teaser|union_find
        """
        m, n, k = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]
        x1, y1, x2, y2 = ac.read_list_ints_minus_one()
        visit = [-1] * m * n

        visit[x1 * n + y1] = 0
        stack = [(x1, y1)]
        while stack:
            nex = []
            for x, y in stack:
                d = visit[x * n + y] + 1
                for a, b in ac.dire4:
                    xx, yy = x, y
                    for _ in range(k):
                        xx, yy = xx + a, yy + b
                        if 0 <= xx < m and 0 <= yy < n and grid[xx][yy] == "." and visit[xx * n + yy] <= d:
                            if visit[xx * n + yy] == -1:
                                stack.append((xx, yy))
                                visit[xx * n + yy] = d
                            elif visit[xx * n + yy] == d:
                                continue
                            else:
                                break
                        else:
                            break
            stack = nex[:]
        ac.st(visit[x2 * n + y2])
        return

    @staticmethod
    def cf_987d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/987/problem/D
        tag: several_source|bfs|brute_force
        """
        n, m, k, s = ac.read_list_ints()
        a = ac.read_list_ints_minus_one()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        start = [[] for _ in range(k)]
        for i in range(n):
            start[a[i]].append(i)

        res = [[] for _ in range(n)]
        dis = [math.inf] * n
        for c in range(k):
            for i in range(n):
                dis[i] = math.inf
            stack = start[c][:]
            for i in stack:
                dis[i] = 0
            while stack:
                nex = []
                for i in stack:
                    for j in dct[i]:
                        if dis[j] == math.inf:
                            dis[j] = dis[i] + 1
                            nex.append(j)
                stack = nex
            for i in range(n):
                res[i].append(dis[i])
        ans = []
        for i in range(n):
            res[i].sort()
            ans.append(sum(res[i][:s]))
        ac.lst(ans)
        return

    @staticmethod
    def cf_1349c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1349/C
        tag: bfs|observation|implemention
        """
        m, n, q = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]

        root = [[-1] * n for _ in range(m)]
        visit = [[-1] * n for _ in range(m)]
        stack = []
        for i in range(m):
            for j in range(n):
                if i + 1 < m and grid[i + 1][j] == grid[i][j]:
                    root[i][j] = root[i + 1][j] = int(grid[i][j])
                    visit[i][j] = visit[i + 1][j] = 0
                if j + 1 < n and grid[i][j + 1] == grid[i][j]:
                    root[i][j] = root[i][j + 1] = int(grid[i][j])
                    visit[i][j] = visit[i][j + 1] = 0
                if visit[i][j] != -1:
                    stack.append((i, j))
        while stack:
            nex = []
            for i, j in stack:
                for x, y in ac.dire4:
                    if 0 <= i + x < m and 0 <= j + y < n and visit[i + x][j + y] == -1:
                        visit[i + x][j + y] = visit[i][j] + 1
                        root[i + x][j + y] = root[i][j]
                        nex.append((i + x, j + y))
            stack = nex
        for _ in range(q):
            i, j, p = ac.read_list_ints_minus_one()
            p += 1
            if visit[i][j] == 0:
                ac.st(int(grid[i][j]) ^ (p & 1))
            elif visit[i][j] == -1:
                ac.st(int(grid[i][j]))
            else:
                if p < visit[i][j]:
                    ac.st(int(grid[i][j]))
                else:
                    ac.st(int(root[i][j]) ^ (p & 1))
        return

    @staticmethod
    def cf_1276b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1276/B
        tag: bfs|unweighted_graph|multiplication_method
        """

        class Graph(WeightedGraphForDijkstra):
            def bfs(self, src=0, target=0):
                dis = [math.inf] * (self.n + 1)
                dis[src] = 0
                stack = [src]
                while stack:
                    nex = []
                    for u in stack:
                        i = self.point_head[u]
                        while i:
                            j = self.edge_to[i]
                            dj = dis[u] + 1
                            if dj < dis[j]:
                                dis[j] = dj
                                if j != target:
                                    nex.append(j)
                            i = self.edge_next[i]
                    stack = nex
                return sum(x < math.inf for x in dis)

        for _ in range(ac.read_int()):
            n, m, a, b = ac.read_list_ints()
            a -= 1
            b -= 1
            graph = Graph(n)
            for _ in range(m):
                u, v = ac.read_list_ints_minus_one()
                graph.add_undirected_edge(u, v)
            ans = (n - graph.bfs(a, b)) * (n - graph.bfs(b, a))
            ac.st(ans)
        return
