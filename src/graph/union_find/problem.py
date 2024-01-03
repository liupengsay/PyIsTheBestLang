"""

Algorithm：union_find|persistent_union_find|permutation_circle
Description：graph|reverse_thinking|permutation_circle|offline_query|merge_wise|root_wise

====================================LeetCode====================================
765（https://leetcode.cn/problems/couples-holding-hands/）union_find
1697（https://leetcode.cn/problems/checking-existence-of-edge-length-limited-paths/）sort|offline_query|implemention
2503（https://leetcode.cn/problems/maximum-number-of-points-from-grid-queries/）sort|offline_query|implemention
2421（https://leetcode.cn/problems/number-of-good-paths/）sort|union_find|counter
2382（https://leetcode.cn/problems/maximum-segment-sum-after-removals/）reverse_order|union_find|implemention|reverse_thinking
2334（https://leetcode.cn/problems/subarray-with-elements-greater-than-varying-threshold/）sort|brute_force|union_find
2158（https://leetcode.cn/problems/amount-of-new-area-painted-each-day/）union_find_left_root
2157（https://leetcode.cn/problems/groups-of-strings/）alphabet|brute_force
2076（https://leetcode.cn/problems/process-restricted-friend-requests/）union_find|reverse_thinking
2459（https://leetcode.cn/problems/sort-array-by-moving-items-to-empty-space/）permutation_circle
2709（https://leetcode.cn/problems/greatest-common-divisor-traversal/）union_find|prime_factorization
2612（https://leetcode.cn/problems/minimum-reverse-operations/）union_find|find_range_merge_to_disjoint
1559（https://leetcode.cn/problems/detect-cycles-in-2d-grid/）union_find|circle_judge|classical
1970（https://leetcode.cn/problems/last-day-where-you-can-still-cross/）reverse_thinking|union_find
1998（https://leetcode.cn/problems/gcd-sort-of-an-array/）union_find|prime_factorization
2158（https://leetcode.cn/problems/amount-of-new-area-painted-each-day/）union_find_range|union_find_left_root|union_find_right_root
2471（https://leetcode.cn/problems/minimum-number-of-operations-to-sort-a-binary-tree-by-level/description/）discretization|permutation_circle
945（https://leetcode.cn/problems/minimum-increment-to-make-array-unique/description/）union_find_right_root|greedy
947（https://leetcode.cn/contest/weekly-contest-112/problems/most-stones-removed-with-same-row-or-column/）brain_teaser|union_find

=====================================LuoGu======================================
P3367（https://www.luogu.com.cn/problem/P3367）connected_part|counter|union_find
P5836（https://www.luogu.com.cn/problem/P5836）union_find|several_union_find
P3144（https://www.luogu.com.cn/problem/P3144）reverse_order|union_find|connected_part|counter
P5836（https://www.luogu.com.cn/problem/P5836）union_find|several_union_find
P5877（https://www.luogu.com.cn/problem/P5877）union_find|implemention|counter
P6111（https://www.luogu.com.cn/problem/P6111）union_find|offline_query
P6121（https://www.luogu.com.cn/problem/P6121）reverse_order|union_find|size
P6153（https://www.luogu.com.cn/problem/P6153）union_find|greedy|classical
P1955（https://www.luogu.com.cn/problem/P1955）union_find
P1196（https://www.luogu.com.cn/problem/P1196）union_find_weighted
P1197（https://www.luogu.com.cn/problem/P1197）reverse_order|union_find，reverse_order|brute_force|part
P1522（https://www.luogu.com.cn/problem/P1522）connected_part|brute_force|high_precision|tree_diameter
P1621（https://www.luogu.com.cn/problem/P1621）euler_series|O(nlogn)|prime_factorization
P1892（https://www.luogu.com.cn/problem/P1892）union_find|bipartite_graph
P2189（https://www.luogu.com.cn/problem/P2189）union_find
P2307（https://www.luogu.com.cn/problem/P2307）union_find
P3420（https://www.luogu.com.cn/problem/P3420）union_find
P5429（https://www.luogu.com.cn/problem/P5429）union_find
P6004（https://www.luogu.com.cn/problem/P6004）union_find|permutation_circle|kruskal|mst|greedy|pointer
P6193（https://www.luogu.com.cn/problem/P6193）permutation_circle
P6706（https://www.luogu.com.cn/problem/P6706）directed_graph|union_find|reverse_order|find_range_merge_to_disjoint
P7991（https://www.luogu.com.cn/problem/P7991）union_find|shrink_point
P8230（https://www.luogu.com.cn/problem/P8230）layer|union_find|implemention
P8637（https://www.luogu.com.cn/problem/P8637）union_find|permutation_circle
P8686（https://www.luogu.com.cn/problem/P8686）union_find
P8785（https://www.luogu.com.cn/problem/P8785）union_find|counter
P8787（https://www.luogu.com.cn/problem/P8787）greedy|heapq|implemention|union_find
P8881（https://www.luogu.com.cn/problem/P8881）brain_teaser|union_find|circle_judge|part

===================================CodeForces===================================
25D（https://codeforces.com/problemset/problem/25/D）union_find
1810E（https://codeforces.com/contest/1810/problem/E）union_find|heuristic_search|bfs|heapq
920E（https://codeforces.com/contest/920/problem/E）union_find
540C（https://codeforces.com/problemset/problem/540/C）union_find
1800E2（https://codeforces.com/problemset/problem/1800/E2）union_find
1691E（https://codeforces.com/contest/1691/problem/E）union_find_range
827A（https://codeforces.com/problemset/problem/827/A）union_find_right_root|implemention|greedy

====================================AtCoder=====================================
ARC065B（https://atcoder.jp/contests/abc049/tasks/arc065_b）union_find|several_union_find
ABC126E（https://atcoder.jp/contests/abc126/tasks/abc126_e）union_find|several_union_find
ABC131F（https://atcoder.jp/contests/abc131/tasks/abc131_f）brain_teaser|union_find|counter

=====================================AcWing=====================================
4309（https://www.acwing.com/problem/content/description/4309/）union_find_right_range
4869（https://www.acwing.com/problem/content/description/4869/）union_find|implemention|size
5148（https://www.acwing.com/problem/content/5148/）union_find|circle_judge

================================LibraryChecker================================
1 （https://judge.yosupo.jp/problem/cycle_detection_undirected）union_find|circle_judge

"""
import decimal
import math
from collections import defaultdict, Counter, deque
from heapq import heappop, heapify, heappush
from typing import List, Optional

from src.basis.tree_node.template import TreeNode
from src.data_structure.sorted_list.template import SortedList
from src.graph.dijkstra.template import Dijkstra
from src.graph.union_find.template import UnionFind, UnionFindWeighted
from src.mathmatics.number_theory.template import NumberTheory
from src.mathmatics.prime_factor.template import PrimeFactor
from src.utils.fast_io import FastIO
from src.utils.fast_io import inf


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1810e_1(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1810/problem/E
        tag: cannot_be_union_find|heuristic_search|bfs|heapq|classical|hard
        """
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            nums = ac.read_list_ints()
            edge = [[] for _ in range(n)]
            for _ in range(m):
                u, v = ac.read_list_ints_minus_one()
                edge[u].append(v)
                edge[v].append(u)

            visit = [-1] * n
            ans = "NO"
            for i in range(n):
                if visit[i] == -1 and not nums[i]:
                    count = 0
                    visit[i] = i
                    stack = [(0, i)]
                    while stack:
                        d, x = heappop(stack)
                        if count < nums[x]:
                            break
                        count += 1
                        for j in edge[x]:
                            if visit[j] != i:
                                visit[j] = i
                                heappush(stack, (nums[j], j))
                    if count == n:
                        ans = "YES"
                        break
            ac.st(ans)

        return

    @staticmethod
    def cf_1810e_2(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1810/problem/E
        tag: cannot_be_union_find|heuristic_search|bfs|heapq|classical|hard|mst
        """
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            nums = ac.read_list_ints()
            nums = [num * n + i for i, num in enumerate(nums)]  # classical skills

            edge = [[] for _ in range(n)]
            for _ in range(m):
                u, v = ac.read_list_ints_minus_one()
                if nums[u] < nums[v]:
                    u, v = v, u
                edge[u].append(v)

            reach = [0] * n
            uf = UnionFind(n)
            nums.sort()
            for val in nums:
                num, i = val // n, val % n
                if not num:
                    reach[i] = 1
                for j in edge[i]:
                    if reach[uf.find(j)] and num <= uf.size(j):
                        reach[i] = 1
                    uf.union_left(i, j)
            ac.st("YES" if uf.part == 1 and reach[uf.find(0)] else "NO")
        return

    @staticmethod
    def ac_5148(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/5148/
        tag: union_find|circle_judge
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]
        uf = UnionFind(m * n)
        for i in range(m):
            for j in range(n):
                for x, y in [[i, j + 1], [i + 1, j]]:
                    if 0 <= x < m and 0 <= y < n and grid[x][y] == grid[i][j]:
                        if not uf.union(i * n + j, x * n + y):
                            ac.st("Yes")
                            return
        ac.st("No")
        return

    @staticmethod
    def cf_872a(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/827/A
        tag: union_find_right_root|implemention|greedy
        """
        n = ac.read_int()
        nums = [ac.read_list_strs() for _ in range(n)]
        length = max(int(ls[-1]) + len(ls[0]) - 1 for ls in nums)
        uf = UnionFind(length + 1)
        ans = ["a"] * length
        for ls in nums:
            st = ls[0]
            m = len(st)
            for x in ls[2:]:
                x = int(x) - 1
                start = x
                while x <= start + m - 1:
                    x = uf.find(x)
                    if x <= start + m - 1:
                        ans[x] = st[x - start]
                        uf.union_right(x, x + 1)
                    else:
                        break
        ac.st("".join(ans))
        return

    @staticmethod
    def cf_920e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/920/problem/E
        tag: union_find
        """
        n, m = ac.read_list_ints()
        edge = set()
        for _ in range(m):
            u, v = ac.read_list_ints_minus_one()
            edge.add((u, v))

        ans = []
        not_visit = set(range(n))
        while not_visit:
            i = next(iter(not_visit))
            not_visit.discard(i)
            stack = [i]
            cnt = 1
            while stack:
                u = stack.pop()
                visit = []
                for v in not_visit:
                    if (u, v) in edge or (v, u) in edge:
                        continue
                    cnt += 1
                    stack.append(v)
                    visit.append(v)
                for v in visit:
                    not_visit.discard(v)
            ans.append(cnt)
        ans.sort()
        ac.st(len(ans))
        ac.lst(ans)
        return

    @staticmethod
    def lc_1697(n: int, edge_list: List[List[int]], queries: List[List[int]]) -> List[bool]:
        """
        url: https://leetcode.cn/problems/checking-existence-of-edge-length-limited-paths/
        tag: sort|offline_query|implemention|classical
        """
        m = len(queries)
        ind = list(range(m))
        ind.sort(key=lambda x: queries[x][2])

        edge_list.sort(key=lambda x: x[2])
        uf = UnionFind(n)
        i = 0
        k = len(edge_list)
        ans = []
        for j in ind:
            p, q, limit = queries[j]
            while i < k and edge_list[i][2] < limit:
                uf.union(edge_list[i][0], edge_list[i][1])
                i += 1
            ans.append([j, uf.is_connected(p, q)])

        ans.sort(key=lambda x: x[0])
        return [an[1] for an in ans]

    @staticmethod
    def lc_2503(grid: List[List[int]], queries: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/maximum-number-of-points-from-grid-queries/
        tag: sort|offline_query|implemention
        """
        dct = []
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if i + 1 < m:
                    x, y = grid[i][j], grid[i + 1][j]
                    dct.append([i * n + j, i * n + n + j, x if x > y else y])
                if j + 1 < n:
                    x, y = grid[i][j], grid[i][j + 1]
                    dct.append([i * n + j, i * n + 1 + j, x if x > y else y])
        dct.sort(key=lambda d: d[2])
        uf = UnionFind(m * n)
        k = len(queries)
        ind = list(range(k))
        ind.sort(key=lambda d: queries[d])

        ans = [0] * k
        j = 0
        length = len(dct)
        for i in ind:
            cur = queries[i]
            while j < length and dct[j][2] < cur:
                uf.union(dct[j][0], dct[j][1])
                j += 1
            if cur > grid[0][0]:
                ans[i] = uf.size(0)
        return ans

    @staticmethod
    def lc_2421(vals: List[int], edges: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/number-of-good-paths/
        tag: sort|union_find|counter|classical|discretization
        """
        n = len(vals)
        index = defaultdict(list)
        for i in range(n):
            index[vals[i]].append(i)
        edges.sort(key=lambda x: max(vals[x[0]], vals[x[1]]))
        uf = UnionFind(n)
        i = 0
        m = len(edges)
        ans = 0
        for val in sorted(index):
            while i < m and vals[edges[i][0]] <= val and vals[edges[i][1]] <= val:
                uf.union(edges[i][0], edges[i][1])
                i += 1
            cnt = Counter(uf.find(x) for x in index[val])
            for w in cnt.values():
                ans += w * (w - 1) // 2 + w
        return ans

    @staticmethod
    def library_check_1(ac=FastIO()):
        """
        url: https://judge.yosupo.jp/problem/cycle_detection_undirected
        tag: union_find|circle_judge
        """
        n, m = ac.read_list_ints()
        edges = [ac.read_list_ints() for i in range(m)]
        uf = UnionFind(n)
        dct = [[] for _ in range(n)]
        for i in range(m):
            u, v = edges[i]
            if not uf.union(u, v):
                stack = [(u, -1)]
                parent = [(-1, -1) for _ in range(n)]
                while stack:
                    x, fa = stack.pop()
                    for y, ind in dct[x]:
                        if y != fa:
                            parent[y] = (x, ind)
                            stack.append((y, x))
                nodes = [v]
                edges = []
                while nodes[-1] != u:
                    edges.append(parent[nodes[-1]][1])
                    nodes.append(parent[nodes[-1]][0])
                edges.append(i)
                ac.st(len(nodes))
                ac.lst(nodes)
                ac.lst(edges)
                break
            dct[u].append((v, i))
            dct[v].append((u, i))
        else:
            ac.st(-1)
        return

    @staticmethod
    def lg_p1196(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1196
        tag: union_find_weighted|classical|hard
        """
        n = 10
        uf = UnionFindWeighted(n)
        for _ in range(ac.read_int()):
            lst = ac.read_list_strs()
            i, j = [int(w) - 1 for w in lst[1:]]
            while i >= n or j >= n:
                uf.root_or_size.append(-1)
                uf.front.append(0)
                n += 1
            if lst[0] == "M":
                uf.union(i, j)
            else:
                root_x = uf.find(i)
                root_y = uf.find(j)
                if root_x != root_y:
                    ac.st(-1)
                else:
                    ac.st(abs(uf.front[i] - uf.front[j]) - 1)
        return

    @staticmethod
    def lg_p1197(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1197
        tag: reverse_order|union_find，reverse_order|brute_force|part
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints()
            dct[i].append(j)
            dct[j].append(i)
        k = ac.read_int()
        rem = [ac.read_int() for _ in range(k)]
        out = set(rem)
        uf = UnionFind(n)
        for i in range(n):
            if i not in out:
                for j in dct[i]:
                    if j not in out:
                        uf.union(i, j)
        ans = []
        for i in range(k - 1, -1, -1):
            ans.append(uf.part - i - 1)
            out.discard(rem[i])
            for j in dct[rem[i]]:
                if j not in out:
                    uf.union(rem[i], j)
        ans.append(uf.part)
        ans.reverse()
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lg_p1522(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1522
        tag: connected_part|brute_force|high_precision|tree_diameter|classical|hard
        """
        def dis(x1, y1, x2, y2):
            return math.sqrt(decimal.Decimal(((x1 - x2) ** 2 + (y1 - y2) ** 2)))

        n = ac.read_int()
        nums = [[w for w in ac.read_list_ints()] for _ in range(n)]
        grid = [ac.read_str() for _ in range(n)]
        dct = [dict() for _ in range(n)]
        uf = UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                if grid[i][j] == "1":
                    uf.union(i, j)
                    d = dis(nums[i][0], nums[i][1], nums[j][0], nums[j][1])
                    dct[i][j] = dct[j][i] = d

        dist = []
        for i in range(n):
            dist.append(Dijkstra().get_shortest_path(dct, i))

        part = uf.get_root_part()
        fast = [inf] * n
        group = dict()
        for p in part:
            for i in part[p]:
                fast[i] = max(dist[i][j] for j in part[p])
            group[p] = max(fast[i] for i in part[p])

        ans = inf
        for i in range(n):
            for j in range(n):
                if not uf.is_connected(i, j):
                    cur = dis(nums[i][0], nums[i][1], nums[j][0], nums[j][1]) + fast[i] + fast[j]
                    cur = ac.max(cur, group[uf.find(i)])
                    cur = ac.max(cur, group[uf.find(j)])
                    ans = ac.min(ans, cur)
        ac.st("%.6f" % ans)
        return

    @staticmethod
    def lg_p1621(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1621
        tag: euler_series|O(nlogn)|prime_factorization|classical
        """
        a, b, p = ac.read_list_ints()
        nums = list(range(a, b + 1))
        ind = {num: num - a for num in nums}
        primes = [x for x in NumberTheory().sieve_of_eratosthenes(b) if x >= p]

        uf = UnionFind(b - a + 1)
        for x in primes:
            lst = []
            y = x
            while y <= b:
                if y in ind:
                    lst.append(ind[y])
                y += x
            m = len(lst)
            for j in range(m - 1):
                uf.union(lst[j], lst[j + 1])
        ac.st(uf.part)
        return

    @staticmethod
    def lg_p1892(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1892
        tag: union_find|bipartite_graph
        """
        n = ac.read_int()
        m = ac.read_int()
        uf = UnionFind(n)
        dct = dict()
        for _ in range(m):
            lst = ac.read_list_strs()
            a, b = int(lst[1]), int(lst[2])
            a -= 1
            b -= 1
            if lst[0] == "E":
                if a in dct:
                    uf.union(dct[a], b)
                if b in dct:
                    uf.union(dct[b], a)
                dct[a] = b
                dct[b] = a
            else:
                uf.union(a, b)
        ac.st(uf.part)
        return

    @staticmethod
    def lg_p1955(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1955
        tag: union_find|discretization
        """
        t = ac.read_int()
        for _ in range(t):
            n = ac.read_int()
            ind = dict()
            uf = UnionFind(n * 2)
            res = []
            for _ in range(n):
                lst = ac.read_list_ints()
                while not lst:
                    lst = ac.read_list_ints()
                i, j, e = lst
                if i not in ind:
                    ind[i] = len(ind)
                if j not in ind:
                    ind[j] = len(ind)
                if e == 1:
                    uf.union(ind[i], ind[j])
                else:
                    res.append((ind[i], ind[j]))
            if any(uf.is_connected(i, j) for i, j in res):
                ac.st("NO")
            else:
                ac.st("YES")
        return

    @staticmethod
    def lg_p2189(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2189
        tag: union_find|classical|preprocess|hard
        """
        n, m, k, q = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)

        for _ in range(q):
            order = ac.read_list_ints_minus_one()
            uf = UnionFind(n)
            visit = [0] * n
            for i in order:
                visit[i] = 1

            ans = True
            pre = order[0]
            for i in range(n):
                if not visit[i]:
                    for j in dct[i]:
                        if not visit[j]:
                            uf.union(i, j)

            for i in order:
                visit[i] = 0
                for j in dct[i]:
                    if not visit[j]:
                        uf.union(i, j)
                if not uf.is_connected(i, pre):
                    ans = False
                    break
                pre = i
            ac.st("Yes" if ans else "No")
        return

    @staticmethod
    def lg_p2307(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2307
        tag: union_find
        """
        while True:
            ans = []
            while True:
                lst = ac.read_list_ints()
                if not lst:
                    break
                ans.extend(lst)
            if ans == [-1, -1]:
                break
            ans = ans[:-2]
            nodes = list(set(ans))
            ind = {num: i for i, num in enumerate(nodes)}
            m = len(ind)
            uf = UnionFind(m)
            res = True
            for i in range(0, len(ans), 2):
                a, b = ind[ans[i]], ind[ans[i + 1]]
                if not uf.union(a, b):
                    res = False
                    break
            ac.st(1 if res and uf.part == 1 else 0)
        return

    @staticmethod
    def lg_p3420(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3420
        tag: union_find
        """
        n = ac.read_int()
        uf = UnionFind(n)
        for i in range(n):
            j = ac.read_int()
            uf.union(i, j - 1)
        ac.st(uf.part)
        return

    @staticmethod
    def lg_p6004(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6004
        tag: union_find|permutation_circle|kruskal|mst|greedy|pointer
        """
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints_minus_one()
        edges = [ac.read_list_ints() for _ in range(m)]
        edges.sort(key=lambda it: -it[2])
        uf = UnionFind(n)
        ind = 0
        ans = -1
        for i, j, w in edges:
            while ind < n and uf.is_connected(ind, nums[ind]):
                ind += 1
            if ind == n:
                ac.st(ans)
                return
            uf.union(i - 1, j - 1)
            ans = w
        ac.st(ans)
        return

    @staticmethod
    def lg_p6193(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6193
        tag: permutation_circle|discretization|brain_teaser|classical|hard
        """
        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]
        lst = sorted(nums)
        ind = {num: i for i, num in enumerate(lst)}
        uf = UnionFind(n)
        x = lst[0]
        for i in range(n):
            uf.union(i, ind[nums[i]])
        part = uf.get_root_part()
        ans = 0
        for p in part:
            y = min(lst[i] for i in part[p])
            s = sum(lst[i] for i in part[p])
            m = len(part[p])
            if m == 1:
                continue
            cost1 = s + (m - 2) * y
            cost2 = s - y + x + (m - 2) * x + (x + y) * 2
            ans += ac.min(cost1, cost2)
        ac.st(ans)
        return

    @staticmethod
    def lc_2709(nums: List[int]) -> bool:
        """
        url: https://leetcode.cn/problems/greatest-common-divisor-traversal/
        tag: union_find|prime_factorization
        """
        pf = PrimeFactor(10 ** 5)
        n = len(nums)
        uf = UnionFind(n)
        pre = dict()
        for i in range(n):
            for num, _ in pf.prime_factor[nums[i]]:
                if num in pre:
                    uf.union(i, pre[num])
                else:
                    pre[num] = i
        return uf.part == 1

    @staticmethod
    def lg_p6706(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6706
        tag: directed_graph|union_find|reverse_order|find_range_merge_to_disjoint|classical|hard
        """
        n = ac.read_int()
        edge = ac.read_list_ints()
        q = ac.read_int()
        query = []
        for _ in range(q):
            op, x = ac.read_list_ints()
            if op == 2:
                edge[x - 1] = -edge[x - 1]
            query.append(op * (n + 1) + x)
        uf = UnionFind(n + 1)
        for i in range(1, n + 1):
            if edge[i - 1] > 0:
                j = uf.find(edge[i - 1])
                if j == i:
                    uf.union_left(0, i)
                    uf.union_left(0, j)
                else:
                    uf.union_right(i, j)

        ans = []
        for i in range(q - 1, -1, -1):
            op, x = query[i] // (n + 1), query[i] % (n + 1)
            if op == 1:
                y = uf.find(x)
                if y == 0:
                    res = "CIKLUS"
                else:
                    res = y
                ans.append(res)
            else:
                j = uf.find(-edge[x - 1])
                if j == x:
                    uf.union_left(0, x)
                    uf.union_left(0, j)
                else:
                    uf.union_right(x, j)
        for i in range(len(ans) - 1, -1, -1):
            ac.st(ans[i])
        return

    @staticmethod
    def lg_p7991(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P7991
        tag: union_find|shrink_point
        """
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            uf = UnionFind(n)
            for _ in range(m):
                i, j = ac.read_list_ints_minus_one()
                uf.union(i, j)
            if uf.is_connected(0, n - 1):
                ac.st(0)
                continue

            dis_0 = [n] * n
            dis_1 = [n] * n

            pre_0 = pre_1 = -1
            for i in range(n):
                if uf.is_connected(0, i):
                    pre_0 = i
                if uf.is_connected(n - 1, i):
                    pre_1 = i
                if pre_0 != -1:
                    dis_0[uf.find(i)] = ac.min(dis_0[uf.find(i)], i - pre_0)
                if pre_1 != -1:
                    dis_1[uf.find(i)] = ac.min(dis_1[uf.find(i)], i - pre_1)

            pre_0 = pre_1 = -1
            for i in range(n - 1, -1, -1):
                if uf.is_connected(0, i):
                    pre_0 = i
                if uf.is_connected(n - 1, i):
                    pre_1 = i
                if pre_0 != -1:
                    dis_0[uf.find(i)] = ac.min(dis_0[uf.find(i)], pre_0 - i)
                if pre_1 != -1:
                    dis_1[uf.find(i)] = ac.min(dis_1[uf.find(i)], pre_1 - i)
            ans = min(dis_0[i] * dis_0[i] + dis_1[i] * dis_1[i] for i in range(n))
            ac.st(ans)
        return

    @staticmethod
    def lc_2612(n: int, p: int, banned: List[int], k: int) -> List[int]:
        """
        url: https://leetcode.cn/problems/minimum-reverse-operations/
        tag: union_find|find_range_merge_to_disjoint|classical|hard|odd_even|bfs|brain_teaser
        """
        ans = [-1] * n
        uf = UnionFind(n + 2)
        for i in banned:
            uf.union_right(i, i + 2)
        stack = deque([p])
        ans[p] = 0
        while stack:
            i = stack.popleft()
            low = max(0, k - 1 - i, i - k + 1)
            high = min(2 * n - k - 1 - i, n - 1, i + k - 1)
            j = uf.find(low)
            while j <= high:
                if ans[j] == -1:
                    ans[j] = ans[i] + 1
                    stack.append(j)
                    uf.union_right(j, j + 2)
                j = uf.find(j + 2)
        return ans

    @staticmethod
    def lg_p8230(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8230
        tag: layer|union_find|implemention|mst|brain_teaser
        """
        k, m, n = ac.read_list_ints()
        ans = 1
        start = [0, 0]
        uf = UnionFind(m * n)
        for _ in range(k):

            lst = []
            end = [-1, -1]
            for i in range(m * n):
                uf.root_or_size[i] = -1
            pre = [-9] * n
            for i in range(m):
                cur = ac.read_list_ints()
                for j in range(n):
                    w = cur[j]
                    if w != -9:
                        lst.append(w * m * n + i * n + j)
                        x = i - 1
                        if 0 <= x < m and pre[j] != -9:
                            uf.union(i * n + j, x * n + j)

                        y = j - 1
                        if 0 <= y < n and cur[y] != -9:
                            uf.union(i * n + j, i * n + y)

                    if w == -1:
                        end = [i, j]
                pre = cur[:]
            lst.sort()

            for num in lst:
                val, i, j = num // m // n, (num % (m * n)) // n, (num % (m * n)) % n
                if val > ans:
                    break
                if uf.is_connected(start[0] * n + start[1], i * n + j):
                    if ans >= val:
                        if val > 0:
                            ans += val
            start = end[:]
        ac.st(ans)
        return

    @staticmethod
    def lg_p8686(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8686
        tag: union_find|classical
        """
        ac.read_int()
        nums = ac.read_list_ints()
        dct = dict()
        ans = []
        for num in nums:
            pre = num
            while pre in dct:
                pre = dct[pre]

            x = num
            while x in dct:
                dct[x], x = pre + 1, dct[x]

            dct[pre] = pre + 1
            ans.append(pre)
        ac.lst(ans)
        return

    @staticmethod
    def lg_p8787_1(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8787
        tag: greedy|heapq|implemention|union_find|classical|hard

        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        stack = [(-nums[i], -i) for i in range(n)]
        heapify(stack)
        uf = UnionFind(n)
        for i in range(n):
            if i and nums[i] == nums[i - 1]:
                uf.union_left(i - 1, i)
        ans = 0
        while stack:
            val, i = heappop(stack)
            val, i = -val, -i
            if val == 1:
                break
            if i != uf.find(i):
                continue
            if i and nums[uf.find(i - 1)] == val:
                uf.union_left(i - 1, i)
                continue
            ans += 1
            val = int(((val // 2) + 1) ** 0.5)
            nums[i] = val
            heappush(stack, (-val, -i))
        ac.st(ans)
        return

    @staticmethod
    def lg_p8787_2(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8787
        tag: greedy|heapq|implemention|union_find|classical|hard

        """
        ac.read_int()
        nums = ac.read_list_ints()
        pre = set()
        ans = 0
        for num in nums:
            cur = set()
            while num > 1:
                if num not in pre:
                    ans += 1
                cur.add(num)
                num = int(((num // 2) + 1) ** 0.5)
            pre = cur
        ac.st(ans)
        return

    @staticmethod
    def lg_p8881(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8881
        tag: brain_teaser|union_find|circle_judge|part|classical
        """
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            uf = UnionFind(n)
            edge = []
            for _ in range(m):
                i, j = ac.read_list_ints_minus_one()
                uf.union(i, j)
                edge.append((i, j))
            cnt = 0
            for i, j in edge:
                if uf.is_connected(0, i):
                    cnt += 1
            ac.st("1.000" if uf.size(0) == cnt + 1 else "0.000")
        return

    @staticmethod
    def lc_945_1(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-increment-to-make-array-unique/description/
        tag: union_find_right_root|greedy
        """
        nums.sort()
        ans = 0
        uf = UnionFind(max(nums) + len(nums) + 2)
        for num in nums:
            x = uf.find(num)
            ans += x - num
            uf.union_right(x, x + 1)
        return ans

    @staticmethod
    def lc_945_2(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-increment-to-make-array-unique/description/
        tag: union_find_right_root|greedy
        """
        nums.sort()
        ans = 0
        pre = -1
        for num in nums:
            if num > pre:
                pre = num
            else:
                ans += pre + 1 - num
                pre += 1
        return ans

    @staticmethod
    def lc_1559(grid: List[List[str]]) -> bool:
        """
        url: https://leetcode.cn/problems/detect-cycles-in-2d-grid/
        tag: union_find|circle_judge|classical
        """
        m, n = len(grid), len(grid[0])
        uf = UnionFind(m * n)
        for i in range(m):
            for j in range(n):
                if i + 1 < m and grid[i + 1][j] == grid[i][j]:
                    if not uf.union(i * n + j, i * n + n + j):
                        return True
                if j + 1 < n and grid[i][j + 1] == grid[i][j]:
                    if not uf.union(i * n + j, i * n + j + 1):
                        return True
        return False

    @staticmethod
    def lc_2158(paint: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/amount-of-new-area-painted-each-day/
        tag: union_find_range|union_find_left_root|union_find_right_root
        """
        m = max(ls[1] for ls in paint) + 10
        uf = UnionFind(m)
        ans = []
        for a, b in paint:
            cnt = 0
            a = uf.find(a)
            while a < b:
                cnt += 1
                uf.union_right(a, a + 1)
                a = uf.find(a + 1)
            ans.append(cnt)
        return ans

    @staticmethod
    def abc_065b(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc049/tasks/arc065_b
        tag: union_find|several_union_find
        """
        n, k, ll = ac.read_list_ints()
        ufa = UnionFind(n)
        for _ in range(k):
            p, q = ac.read_list_ints_minus_one()
            ufa.union(p, q)

        ufb = UnionFind(n)
        for _ in range(ll):
            p, q = ac.read_list_ints_minus_one()
            ufb.union(p, q)
        pre = defaultdict(int)
        for i in range(n):
            pre[(ufa.find(i), ufb.find(i))] += 1
        ans = [pre[(ufa.find(i), ufb.find(i))] for i in range(n)]
        ac.lst(ans)
        return

    @staticmethod
    def abc_131f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc131/tasks/abc131_f
        tag: brain_teaser|union_find|counter|hard|classical
        """
        n = ac.read_int()
        m = 10 ** 5
        uf = UnionFind(2 * m)
        for _ in range(n):
            x, y = ac.read_list_ints_minus_one()
            y += m
            uf.union(x, y)
        group = uf.get_root_part()
        ans = 0
        for g in group:
            x = sum(xx < m for xx in group[g])
            y = sum(xx >= m for xx in group[g])
            ans += x * y
        ac.st(ans - n)
        return

    @staticmethod
    def ac_4309(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4309/
        tag: union_find_right_range|greedy
        """
        n = ac.read_int()
        a = ac.read_list_ints()
        uf = UnionFind(n * 2 + 2)
        a.sort()
        ans = 0
        for num in a:
            x = uf.find(num)
            ans += x - num
            uf.union_right(x, x + 1)
        ac.st(ans)
        return

    @staticmethod
    def ac_4869(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4869/
        tag: union_find|implemention|size|classical
        """
        n, d = ac.read_list_ints()
        uf = UnionFind(n)
        lst = SortedList([1] * n)
        pre = ans = 1
        for i in range(d):
            x, y = ac.read_list_ints_minus_one()
            if uf.is_connected(x, y):
                pre += 1
                if len(lst) - pre >= 0:
                    ans += lst[len(lst) - pre]
            else:
                for w in [x, y]:
                    i = lst.bisect_left(uf.size(w))
                    if i >= len(lst) - pre:
                        ans -= lst.pop(i)
                        if len(lst) - pre >= 0:
                            ans += lst[len(lst) - pre]
                    else:
                        lst.pop(i)

                uf.union(x, y)
                lst.add(uf.size(x))
                i = lst.bisect_right(uf.size(x))
                if i >= len(lst) - pre:
                    ans += uf.size(x)
                    if len(lst) - pre - 1 >= 0:
                        ans -= lst[len(lst) - pre - 1]
            ac.st(ans - 1)
        return

    @staticmethod
    def lc_2471(root: Optional[TreeNode]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-number-of-operations-to-sort-a-binary-tree-by-level/description/
        tag: discretization|permutation_circle|classical
        """
        def check():
            nonlocal ans
            ind = {num: i for i, num in enumerate(cur)}
            lst = sorted(cur)
            m = len(lst)
            uf = UnionFind(m)
            for i in range(m):
                uf.union(ind[lst[i]], ind[cur[i]])
            group = uf.get_root_size()
            for p in group:
                ans += group[p] - 1
            return

        ans = 0
        stack = [root] if root else []
        while stack:
            nex = []
            cur = [node.val for node in stack]
            check()
            for node in stack:
                if node.left:
                    nex.append(node.left)
                if node.right:
                    nex.append(node.right)
            stack = nex[:]
        return ans
