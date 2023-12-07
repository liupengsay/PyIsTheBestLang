"""

Algorithm：union_find、可持久化union_find、permutation_ring|
Function：用来处理图论相关的联通问题，通常结合逆向思考、permutation_ring|或者offline_query求解，连通块不一定是秩大小，也可以是最大最小值、和等

====================================LeetCode====================================
765（https://leetcode.com/problems/couples-holding-hands/）union_find
1697（https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/）sorting后offline_query两点间所有路径的最大边权值
2503（https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/）sorting后offline_query与起点相连的连通块的大小
2421（https://leetcode.com/problems/number-of-good-paths/）根据权值sorting更新union_find连通分块满足条件的节点对数
2382（https://leetcode.com/problems/maximum-segment-sum-after-removals/）逆向访问查询并更新连通块的结果
2334（https://leetcode.com/problems/subarray-with-elements-greater-than-varying-threshold/）sorting后brute_force动态维护union_find连通块
2158（https://leetcode.com/problems/amount-of-new-area-painted-each-day/）union_find维护区间左端点，不断合并
2157（https://leetcode.com/problems/groups-of-strings/）利用字母的有限数量变换brute_force分组
2076（https://leetcode.com/problems/process-restricted-friend-requests/）union_find变种，维护群体的不喜欢关系
2459（https://leetcode.com/problems/sort-array-by-moving-items-to-empty-space/）permutation_ring|题目
2709（https://leetcode.com/problems/greatest-common-divisor-traversal/）union_find具有相同质因数的连通块
2612（https://leetcode.com/problems/minimum-reverse-operations/）union_find应用 find_range_merge_to_disjoint 灵活
1559（https://leetcode.com/problems/detect-cycles-in-2d-grid/）union_find判环
1569（https://leetcode.com/problems/number-of-ways-to-reorder-array-to-get-same-bst/）reverse_thinking，reverse_order|利用union_find建立二叉搜索树，排列组合|union_find
1970（https://leetcode.com/problems/last-day-where-you-can-still-cross/）reverse_thinkingunion_find
1998（https://leetcode.com/problems/gcd-sort-of-an-array/）union_find|质factorization|
2158（https://leetcode.com/problems/amount-of-new-area-painted-each-day/）区间union_find
2471（https://leetcode.com/problems/minimum-number-of-operations-to-sort-a-binary-tree-by-level/description/）离散化permutation_ring|
945（https://leetcode.com/problems/minimum-increment-to-make-array-unique/description/）可向右合并的区间union_find，正解为greedy
947（https://leetcode.com/contest/weekly-contest-112/problems/most-stones-removed-with-same-row-or-column/）brain_teaserunion_find

=====================================LuoGu======================================
3367（https://www.luogu.com.cn/problem/P3367）连通分块的数量
5836（https://www.luogu.com.cn/problem/P5836）两个union_find不同方面的查询
3144（https://www.luogu.com.cn/problem/P3144）逆序union_find，考察连通块的数量
5836（https://www.luogu.com.cn/problem/P5836）两个union_find连通情况查询
5877（https://www.luogu.com.cn/problem/P5877）正向implemention实时更新连通块的数量
6111（https://www.luogu.com.cn/problem/P6111）union_find|offline_query
6121（https://www.luogu.com.cn/problem/P6121）逆序union_find根据连通块大小连通性判定
6153（https://www.luogu.com.cn/problem/P6153）union_find思想greedy题，体现了union_find的思想
1955（https://www.luogu.com.cn/problem/P1955）union_find裸题
1196（https://www.luogu.com.cn/problem/P1196）带权union_find
1197（https://www.luogu.com.cn/problem/P1197）逆序union_find，reverse_order|brute_force联通块个数
1522（https://www.luogu.com.cn/problem/P1522）连通块，brute_force新增路径并high_precision联通块直径
1621（https://www.luogu.com.cn/problem/P1621）利用素数筛的思想对数复杂度合并公共质因数大于p的数并连通块数量
1892（https://www.luogu.com.cn/problem/P1892）union_find，敌人与朋友关系
2189（https://www.luogu.com.cn/problem/P2189）union_find题，确定访问顺序的合法性
2307（https://www.luogu.com.cn/problem/P2307）union_find判定树的生成是否合法
3420（https://www.luogu.com.cn/problem/P3420）union_find变形问题
5429（https://www.luogu.com.cn/problem/P5429）简单union_find应用题
6193（https://www.luogu.com.cn/problem/P6193）permutation_ring|交换代价
6706（https://www.luogu.com.cn/problem/P6706）有向图union_find逆序更新边 find_range_merge_to_disjoint 灵活
7991（https://www.luogu.com.cn/problem/P7991）union_find连通块缩点使得 1 和 n 连通最多|两条路的代价
8230（https://www.luogu.com.cn/problem/P8230）分层union_find|implemention
8637（https://www.luogu.com.cn/problem/P8637）union_findpermutation_ring|
8686（https://www.luogu.com.cn/problem/P8686）union_find灵活应用
8785（https://www.luogu.com.cn/problem/P8785）根据边界union_find构建counter
8787（https://www.luogu.com.cn/problem/P8787）greedy二叉堆implemention与union_find灵活应用
8881（https://www.luogu.com.cn/problem/P8881）brain_teaser，union_find判断所属连通分量是否有环

===================================CodeForces===================================
25D（https://codeforces.com/problemset/problem/25/D）union_find将原来的边断掉重新来连接使得成为一整个连通集
1810E（https://codeforces.com/contest/1810/problem/E）union_find|启发式搜索，BFS与heapq实现
920E（https://codeforces.com/contest/920/problem/E）union_find，|线性动态维护剩余节点
540C（https://codeforces.com/problemset/problem/540/C）路径可达
1800E2（https://codeforces.com/problemset/problem/1800/E2）union_find分组可达
1691E（https://codeforces.com/contest/1691/problem/E）线段union_find
827A（https://codeforces.com/problemset/problem/827/A）UnionFindRightRoot|implemention|greedy

====================================AtCoder=====================================
D - Connectivity（https://atcoder.jp/contests/abc049/tasks/arc065_b）双union_find应用
E - 1 or 2（https://atcoder.jp/contests/abc126/tasks/abc126_e）双union_find的union_find应用
F - Must Be Rectangular!（https://atcoder.jp/contests/abc131/tasks/abc131_f）brain_teaser|union_findcounter

=====================================AcWing=====================================
4306（https://www.acwing.com/problem/content/description/4309/）向右合并的区间union_find
4866（https://www.acwing.com/problem/content/description/4869/）union_findimplemention维护连通块大小与多余的边数量
5145（https://www.acwing.com/problem/content/5148/）union_find判矩阵四元及以上的环

================================LibraryChecker================================
1 Cycle Detection (Undirected)（https://judge.yosupo.jp/problem/cycle_detection_undirected）use unionfind to detect circle in undirected graph

"""
import decimal
import math
from collections import defaultdict, Counter, deque
from heapq import heappop, heapify, heappush
from math import inf
from typing import List, Optional

from src.basis.tree_node.template import TreeNode
from src.data_structure.sorted_list.template import LocalSortedList
from src.graph.dijkstra.template import Dijkstra
from src.graph.union_find.template import UnionFind, UnionFindWeighted, UnionFindLeftRoot, \
    UnionFindRightRoot
from src.mathmatics.comb_perm.template import Combinatorics
from src.mathmatics.number_theory.template import NumberTheory
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1810e(ac=FastIO()):
        # union_find|启发式搜索，线性遍历维护
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
                    stack = [[0, i]]
                    while stack:
                        d, x = heappop(stack)
                        if count < nums[x]:
                            break
                        count += 1
                        for j in edge[x]:
                            if visit[j] != i:
                                visit[j] = i
                                heappush(stack, [nums[j], j])
                    if count == n:
                        ans = "YES"
                        break
            ac.st(ans)

        return

    @staticmethod
    def ac_5145(ac=FastIO()):
        # union_find判矩阵四元及以上的环
        m, n = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]
        edges = []
        uf = UnionFind(m * n)
        for i in range(m):
            for j in range(n):
                for x, y in [[i, j + 1], [i + 1, j]]:
                    # 只有上下左右，所以不会有三元环
                    if 0 <= x < m and 0 <= y < n and grid[x][y] == grid[i][j]:
                        edges.append([i * n + j, x * n + y])
                        uf.union(i * n + j, x * n + y)
        group = uf.get_root_part()
        degree = defaultdict(int)
        for i, j in edges:
            degree[uf.find(i)] += 1
        for g in group:
            # union_find多边必然是四元环及以上
            if degree[g] >= len(group[g]) >= 4:
                ac.st("Yes")
                return
        ac.st("No")
        return

    @staticmethod
    def cf_872a(ac=FastIO()):
        n = ac.read_int()
        nums = [ac.read_list_strs() for _ in range(n)]
        length = max(int(ls[-1]) + len(ls[0]) - 1 for ls in nums)
        uf = UnionFindRightRoot(length + 1)
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
                        uf.union(x, x + 1)
                    else:
                        break
        ac.st("".join(ans))
        return

    @staticmethod
    def cf_920e(ac=FastIO()):
        # union_find线性更新，集合维护
        n, m = ac.read_list_ints()
        edge = set()
        for _ in range(m):
            u, v = ac.read_list_ints_minus_one()
            edge.add((u, v))
        ans = []
        not_visit = set(range(n))
        for i in range(n):
            if i in not_visit:
                stack = [i]
                cnt = 1
                not_visit.discard(i)
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
        # union_find与离线sorting查询结合
        m = len(queries)

        # 按照 limit sorting
        ind = list(range(m))
        ind.sort(key=lambda x: queries[x][2])

        # 按照边权值sorting
        edge_list.sort(key=lambda x: x[2])
        uf = UnionFind(n)
        i = 0
        k = len(edge_list)
        ans = []
        # 查询 queries 里面的 [p, q, limit] 即 p 和 q 之间存在最大边权值严格小于 limit 的路径是否成立
        for j in ind:
            # 实时|入可用于连通的边并查询结果
            p, q, limit = queries[j]
            while i < k and edge_list[i][2] < limit:
                uf.union(edge_list[i][0], edge_list[i][1])
                i += 1
            ans.append([j, uf.is_connected(p, q)])

        # 按照顺序返回结果
        ans.sort(key=lambda x: x[0])
        return [an[1] for an in ans]

    @staticmethod
    def lc_2503(grid: List[List[int]], queries: List[int]) -> List[int]:
        # union_find与离线sorting查询结合
        dct = []
        # 根据邻居关系建图处理
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

        # 按照查询值的大小sorting，依次查询
        k = len(queries)
        ind = list(range(k))
        ind.sort(key=lambda d: queries[d])

        # 根据查询值的大小利用pointer持续更新union_find
        ans = [0] * k
        j = 0
        length = len(dct)
        for i in ind:
            cur = queries[i]
            while j < length and dct[j][2] < cur:
                uf.union(dct[j][0], dct[j][1])
                j += 1
            if cur > grid[0][0]:
                ans[i] = uf.size[uf.find(0)]
        return ans

    @staticmethod
    def lc_2421(vals: List[int], edges: List[List[int]]) -> int:
        # union_find与离线sorting查询结合
        n = len(vals)
        index = defaultdict(list)
        for i in range(n):
            index[vals[i]].append(i)
        edges.sort(key=lambda x: max(vals[x[0]], vals[x[1]]))
        uf = UnionFind(n)
        # offline_querycounter
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
        n, m = ac.read_list_ints()
        edges = [ac.read_list_ints() + [i] for i in range(m)]
        uf = UnionFind(n)
        dct = [[] for _ in range(n)]
        for u, v, i in edges:
            if not uf.union(u, v):
                stack = [[u, -1]]
                parent = [[-1, -1] for _ in range(n)]
                while stack:
                    x, fa = stack.pop()
                    for y, ind in dct[x]:
                        if y != fa:
                            parent[y] = [x, ind]
                            stack.append([y, x])
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
            dct[u].append([v, i])
            dct[v].append([u, i])
        else:
            ac.st(-1)

        return

    @staticmethod
    def lg_p1196(ac=FastIO()):
        # 带权union_find
        uf = UnionFindWeighted(30000)
        for _ in range(ac.read_int()):
            lst = ac.read_list_strs()
            i, j = [int(w) - 1 for w in lst[1:]]
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
        # 逆序union_find，reverse_order|brute_force联通块个数
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

        # 连通块，brute_force新增路径并high_precision联通块直径

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
            dist.append(Dijkstra().get_dijkstra_result(dct, i))

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
        # 利用素数筛的思想对数复杂度合并公共质因数大于p的数并连通块数量
        a, b, p = ac.read_list_ints()
        nums = list(range(a, b + 1))
        ind = {num: num - a for num in nums}
        primes = [x for x in NumberTheory().sieve_of_eratosthenes(b) if x >= p]

        # 利用素数合并
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
        # union_find，敌人与朋友关系
        n = ac.read_int()
        m = ac.read_int()
        uf = UnionFind(n)
        dct = dict()
        for _ in range(m):
            lst = [w for w in input().strip().split() if w]
            a, b = int(lst[1]), int(lst[2])
            a -= 1
            b -= 1
            if lst[0] == "E":
                # 敌人的敌人是朋友
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
        # union_find裸题
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
                    res.append([ind[i], ind[j]])
            if any(uf.is_connected(i, j) for i, j in res):
                ac.st("NO")
            else:
                ac.st("YES")
        return

    @staticmethod
    def lg_p2189(ac=FastIO()):

        # union_find题，确定访问顺序的合法性
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

            # 不在路径上的直接连通
            ans = True
            pre = order[0]
            for i in range(n):
                if not visit[i]:
                    for j in dct[i]:
                        if not visit[j]:
                            uf.union(i, j)

            # 遍历连接确认当前的连通性
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
        # union_find判定树的生成是否合法
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
        # 特殊图 n 个节点 n 条边的联通块数量
        n = ac.read_int()
        uf = UnionFind(n)
        for i in range(n):
            j = ac.read_int()
            uf.union(i, j - 1)
        ac.st(uf.part)
        return

    @staticmethod
    def lg_p6193(ac=FastIO()):
        # permutation_ring|交换代价
        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]
        lst = sorted(nums)
        # 离散化
        ind = {num: i for i, num in enumerate(lst)}
        uf = UnionFind(n)
        x = lst[0]
        # 寻找permutation_ring|
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
            #  当前permutation_ring|最小值交换
            cost1 = s + (m - 2) * y
            # 或者全局最小值交换
            cost2 = s - y + x + (m - 2) * x + (x + y) * 2
            ans += ac.min(cost1, cost2)
        ac.st(ans)
        return

    @staticmethod
    def lc_2709(nums: List[int]) -> bool:
        # union_find具有相同质因数的连通块
        prime_factor = NumberTheory().get_num_prime_factor(10 ** 5)  # 放在全局
        n = len(nums)
        uf = UnionFind(n)
        pre = dict()
        for i in range(n):
            for num in prime_factor[nums[i]]:
                if num in pre:
                    uf.union(i, pre[num])
                else:
                    pre[num] = i
        return uf.part == 1

    @staticmethod
    def lg_p6706(ac=FastIO()):
        # 有向图union_find逆序更新边 find_range_merge_to_disjoint 灵活
        n = ac.read_int()
        edge = ac.read_list_ints_minus_one()
        q = ac.read_int()
        query = [ac.read_list_ints() for _ in range(q)]
        rem = dict()
        for op, x in query:
            if op == 2:
                rem[x - 1] = edge[x - 1]
                edge[x - 1] = -1

        def find_range_merge_to_disjoint(y):
            tmp = [y]
            while edge[tmp[-1]] not in [-1, n, y]:
                tmp.append(edge[tmp[-1]])
            if edge[tmp[-1]] == -1:
                for yy in tmp[:-1]:
                    edge[yy] = tmp[-1]
            else:
                for yy in tmp[:-1]:
                    edge[yy] = n
            return

        ans = []
        for i in range(q - 1, -1, -1):
            op, x = query[i]
            x -= 1
            if op == 1:
                find_range_merge_to_disjoint(x)
                if edge[x] == n:
                    res = "CIKLUS"
                elif edge[x] == -1:
                    res = x + 1
                else:
                    res = edge[x] + 1
                ans.append(res)
            else:
                edge[x] = rem[x]
                find_range_merge_to_disjoint(x)
        for i in range(len(ans) - 1, -1, -1):
            ac.st(ans[i])
        return

    @staticmethod
    def lg_p7991(ac=FastIO()):
        # union_find连通块缩点使得 1 和 n 连通最多|两条路的代价
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            uf = UnionFind(n)
            for _ in range(m):
                i, j = ac.read_list_ints_minus_one()
                uf.union(i, j)
            if uf.is_connected(0, n - 1):
                ac.st(0)
                continue

            dis_0 = [inf] * n
            dis_1 = [inf] * n

            pre_0 = pre_1 = -1
            for i in range(n):
                if uf.is_connected(0, i):
                    pre_0 = i
                if uf.is_connected(n - 1, i):
                    pre_1 = i
                if pre_0 != -1:
                    dis_0[uf.find(i)] = ac.min(dis_0[uf.find(i)], (i - pre_0) ** 2)
                if pre_1 != -1:
                    dis_1[uf.find(i)] = ac.min(dis_1[uf.find(i)], (i - pre_1) ** 2)

            pre_0 = pre_1 = -1
            for i in range(n - 1, -1, -1):
                if uf.is_connected(0, i):
                    pre_0 = i
                if uf.is_connected(n - 1, i):
                    pre_1 = i
                if pre_0 != -1:
                    dis_0[uf.find(i)] = ac.min(dis_0[uf.find(i)], (i - pre_0) ** 2)
                if pre_1 != -1:
                    dis_1[uf.find(i)] = ac.min(dis_1[uf.find(i)], (i - pre_1) ** 2)
            ans = min(dis_0[i] + dis_1[i] for i in range(n))
            ac.st(ans)
        return

    @staticmethod
    def lc_2612(n: int, p: int, banned: List[int], k: int) -> List[int]:

        def find_range_merge_to_disjoint(x):
            # union_find父节点表示下一个为访问的点类似链表
            tmp = []
            while x != fa[x]:
                tmp.append(x)
                x = fa[x]
            for y in tmp:
                fa[y] = x
            return x

        ans = [-1] * n
        fa = list(range(n + 2))
        for i in banned:
            fa[i] = i + 2

        stack = deque([p])
        ans[p] = 0
        while stack:
            i = stack.popleft()
            # 满足 low <= j <= high 且要有相同的odd_even
            low = max(0, k - 1 - i, i - k + 1)
            high = min(2 * n - k - 1 - i, n - 1, i + k - 1)
            j = find_range_merge_to_disjoint(low)
            while j <= high:
                if ans[j] == -1:
                    # 未访问过
                    ans[j] = ans[i] + 1
                    fa[j] = j + 2  # range_merge_to_disjoint到下一个
                    stack.append(j)
                # 继续访问下一个
                j = find_range_merge_to_disjoint(j + 2)
        return ans

    @staticmethod
    def lg_p8230(ac=FastIO()):
        # 分层union_find|implemention
        k, m, n = ac.read_list_ints()
        ans = 1
        start = [0, 0]
        for _ in range(k):
            grid = [ac.read_list_ints() for _ in range(m)]
            lst = []
            end = [-1, -1]
            uf = UnionFind(m * n)
            for i in range(m):
                for j in range(n):
                    w = grid[i][j]
                    if w != -9:
                        lst.append([w, i, j])
                        for x, y in [[i + 1, j], [i - 1, j], [i, j - 1], [i, j + 1]]:
                            if 0 <= x < m and 0 <= y < n and grid[x][y] != -9:
                                uf.union(i * n + j, x * n + y)
                    if w == -1:
                        end = [i, j]
            lst.sort()

            for val, i, j in lst:
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
        # union_find灵活应用
        ac.read_int()
        nums = ac.read_list_ints()
        post = dict()
        ans = []
        for num in nums:
            lst = [num]
            while lst[-1] in post:
                lst.append(post[lst[-1]])
            for x in lst:
                post[x] = lst[-1] + 1
            ans.append(lst[-1])
        ac.lst(ans)
        return

    @staticmethod
    def lg_p8787(ac=FastIO()):
        # greedy二叉堆implemention与union_find灵活应用
        n = ac.read_int()
        nums = ac.read_list_ints()
        stack = [[-nums[i], -i] for i in range(n)]
        heapify(stack)
        uf = UnionFindLeftRoot(n)
        for i in range(n):
            if i and nums[i] == nums[i - 1]:
                uf.union(i - 1, i)
        ans = 0
        while stack:
            val, i = heappop(stack)
            val, i = -val, -i
            if val == 1:
                break
            if i != uf.find(i):
                continue
            if i and nums[uf.find(i - 1)] == val:
                uf.union(i - 1, i)
                continue
            ans += 1
            val = int(((val // 2) + 1) ** 0.5)
            nums[i] = val
            heappush(stack, [-nums[i], -i])
        ac.st(ans)
        return

    @staticmethod
    def lg_p8881(ac=FastIO()):
        # brain_teaser，union_find判断所属连通分量是否有环
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            uf = UnionFind(n)
            edge = []
            for _ in range(m):
                i, j = ac.read_list_ints_minus_one()
                uf.union(i, j)
                edge.append([i, j])
            cnt = 0
            for i, j in edge:
                if uf.is_connected(0, i):
                    cnt += 1
            ac.st("1.000" if uf.size[uf.find(0)] == cnt + 1 else "0.000")
        return

    @staticmethod
    def lc_945(nums: List[int]) -> int:
        # 可向右合并的区间union_find，正解为greedy
        nums.sort()
        ans = 0
        uf = UnionFindRightRoot(max(nums) + len(nums) + 2)
        for num in nums:
            # 其根节点就是当前还未被占据的节点
            x = uf.find(num)
            ans += x - num
            uf.union(x, x + 1)
        return ans

    @staticmethod
    def lc_1559(grid: List[List[str]]) -> bool:
        # union_find判环
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
    def lc_1569(nums: List[int]) -> int:

        # reverse_thinking，排列组合|union_find
        len(nums)
        mod = 10 ** 9 + 7
        n = 10 ** 3
        cb = Combinatorics(n, mod)

        # reverse_thinking，reverse_order|利用union_find建立二叉搜索树
        dct = [[] for _ in range(n)]
        uf = UnionFindRightRoot(n)
        post = {}
        for i in range(n - 1, -1, -1):
            x = nums[i]
            if x + 1 in post:
                r = uf.find(post[x + 1])
                dct[i].append(r)
                uf.union(i, r)
            if x - 1 in post:
                r = uf.find(post[x - 1])
                dct[i].append(r)
                uf.union(i, r)
            post[x] = i
        # 树形 DP
        stack = [0]
        sub = [0] * n
        ans = 1
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in dct[i]:
                    stack.append(j)
            else:
                i = ~i
                lst = [0]
                for j in dct[i]:
                    lst.append(sub[j])
                    sub[i] += sub[j]
                s = sum(lst)
                ans *= cb.comb(s, lst[-1])
                ans %= mod
                sub[i] += 1
        ans = (ans - 1) % mod
        return ans

    @staticmethod
    def lc_2158(paint: List[List[int]]) -> List[int]:
        # 区间union_find
        m = 5 * 10 ** 4 + 10
        uf = UnionFindRightRoot(m)
        ans = []
        for a, b in paint:
            cnt = 0
            while a < b:
                a = uf.find(a)
                if a < b:
                    cnt += 1
                    uf.union(a, a + 1)
                    a += 1
            ans.append(cnt)
        return ans

    @staticmethod
    def abc_49d(ac=FastIO()):
        # 双union_find应用
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
        # brain_teaser|union_findcounter
        n = ac.read_int()
        m = 10 ** 5
        uf = UnionFind(2 * m)
        for _ in range(n):
            x, y = ac.read_list_ints()
            x -= 1
            y -= 1
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
    def ac_4306(ac=FastIO()):
        # 向右合并的区间union_find
        n = ac.read_int()
        a = ac.read_list_ints()
        uf = UnionFindRightRoot(n * 2 + 2)
        a.sort()
        ans = 0
        for num in a:
            # 其根节点就是当前还未被占据的节点
            x = uf.find(num)
            ans += x - num
            uf.union(x, x + 1)
        ac.st(ans)
        return

    @staticmethod
    def ac_4866(ac=FastIO()):
        # union_findimplemention维护连通块大小与多余的边数量
        n, d = ac.read_list_ints()
        uf = UnionFind(n)
        lst = LocalSortedList([1] * n)
        pre = 0
        for i in range(d):
            x, y = ac.read_list_ints_minus_one()
            if uf.is_connected(x, y):
                pre += 1
            else:
                lst.discard(uf.size[uf.find(x)])
                lst.discard(uf.size[uf.find(y)])
                uf.union(x, y)
                lst.add(uf.size[uf.find(x)])
            ans = 0
            m = len(lst)
            for j in range(m - 1, m - pre - 2, -1):
                ans += lst[j]
            ac.st(ans - 1)
        return

    @staticmethod
    def lc_2471(root: Optional[TreeNode]) -> int:
        # 离散化permutation_ring|

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
            ans += check()
            for node in stack:
                if node.left:
                    nex.append(node.left)
                if node.right:
                    nex.append(node.right)
            stack = nex[:]
        return ans