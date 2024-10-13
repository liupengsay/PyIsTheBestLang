"""

Algorithm：topological_sort|circle_based_tree|bfs_order|topological_order|topological_lexicographic_order
Description：undirected_topological_sort|directed_topological_sort|directed_circle_based_tree|undirected_circle_based_tree

====================================LeetCode====================================
360（https://leetcode.cn/problems/longest-cycle-in-a-graph/）topological_sort|directed_circle_based_tree|longest_circle
2392（https://leetcode.cn/problems/build-a-matrix-with-conditions/）build_graph|union_find|topological_sort
2371（https://leetcode.cn/problems/minimize-maximum-value-in-a-grid/）build_graph|topological_sort|greedy
2127（https://leetcode.cn/problems/maximum-employees-to-be-invited-to-a-meeting/）topological_sort|dag|directed_circle_based_tree|classification_discussion
127（https://leetcode.cn/problems/maximum-employees-to-be-invited-to-a-meeting/）topological_sort|directed_circle_based_tree|
269（https://leetcode.cn/problems/alien-dictionary/）lexicographical_order|build_graph|topological_sort
2603（https://leetcode.cn/problems/collect-coins-in-a-tree/）undirected_topological_sort|undirected_circle_based_tree
2204（https://leetcode.cn/problems/distance-to-a-cycle-in-undirected-graph/）undirected_topological_sort
1857（https://leetcode.cn/problems/largest-color-value-in-a-directed-graph/）topological_sort|dag_dp
1932（https://leetcode.cn/problems/merge-bsts-to-create-single-bst/）union_find|topological_sort|union_find|binary_search_tree
1591（https://leetcode.cn/problems/strange-printer-ii/）build_graph|topological_sort|circle_judge
2192（https://leetcode.cn/problems/all-ancestors-of-a-node-in-a-directed-acyclic-graph/）directed_topological_sort|dag_dp

=====================================LuoGu======================================
P1960（https://www.luogu.com.cn/problem/P1960）topological_sort|topological_order
P1992（https://www.luogu.com.cn/problem/P1992）directed_topological_sort|directed_circle_judge
P2712（https://www.luogu.com.cn/problem/P2712）topological_sort|circle_judge|find_circle
P6145（https://www.luogu.com.cn/problem/P6145）directed_topological_sort|dag_dp
P1137（https://www.luogu.com.cn/problem/P1137）topological_sort|dag_dp
P1347（https://www.luogu.com.cn/problem/P1347）topological_sort|lexicographical_order|construction
P1685（https://www.luogu.com.cn/problem/P1685）dag_dp|directed_topological_sort|counter
P3243（https://www.luogu.com.cn/problem/P3243）reverse_graph|topological_sort|heapq|implemention|topological_lexicographic_order
P5536（https://www.luogu.com.cn/problem/P5536）undirected_topological_sort
P6037（https://www.luogu.com.cn/problem/P6037）undirected_circle_based_tree|union_find|topological_sort|implemention
P6255（https://www.luogu.com.cn/problem/P6255）union_find|topological_sort|circle_judge
P6417（https://www.luogu.com.cn/problem/P6417）directed_circle_based_tree|greedy|topological_sort
P6560（https://www.luogu.com.cn/problem/P6560）reverse_graph|topological_sort|game_dp
P8655（https://www.luogu.com.cn/problem/P8655）topological_sort|directed_circle_based_tree
P8943（https://www.luogu.com.cn/problem/P8943）undirected_circle_based_tree|game_dp
P1983（https://www.luogu.com.cn/problem/P1983）topological_sort
P2921（https://www.luogu.com.cn/problem/P2921）circle_based_tree|topological_sort|classical

===================================CodeForces===================================
1454E（https://codeforces.com/contest/1454/problem/E）circle_based_tree|counter|brute_force|inclusion_exclusion
1907G（https://codeforces.com/contest/1907/problem/G）directed_circle_based_tree|greedy|implemention|topological_sort
1914F（https://codeforces.com/contest/1914/problem/F）topological_sort|greedy
1829F（https://codeforces.com/contest/1829/problem/F）reverse_thinking|degree|undirected_graph
1873H（https://codeforces.com/contest/1873/problem/H）circle_based_tree|topological_sort
1029E（https://codeforces.com/contest/1029/problem/E）greedy|implemention|rooted_tree|depth|degree
1872F（https://codeforces.com/contest/1872/problem/F）topological_sort|greedy
1388D（https://codeforces.com/problemset/problem/1388/D）topological_sort|dag_dp|heuristic_merge|classical
919D（https://codeforces.com/problemset/problem/919/D）topological_sort|dag_dp|classical
1335F（https://codeforces.com/problemset/problem/1335/F）circle_based_tree|implemention|observation|greedy
1027D（https://codeforces.com/problemset/problem/1027/D）topological_sort|classical|circle_based_tree
1867D（https://codeforces.com/problemset/problem/1867/D）topological_sort|classical|circle_based_tree

====================================AtCoder=====================================
ABC266F（https://atcoder.jp/contests/abc266/tasks/abc266_f）undirected_circle_based_tree
ABC303E（https://atcoder.jp/contests/abc303/tasks/abc303_e）undirected_graph|topological_sort
ABC296E（https://atcoder.jp/contests/abc296/tasks/abc296_e）topological_sort|directed_graph
ABC256E（https://atcoder.jp/contests/abc256/tasks/abc256_e）topological_sort|greedy|circle_based_tree|classical
ABC223D（https://atcoder.jp/contests/abc223/tasks/abc223_d）topological_sort


=====================================AcWing=====================================
3696（https://www.acwing.com/problem/content/description/3699/）topological_order|dag|construction
3831（https://www.acwing.com/problem/content/description/3831/）topological_sort|dag_dp|circle_judge
4629（https://www.acwing.com/problem/content/description/4629/）directed_circle_based_tree|circle_judge

=====================================LibraryChecker=====================================
1（https://www.hackerrank.com/challenges/favourite-sequence/problem?isFullScreen=true）topological_lexicographic_order


"""

import math
from collections import defaultdict, deque
from heapq import heapify, heappop, heappush
from typing import List, Optional

from src.basis.tree_node.template import TreeNode
from src.graph.network_flow.template import UndirectedGraph
from src.graph.union_find.template import UnionFind
from src.utils.fast_io import FastIO



class Solution:
    def __init__(self):
        return

    @staticmethod
    def abc_266f_1(ac=FastIO()):
        """
        urL: https://atcoder.jp/contests/abc266/tasks/abc266_f
        tag: undirected_circle_based_tree|classical|connected
        """
        n = ac.read_int()
        edge = [[] for _ in range(n)]
        uf = UnionFind(n)

        degree = [0] * n
        for _ in range(n):
            u, v = ac.read_list_ints_minus_one()
            edge[u].append(v)
            edge[v].append(u)
            degree[u] += 1
            degree[v] += 1

        que = deque()
        for i in range(n):
            if degree[i] == 1:
                que.append(i)
        while que:
            now = que.popleft()
            nex = edge[now][0]
            degree[now] -= 1
            degree[nex] -= 1
            edge[nex].remove(now)
            uf.union(now, nex)
            if degree[nex] == 1:
                que.append(nex)

        q = ac.read_int()
        for _ in range(q):
            x, y = ac.read_list_ints_minus_one()
            if uf.is_connected(x, y):
                ac.yes()
            else:
                ac.no()
        return

    @staticmethod
    def abc_266f_2(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc266/tasks/abc266_f
        tag: topological|sort|circle_based_tree|classical
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(n):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
            degree[i] += 1
            degree[j] += 1
        stack = [i for i in range(n) if degree[i] == 1]
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if degree[j] == 1:
                        nex.append(j)
            stack = nex[:]
        circle = [i for i in range(n) if degree[i] >= 2]
        father = [-1] * n
        for i in circle:
            father[i] = i
        stack = circle[:]
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    if father[j] == -1:
                        father[j] = father[i]
                        nex.append(j)
            stack = nex[:]
        for _ in range(ac.read_int()):
            x, y = ac.read_list_ints_minus_one()
            if father[x] != father[y]:
                ac.no()
            else:
                ac.yes()
        return

    @staticmethod
    def ac_3696(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/3699/
        tag: topological_order|dag|construction|classical|hard|brain_teaser
        """
        for _ in range(ac.read_int()):

            def check():
                n, m = ac.read_list_ints()
                dct = [[] for _ in range(n)]
                degree = [0] * n
                edges = []
                ans = []
                for _ in range(m):
                    t, x, y = ac.read_list_ints()
                    x -= 1
                    y -= 1
                    if t == 1:
                        ans.append([x, y])
                        dct[x].append(y)
                        degree[y] += 1
                    else:
                        edges.append([x, y])

                order = [0] * n
                stack = [i for i in range(n) if degree[i] == 0]
                ind = 0
                while stack:
                    nex = []
                    for i in stack:
                        order[i] = ind
                        ind += 1
                        for j in dct[i]:
                            degree[j] -= 1
                            if degree[j] == 0:
                                nex.append(j)
                    stack = nex[:]
                if any(degree[x] > 0 for x in range(n)):
                    ac.no()
                    return

                ac.yes()
                for x, y in edges:
                    if order[x] < order[y]:
                        ac.lst([x + 1, y + 1])
                    else:
                        ac.lst([y + 1, x + 1])
                for x, y in ans:
                    ac.lst([x + 1, y + 1])
                return

            check()
        return

    @staticmethod
    def lc_2392(k: int, row_conditions: List[List[int]], col_conditions: List[List[int]]) -> List[List[int]]:
        """
        url: https://leetcode.cn/problems/build-a-matrix-with-conditions/
        tag: build_graph|union_find|topological_sort
        """

        def check(cond):
            dct = defaultdict(list)
            degree = defaultdict(int)
            for i, j in cond:
                dct[i].append(j)
                degree[j] += 1
            stack = [i for i in range(1, k + 1) if not degree[i]]
            ans = []
            while stack:
                ans.extend(stack)
                nex = []
                for i in stack:
                    for j in dct[i]:
                        degree[j] -= 1
                        if not degree[j]:
                            nex.append(j)
                stack = nex
            return ans

        row = check(row_conditions)
        col = check(col_conditions)
        if len(row) != k or len(col) != k:
            return []

        row_ind = {row[i]: i for i in range(k)}
        col_ind = {col[i]: i for i in range(k)}
        res = [[0] * k for _ in range(k)]
        for i in range(1, k + 1):
            res[row_ind[i]][col_ind[i]] = i
        return res

    @staticmethod
    def lg_p1137(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1137
        tag: topological_sort|dag_dp|longest_path
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(m):
            i, j = ac.read_list_ints()
            degree[j - 1] += 1
            dct[i - 1].append(j - 1)
        cnt = [1] * n
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            i = stack.pop()
            for j in dct[i]:
                cnt[j] = max(cnt[j], cnt[i] + 1)
                degree[j] -= 1
                if not degree[j]:
                    stack.append(j)
        for a in cnt:
            ac.st(a)
        return

    @staticmethod
    def lg_p1347(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1347
        tag: topological_sort|lexicographical_order|construction
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        degree = [0] * n

        def check():
            stack = [k for k in range(n) if not degree[k]]
            res = []
            unique = True
            while stack:
                res.extend(stack)
                if len(stack) > 1:
                    unique = False
                nex = []
                for i in stack:
                    for j in dct[i]:
                        degree[j] -= 1
                        if not degree[j]:
                            nex.append(j)
                stack = nex

            if unique and len(nodes) == n:
                ss = "".join([chr(ord("A") + w) for w in res])
                return True, f"Sorted sequence determined after {x} relations: {ss}."

            if len(res) < n:
                return True, f"Inconsistency found after {x} relations."

            return False, "Sorted sequence cannot be determined."

        nodes = set()
        for x in range(1, m + 1):
            s = ac.read_str()
            dct[ord(s[0]) - ord("A")].append(ord(s[2]) - ord("A"))
            degree[ord(s[2]) - ord("A")] += 1
            nodes.add(s[0])
            nodes.add(s[2])
            original = degree[:]
            flag, ans = check()
            if flag:
                ac.st(ans)
                break
            degree = original[:]
        else:
            ac.st("Sorted sequence cannot be determined.")
        return

    @staticmethod
    def lg_p1685(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1685
        tag: dag_dp|directed_topological_sort|counter|classical
        """
        n, m, s, e, t = ac.read_list_ints()
        s -= 1
        e -= 1
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(m):
            i, j, w = ac.read_list_ints()
            i -= 1
            j -= 1
            dct[i].append([j, w])
            degree[j] += 1
        mod = 10000

        time = [0] * n
        cnt = [0] * n
        stack = [s]
        cnt[s] = 1
        while stack:
            i = stack.pop()
            for j, w in dct[i]:
                degree[j] -= 1
                if not degree[j]:
                    stack.append(j)
                cnt[j] += cnt[i]
                time[j] += cnt[i] * w + time[i]
                time[j] %= mod
        ans = time[e] + (cnt[e] - 1) * t
        ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def lg_p3243(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3243
        tag: reverse_graph|topological_sort|heapq|implemention|topological_lexicographic_order|brain_teaser|classical|hard
        """
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            dct = [[] for _ in range(n)]
            degree = [0] * n
            for _ in range(m):
                i, j = ac.read_list_ints_minus_one()
                dct[j].append(i)
                degree[i] += 1
            ans = []
            stack = [-i for i in range(n) if not degree[i]]
            heapify(stack)
            while stack:
                i = -heappop(stack)
                ans.append(i)
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        heappush(stack, -j)
            if len(ans) == n:
                ans.reverse()
                ac.lst([x + 1 for x in ans])
            else:
                ac.st("Impossible!")
        return

    @staticmethod
    def lg_p5536(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5536
        tag: undirected_topological_sort|classical|brain_teaser
        """
        n, k = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
            degree[i] += 1
            degree[j] += 1

        rem = n - k
        ans = 0
        stack = deque([[i, 1] for i in range(n) if degree[i] == 1])
        while rem:
            i, d = stack.popleft()
            ans = d
            rem -= 1
            for j in dct[i]:
                if degree[j] > 1:
                    degree[j] -= 1
                    if degree[j] == 1:
                        stack.append([j, d + 1])
        ac.st(ans)
        return

    @staticmethod
    def lg_p6037(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6037
        tag: undirected_circle_based_tree|union_find|topological_sort|implemention
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        tot = 0
        for _ in range(n):
            u, v, w, p = ac.read_list_ints()
            u -= 1
            v -= 1
            dct[u].append((v, w, p))
            dct[v].append((u, w, p))
            degree[u] += 1
            degree[v] += 1
            tot += w
        stack = [i for i in range(n) if degree[i] == 1]
        while stack:
            nex = []
            for i in stack:
                for j, _, _ in dct[i]:
                    degree[j] -= 1
                    if degree[j] == 1:
                        nex.append(j)
            stack = nex
        ans = [0] * n
        for i in range(n):
            if degree[i] > 1:
                p1 = math.inf
                w1 = 0
                for j, w, p in dct[i]:
                    if degree[j] > 1 and p < p1:
                        p1 = p
                        w1 = w
                ans[i] = tot - w1
            stack = [(i, -1)]
            while stack:
                x, fa = stack.pop()
                for y, _, _ in dct[x]:
                    if degree[y] <= 1 and y != fa:
                        stack.append((y, x))
                        ans[y] = ans[i]
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lg_p6037_2(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6037
        tag: undirected_circle_based_tree|union_find|topological_sort|implemention
        """

        n = ac.read_int()

        graph = UndirectedGraph(n)

        degree = [0] * (n + 1)
        tot = 0
        for i in range(n):
            u, v, w, p = ac.read_list_ints()
            graph.add_edge(u, v, w, p)
            degree[u] += 1
            degree[v] += 1
            tot += w
        stack = [i for i in range(1, n + 1) if degree[i] == 1]
        while stack:
            nex = []
            for u in stack:
                i = graph.point_head[u]
                while i:
                    j = graph.edge_to[i]
                    degree[j] -= 1
                    if degree[j] == 1:
                        nex.append(j)
                    i = graph.edge_next[i]
            stack = nex

        ans = [0] * (n + 1)
        for u in range(1, n + 1):
            if degree[u] > 1:
                p1 = math.inf
                w1 = 0
                i = graph.point_head[u]
                while i:
                    p, w = graph.edge_p[i], graph.edge_w[i]
                    j = graph.edge_to[i]
                    i = graph.edge_next[i]
                    if degree[j] > 1 and p < p1:
                        p1 = p
                        w1 = w
                ans[u] = tot - w1
            stack = [(u, -1)]
            while stack:
                x, fa = stack.pop()
                i = graph.point_head[x]
                while i:
                    j = graph.edge_to[i]
                    if degree[j] <= 1:
                        stack.append((j, x))
                        ans[j] = ans[u]
                    i = graph.edge_next[i]
        for a in ans[1:]:
            ac.st(a)
        return


    @staticmethod
    def lg_p6255(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6255
        tag: topological_sort|circle_judge|classical|simple_graph|brain_teaser|bfs
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            degree[j] += 1
            degree[i] += 1
            dct[i].append(j)
            dct[j].append(i)
        original = degree[:]

        stack = deque([i for i in range(n) if degree[i] == 1])
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                degree[j] -= 1
                if degree[j] == 1:
                    stack.append(j)

        stack = deque([i for i in range(n) if degree[i] >= 2])
        ans = []
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                if 0 <= degree[j] < 2:
                    if degree[i] >= 2:
                        ans.append([i + 1, j + 1])
                    degree[j] = -1
                    stack.append(j)

        for i in range(n):
            if 0 <= degree[i] < 2 and original[i] == 1:
                ans.append([i + 1, dct[i][0] + 1])
        ans.sort()
        ac.st(len(ans))
        for ls in ans:
            ac.lst(ls)
        return

    @staticmethod
    def lg_p6417(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6417
        tag: directed_circle_based_tree|greedy|topological_sort|brain_teaser|classical
        """
        n = ac.read_int() # MLE
        dct = [ac.read_int() - 1 for _ in range(n)]
        degree = [0] * n
        for i in range(n):
            degree[dct[i]] += 1

        stack = [x for x in range(n) if not degree[x]]
        visit = [-1] * n
        ans = len(stack)
        for i in stack:
            visit[i] = 1
        while stack:
            nex = []
            for i in stack:
                degree[dct[i]] -= 1
                if (not degree[dct[i]] or visit[i] == 1) and visit[dct[i]] == -1:
                    if visit[i] == 1:
                        visit[dct[i]] = 0
                    else:
                        ans += 1
                        visit[dct[i]] = 1
                    nex.append(dct[i])
            stack = nex[:]

        for i in range(n):
            x = 0
            while visit[i] == -1:
                visit[i] = 1
                x += 1
                i = dct[i]
            ans += x // 2
        ac.st(ans)
        return

    @staticmethod
    def lg_p6560(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6560
        tag: reverse_graph|topological_sort|game_dp|brain_teaser|game_dp|classical
        """
        n, m, q = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        degree = [0 for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            dct[j].append(i)
            degree[i] += 1
        original = degree[:]
        out = [i for i in range(n) if not degree[i]]
        state = [0] * n
        for i in out:
            state[i] = -1
        dp = state[:]
        for _ in range(q):
            x, y = ac.read_list_ints_minus_one()
            for i in range(n):
                degree[i] = original[i]
                dp[i] = state[i]
            stack = deque(out + [y]) if degree[y] else deque(out)
            if degree[y]:
                dp[y] = -1
            while stack:
                u = stack.popleft()
                for v in dct[u]:
                    if dp[v] != 0:
                        continue
                    if dp[u] == 1:
                        degree[v] -= 1
                        if not degree[v]:
                            dp[v] = -1
                            stack.append(v)
                    else:
                        dp[v] = 1
                        stack.append(v)
            ac.st(dp[x])
        return

    @staticmethod
    def lg_p8655(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8655
        tag: topological_sort|directed_circle_based_tree
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(n):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
            degree[i] += 1
            degree[j] += 1
        stack = [x for x in range(n) if degree[x] == 1]
        while stack:
            y = stack.pop()
            for z in dct[y]:
                degree[z] -= 1
                if degree[z] == 1:
                    stack.append(z)
        ans = [x + 1 for x in range(n) if degree[x] == 2]
        ac.lst(ans)
        return

    @staticmethod
    def lg_p8943(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8943
        tag: undirected_circle_based_tree|game_dp
        """
        n, q = ac.read_list_ints()
        degree = [0] * n
        dct = [[] for _ in range(n)]
        for _ in range(n):
            x, y = ac.read_list_ints_minus_one()
            dct[x].append(y)
            dct[y].append(x)
            degree[x] += 1
            degree[y] += 1

        stack = deque([i for i in range(n) if degree[i] == 1])
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                degree[j] -= 1
                if degree[j] == 1:
                    stack.append(j)

        dis = [math.inf] * n
        stack = [i for i in range(n) if degree[i] >= 2]
        path = [stack[0]]
        degree[stack[0]] = 0
        while True:
            for j in dct[path[-1]]:
                if degree[j] >= 2 and j != path[-1]:
                    path.append(j)
                    degree[j] = 0
                    break
            else:
                break
        ind = {num: i for i, num in enumerate(path)}

        ancestor = [-1] * n
        for i in stack:
            dis[i] = 0
            ancestor[i] = i
        while stack:
            i = stack.pop()
            for j in dct[i]:
                if dis[j] == math.inf:
                    stack.append(j)
                    dis[j] = dis[i] + 1
                    ancestor[j] = ancestor[i]

        for _ in range(q):
            x, y = ac.read_list_ints_minus_one()
            if x == y:
                ac.st("Deception")
                continue
            dis_x = dis[x]
            xy = abs(ind[ancestor[x]] - ind[ancestor[y]])
            dis_y = dis[y] + min(xy, len(ind) - xy)
            if dis_x < dis_y:
                ac.st("Survive")
            else:
                ac.st("Deception")
        return

    @staticmethod
    def lc_2127(favorite: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-employees-to-be-invited-to-a-meeting/
        tag: topological_sort|dag|directed_circle_based_tree|classification_discussion|classical|hard
        """
        n = len(favorite)
        degree = [0] * n
        for i in range(n):
            degree[favorite[i]] += 1
        depth = [0] * n
        stack = [i for i in range(n) if degree[i] == 0]
        while stack:
            nex = []
            for i in stack:
                j = favorite[i]
                degree[j] -= 1
                a, b = depth[i] + 1, depth[j]
                depth[j] = a if a > b else b
                if not degree[j]:
                    nex.append(j)
            stack = nex[:]
        ans = 0
        bicycle = 0
        for i in range(n):
            if not degree[i]:
                continue
            lst = [i]
            degree[i] = 0
            x = favorite[i]
            while x != i:
                lst.append(x)
                degree[x] = 0
                x = favorite[x]
            if len(lst) == 2:
                bicycle += depth[lst[0]] + depth[lst[1]] + 2
            elif len(lst) > ans:
                ans = len(lst)

        ans = ans if ans > bicycle else bicycle
        return ans

    @staticmethod
    def lc_2192(n: int, edges: List[List[int]]) -> List[List[int]]:
        """
        url: https://leetcode.cn/problems/all-ancestors-of-a-node-in-a-directed-acyclic-graph/
        tag: directed_topological_sort|dag_dp
        """
        ans = [set() for _ in range(n)]
        degree = [0] * n
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            degree[j] += 1
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    for x in ans[i]:
                        ans[j].add(x)
                    ans[j].add(i)
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return [sorted(list(a)) for a in ans]

    @staticmethod
    def lc_2204(n: int, edges: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/distance-to-a-cycle-in-undirected-graph/
        tag: undirected_topological_sort
        """
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)
            degree[i] += 1
            degree[j] += 1
        stack = deque([i for i in range(n) if degree[i] == 1])
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                degree[j] -= 1
                if degree[j] == 1:
                    stack.append(j)

        circle = deque([i for i in range(n) if degree[i] > 1])
        ans = [-1] * n
        for i in circle:
            ans[i] = 0
        while circle:
            i = circle.popleft()
            for j in dct[i]:
                if ans[j] == -1:
                    ans[j] = ans[i] + 1
                    circle.append(j)
        return ans

    @staticmethod
    def lc_1857(colors: str, edges: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/largest-color-value-in-a-directed-graph/
        tag: topological_sort|dag_dp|alphabet|data_range|classical
        """
        n = len(colors)
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for i, j in edges:
            if i == j:
                return -1
            dct[i].append(j)
            degree[j] += 1

        cnt = [[0] * 26 for _ in range(n)]
        stack = deque([i for i in range(n) if not degree[i]])
        for i in stack:
            cnt[i][ord(colors[i]) - ord("a")] += 1
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                degree[j] -= 1
                for c in range(26):
                    a, b = cnt[j][c], cnt[i][c]
                    cnt[j][c] = a if a > b else b
                if not degree[j]:
                    cnt[j][ord(colors[j]) - ord("a")] += 1
                    stack.append(j)
        if not all(x == 0 for x in degree):
            return -1
        return max(max(c) for c in cnt)

    @staticmethod
    def lc_1932(trees: List[TreeNode]) -> Optional[TreeNode]:
        """
        url: https://leetcode.cn/problems/merge-bsts-to-create-single-bst/
        tag: union_find|topological_sort|union_find|binary_search_tree|classical
        """
        n = len(trees)
        ind = {tree.val: i for i, tree in enumerate(trees)}

        degree = [0] * n
        for i in range(n):
            if trees[i].left:
                if trees[i].left.val in ind:
                    degree[ind[trees[i].left.val]] += 1
            if trees[i].right:
                if trees[i].right.val in ind:
                    degree[ind[trees[i].right.val]] += 1

        stack = [(i, -math.inf, math.inf) for i in range(n) if not degree[i]]
        if len(stack) != 1 or any(x > 1 for x in degree):
            return None
        ans = trees[stack[0][0]]
        while stack:
            i, low, high = stack.pop()
            if not low < trees[i].val < high:
                return None
            if trees[i].left:
                if trees[i].left.val in ind:
                    degree[ind[trees[i].left.val]] -= 1
                    trees[i].left = trees[ind[trees[i].left.val]]
                    stack.append((ind[trees[i].left.val], low, trees[i].val))
                else:
                    if not low < trees[i].left.val < trees[i].val:
                        return None
            if trees[i].right:
                if trees[i].right.val in ind:
                    degree[ind[trees[i].right.val]] -= 1
                    trees[i].right = trees[ind[trees[i].right.val]]
                    stack.append((ind[trees[i].right.val], trees[i].val, high))
                else:
                    if not trees[i].val < trees[i].right.val < high:
                        return
        return ans if all(x == 0 for x in degree) else None

    @staticmethod
    def ac_3831(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/3831/
        tag: topological_sort|dag_dp|circle_judge
        """
        m, n = ac.read_list_ints()
        ind = {w: i for i, w in enumerate("QWER")}
        grid = [ac.read_str() for _ in range(m)]

        dct = [[] for _ in range(m * n)]
        degree = [0] * (m * n)
        for i in range(m):
            for j in range(n):
                x = ind[grid[i][j]]
                for a, b in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                    if 0 <= a < m and 0 <= b < n:
                        y = ind[grid[a][b]]
                        if y == (x + 1) % 4:
                            dct[i * n + j].append(a * n + b)
                            degree[a * n + b] += 1

        pre = [0] * (m * n)
        stack = [i for i in range(m * n) if not degree[i]]
        for i in stack:
            if grid[i // n][i % n] == "Q":
                pre[i] = 1
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if pre[i] > pre[j]:
                        pre[j] = pre[i]
                    if not degree[j]:
                        if not pre[j]:
                            if grid[j // n][j % n] == "Q":
                                pre[j] = 1
                        else:
                            pre[j] += 1
                        nex.append(j)
            stack = nex[:]
        if any(d > 0 for d in degree):
            ac.st("math.infinity")
            return
        ans = max(x // 4 for x in pre)
        ac.st(ans if ans else "none")
        return

    @staticmethod
    def ac_4629(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4629/
        tag: directed_circle_based_tree|circle_judge|classical|brain_teaser
        """
        n = ac.read_int()
        a = ac.read_list_ints_minus_one()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for i in range(n):
            dct[i].append(a[i])
            degree[a[i]] += 1
        if any(d == 0 for d in degree):
            ac.st(-1)
            return

        ans = 1
        for i in range(n):
            if degree[i] == 0:
                continue
            stack = [i]
            degree[i] = 0
            cur = 1
            while stack:
                x = stack.pop()
                for j in dct[x]:
                    if degree[j] > 0:
                        degree[j] = 0
                        cur += 1
                        stack.append(j)
            if cur % 2:
                ans = math.lcm(cur, ans)
            else:
                ans = math.lcm(cur // 2, ans)
        ac.st(ans)
        return

    @staticmethod
    def lc_1591(grid: List[List[int]]) -> bool:
        """
        url: https://leetcode.cn/problems/strange-printer-ii/
        tag: build_graph|topological_sort|circle_judge
        """
        color = defaultdict(list)
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                color[grid[i][j]].append([i, j])

        pos = defaultdict(list)
        for c in color:
            lst = color[c]
            x1 = min(x for x, _ in lst)
            x2 = max(x for x, _ in lst)

            y1 = min(x for _, x in lst)
            y2 = max(x for _, x in lst)
            pos[c] = [x1, x2, y1, y2]

        degree = defaultdict(int)
        dct = defaultdict(list)
        for x in pos:
            a1, a2, b1, b2 = pos[x]
            for y in pos:
                if x != y:
                    for a, b in color[y]:
                        if a1 <= a <= a2 and b1 <= b <= b2:
                            dct[x].append(y)
                            degree[y] += 1
        stack = [x for x in pos if not degree[x]]
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex[:]
        return all(degree[x] == 0 for x in pos)

    @staticmethod
    def abc_256e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc256/tasks/abc256_e
        tag: topological_sort|greedy|circle_based_tree|classical
        """
        n = ac.read_int()
        x = ac.read_list_ints_minus_one()
        c = ac.read_list_ints()
        degree = [0] * n
        for i in range(n):
            degree[x[i]] += 1
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            i = stack.pop()
            degree[x[i]] -= 1
            if not degree[x[i]]:
                stack.append(x[i])
        ans = 0
        for i in range(n):
            if degree[i]:
                p = i
                cur = c[p]
                degree[p] = 0
                while x[p] != i:
                    p = x[p]
                    cur = min(cur, c[p])
                    degree[p] = 0
                ans += cur
        ac.st(ans)
        return

    @staticmethod
    def cf_1388d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1388/D
        tag: topological_sort|dag_dp|heuristic_merge|classical
        """
        n = ac.read_int()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        degree = [0] * n
        for i in range(n):
            if b[i] != -1:
                degree[b[i] - 1] += 1
        res = []
        ans = 0
        stack = [i for i in range(n) if degree[i] == 0]
        ind = [-1] * n
        path = [[] for _ in range(n)]
        for i in stack:
            ind[i] = i
            path[i].append(i)
        while stack:
            nex = []
            for i in stack:
                ans += a[i]
            for i in stack:
                j = b[i]
                if j == -1:
                    res.extend(path[ind[i]][::-1])
                    continue
                j -= 1
                if a[i] < 0:
                    res.extend(path[ind[i]][::-1])
                    path[ind[i]] = []
                    ind[i] = -1
                degree[j] -= 1
                a[j] += max(a[i], 0)
                if ind[j] == -1 and ind[i] != -1:
                    ind[j] = ind[i]
                elif ind[i] != -1 and ind[j] != -1:
                    x, y = ind[i], ind[j]
                    if len(path[x]) < len(path[y]):
                        path[y].extend(path[x])
                        ind[j] = y
                    else:
                        path[x].extend(path[y])
                        ind[j] = x
                elif ind[j] == -1:
                    ind[j] = j
                if degree[j] == 0:
                    nex.append(j)
                    path[ind[j]].append(j)
            stack = nex[:]
        ac.st(ans)
        ac.lst([x + 1 for x in res[::-1]])
        return

    @staticmethod
    def cf_1335f(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1335/F
        tag: circle_based_tree|implemention|observation|greedy
        """
        move = dict()
        move["U"] = (-1, 0)
        move["D"] = (1, 0)
        move["L"] = (0, -1)
        move["R"] = (0, 1)
        tm = 10 ** 9
        for _ in range(ac.read_int()):
            m, n = ac.read_list_ints()
            color = [ac.read_str() for _ in range(m)]
            grid = [ac.read_str() for _ in range(m)]
            ans = [0, 0]
            dct = [-1] * m * n
            rev = [[] for _ in range(m * n)]
            degree = [0] * m * n
            for i in range(m):
                for j in range(n):
                    w = move[grid[i][j]]
                    dct[i * n + j] = (i + w[0]) * n + j + w[1]
                    rev[(i + w[0]) * n + j + w[1]].append(i * n + j)
                    degree[(i + w[0]) * n + j + w[1]] += 1
            stack = [i for i in range(m * n) if degree[i] == 0]
            while stack:
                nex = []
                for i in stack:
                    j = dct[i]
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
                stack = nex
            dis = [-1] * m * n
            parent = [-1] * m * n
            index = [-1] * m * n
            for i in range(m * n):
                if degree[i]:
                    lst = [i]
                    while len(lst) == 1 or lst[-1] != i:
                        lst.append(dct[lst[-1]])
                    lst.pop()
                    k = len(lst)
                    for ii, x in enumerate(lst):
                        index[x] = ii
                        degree[x] = 0
                    ans[0] += len(lst)
                    stack = lst[:]
                    for x in stack:
                        dis[x] = 0
                        parent[x] = x
                    cur = set()
                    while stack:
                        for x in stack:
                            pi = index[parent[x]]
                            if color[x // n][x % n] == "0":
                                cur.add((tm - dis[x] + dis[parent[x]] + pi) % k)
                        nex = []
                        for x in stack:
                            for y in rev[x]:
                                if dis[y] == -1:
                                    dis[y] = dis[x] + 1
                                    parent[y] = parent[x]
                                    nex.append(y)
                        stack = nex
                    ans[1] += len(cur)
            ac.lst(ans)
        return