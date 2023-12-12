"""

Algorithm：lca|multiplication_method|tree_chain_split|tree_centroid|offline_lca|tree_diff_array
Description：tree_diff_array_edge|tree_diff_array_point

====================================LeetCode====================================
1483（https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/）sparse_table|tree_array|lca|tree_lca|classical
2646（https://leetcode.cn/problems/minimize-the-total-price-of-the-trips/）offline_lca|tree_diff_array|counter|tree_dp

=====================================LuoGu======================================
P3379（https://www.luogu.com.cn/problem/P3379）tree_lca|classical
P7128（https://www.luogu.com.cn/problem/P7128）lca|implemention|sort
P3128（https://www.luogu.com.cn/problem/P3128）offline_lca|tree_diff_array
P7167（https://www.luogu.com.cn/problem/P7167）monotonic_stack|tree_lca|build_tree
P3384（https://www.luogu.com.cn/problem/P3384）tree_chain_split|tree_array|implemention
P2912（https://www.luogu.com.cn/problem/P2912）offline_lca|offline_query
P3019（https://www.luogu.com.cn/problem/P3019）offline_query|lca
P3258（https://www.luogu.com.cn/problem/P3258）offline_lca|tree_diff_array|tree_dp
P6869（https://www.luogu.com.cn/problem/P6869）offline_lca|tree_diff_array_edge|tree_diff_array_point

======================================LibreOJ==================================
（https://loj.ac/p/10135）lca

===================================CodeForces===================================
1328E（https://codeforces.com/problemset/problem/1328/E）tree_lca|dfs_order
321C（https://codeforces.com/problemset/problem/321/C）tree_centroid_recursion|classical
519E（https://codeforces.com/problemset/problem/519/E）lca|kth_ancestor|counter
1296F（https://codeforces.com/contest/1296/problem/F）offline_lca|greedy|construction|multiplication_method

====================================AtCoder=====================================
ABC70D（https://atcoder.jp/contests/abc070/tasks/abc070_d）classical|lca|offline_lca

=====================================AcWing=====================================
4202（https://www.acwing.com/problem/content/4205/）bit_operation|build_graph|tree_lca|tree_dis


"""
from typing import List

from src.graph.tree_diff_array.template import TreeDiffArray
from src.graph.tree_lca.template import OfflineLCA
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_6738(n: int, edges: List[List[int]], price: List[int], trips: List[List[int]]) -> int:

        # offline_lca|tree_diff_array|tree_dp
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        # offline_lca
        res = OfflineLCA().bfs_iteration(dct, trips)
        # res = OfflineLCA().dfs_recursion(dct, trips)   # 也可以recursion

        # tree_diff_array
        m = len(trips)
        queries = [trips[i] + [res[i]] for i in range(m)]
        cnt = TreeDiffArray().bfs_iteration(dct, queries)
        # cnt = TreeDiffArray().dfs_recursion(dct, queries)  # 也可以recursion

        # 迭代版的tree_dp
        stack = [0]
        sub = [[] for _ in range(n)]
        parent = [-1] * n
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in dct[i]:
                    if j != parent[i]:
                        parent[j] = i
                        stack.append(j)
            else:
                i = ~i
                res = [cnt[i] * price[i], cnt[i] * price[i] // 2]
                for j in dct[i]:
                    if j != parent[i]:
                        a, b = sub[j]
                        res[0] += a if a < b else b
                        res[1] += a
                sub[i] = res

        return min(sub[0])

    @staticmethod
    def lg_p3128(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3128
        tag: offline_lca|tree_diff_array
        """
        # offline_lca|tree_diff_array
        n, k = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        queries = [ac.read_list_ints_minus_one() for _ in range(k)]
        res = OfflineLCA().bfs_iteration(dct, queries)
        # res = OfflineLCA().dfs_recursion(dct, trips)  # 也可以recursion
        queries = [queries[i] + [res[i]] for i in range(k)]
        cnt = TreeDiffArray().bfs_iteration(dct, queries)
        # cnt = TreeDiffArray().dfs_recursion(dct, queries)  # 也可以recursion
        ac.st(max(cnt))
        return

    @staticmethod
    def lc_2646(n: int, edges: List[List[int]], price: List[int], trips: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimize-the-total-price-of-the-trips/
        tag: offline_lca|tree_diff_array|counter|tree_dp
        """
        # offline_lca与tree_diff_arraycounter，再tree_dp| 
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)
        res = OfflineLCA().bfs_iteration(dct, trips)
        m = len(trips)
        queries = [trips[i] + [res[i]] for i in range(m)]
        cnt = TreeDiffArray().bfs_iteration(dct, queries)

        stack = [[0, 1]]
        sub = [[] for _ in range(n)]
        parent = [-1] * n
        while stack:
            i, state = stack.pop()
            if state:
                stack.append([i, 0])
                for j in dct[i]:
                    if j != parent[i]:
                        parent[j] = i
                        stack.append([j, 1])
            else:
                res = [cnt[i] * price[i], cnt[i] * price[i] // 2]
                for j in dct[i]:
                    if j != parent[i]:
                        a, b = sub[j]
                        res[0] += a if a < b else b
                        res[1] += a
                sub[i] = res

        return min(sub[0])