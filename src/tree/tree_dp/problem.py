"""
Algorithm：tree_dp|tree_diameter|tree_diff_array|tree_centroid
Description：reroot_dp|up_to_down|down_to_up

====================================LeetCode====================================
2440（https://leetcode.cn/problems/create-components-with-same-value/）tree_dp|number_theory|recursion|union_find|brute_force
1569 https://leetcode.cn/problems/number-of-ways-to-reorder-array-to-get-same-bst/solution/）counter|comb|binary_search_tree|tree_dp
968（https://leetcode.cn/problems/binary-tree-cameras/）tree_dp
2538（https://leetcode.cn/problems/difference-between-maximum-and-minimum-price-sum/）reroot_dp|tree_diameter
124（https://leetcode.cn/problems/binary-tree-maximum-path-sum/）tree_dp
2378（https://leetcode.cn/problems/choose-edges-to-maximize-score-in-a-tree/）tree_dp
2445（https://leetcode.cn/problems/number-of-nodes-with-value-one/）up_to_down|tree_dp|implemention
834（https://leetcode.cn/problems/sum-of-distances-in-tree/）tree_dis|tree_centroid
2003（https://leetcode.cn/problems/smallest-missing-genetic-value-in-each-subtree/）tree_dp|heuristic_merge|classical
2673（https://leetcode.cn/problems/make-costs-of-paths-equal-in-a-binary-tree/）tree_dp|greedy
1367（https://leetcode.cn/problems/linked-list-in-binary-tree/description/）classical|2-tree|linked_list|memory_dp
979（https://leetcode.cn/problems/distribute-coins-in-binary-tree/description/）tree_dp|greedy
1373（https://leetcode.cn/problems/maximum-sum-bst-in-binary-tree/）tree_dp|2-tree
971（https://leetcode.cn/problems/flip-binary-tree-to-match-preorder-traversal/description/）tree_dp|greedy|implemention
100041（https://www.acwing.com/problem/content/description/4384/）reroot_dp|dfs_order|diff_array
100047（https://leetcode.cn/problems/count-valid-paths-in-a-tree/description/）tree_dp|union_find|bfs
3241（https://leetcode.cn/problems/time-taken-to-mark-all-nodes/）reroot_dp|classical
100430（https://leetcode.cn/problems/find-subtree-sizes-after-changes/）tree_dp|build_graph

=====================================LuoGu======================================
P1395（https://www.luogu.com.cn/problem/P1395）tree_dis|tree_centroid|reroot_dp|classical|up_to_down|down_to_up
P1352（https://www.luogu.com.cn/problem/P1352）tree_dp|mis|maximum_independent_set
P1922（https://www.luogu.com.cn/problem/P1922）tree_dp|greedy
P2016（https://www.luogu.com.cn/problem/P2016）tree_dp|classical
P1122（https://www.luogu.com.cn/problem/P1122）tree_dp|union_find
P2932（https://www.luogu.com.cn/problem/P2932）tree_dp|greedy
P2996（https://www.luogu.com.cn/problem/P2996）tree_dp
P3074（https://www.luogu.com.cn/problem/P3074）longest_path|tree_dp
P3884（https://www.luogu.com.cn/problem/P3884）tree_dp
P3915（https://www.luogu.com.cn/problem/P3915）recursion|union_find|brute_force
P4615（https://www.luogu.com.cn/problem/P4615）tree_dp
P5002（https://www.luogu.com.cn/problem/P5002）tree_dp|inclusion_exclusion|counter
P5651（https://www.luogu.com.cn/problem/P5651）brain_teaser|union_find|tree_dp|simple_path_xor
P6591（https://www.luogu.com.cn/problem/P6591）reroot_dp|recursion
P7159（https://www.luogu.com.cn/problem/P7159）tree_dp|brute_force|counter|fast_power
P2015（https://www.luogu.com.cn/problem/P2015）tree_dp|tree_bag_dp
P2014（https://www.luogu.com.cn/problem/P2014）tree_dp|tree_bag_dp
P1351（https://www.luogu.com.cn/problem/P1351）brute_force
P3408（https://www.luogu.com.cn/problem/P3408）tree_dp
P3478（https://www.luogu.com.cn/problem/P3478）tree_centroid
P3931（https://www.luogu.com.cn/problem/P3931）classical|tree_dp
P4084（https://www.luogu.com.cn/problem/P4084）classical|tree_dp
P4395（https://www.luogu.com.cn/problem/P4395）tree_dp|greedy
P5765（https://www.luogu.com.cn/problem/P5765）tree_dp|P4395
P8602（https://www.luogu.com.cn/problem/P8602）tree_diameter|bfs|tree_dp
P8625（https://www.luogu.com.cn/problem/P8625）tree_dp|classical
P8744（https://www.luogu.com.cn/problem/P8744）tree_dp
P3047（https://www.luogu.com.cn/problem/P3047）reroot_dp|classical
U420033（https://www.luogu.com.cn/problem/U420033）reroot_dp|classical

====================================AtCoder=====================================
ABC222F（https://atcoder.jp/contests/abc222/tasks/abc222_f）reroot_dp
ABC333D（https://atcoder.jp/contests/abc333/tasks/abc333_d）tree_dp|greedy
ABC329F（https://atcoder.jp/contests/abc329/tasks/abc329_f）heuristic_merge|classical|implemention
ABC348E（https://atcoder.jp/contests/abc348/tasks/abc348_e）reroot_dp|classical
ABC259F（https://atcoder.jp/contests/abc259/tasks/abc259_f）tree_dp|brain_teaser|greedy
ABC239E（https://atcoder.jp/contests/abc239/tasks/abc239_e）tree_dp|classical
ABC222F（https://atcoder.jp/contests/abc222/tasks/abc222_f）reroot_dp|classical
ABC220F（https://atcoder.jp/contests/abc220/tasks/abc220_f）reroot_dp|classical
ABC218F（https://atcoder.jp/contests/abc218/tasks/abc218_f）tree_dp|game_dp|implemention|dfs_order|median
ABC359G（https://atcoder.jp/contests/abc359/tasks/abc359_g）heuristic_merge|classical

===================================CodeForces===================================
1388C（https://codeforces.com/problemset/problem/1388/C）tree_dp|implemention|recursion|down_to_up|up_to_down
1324F（https://codeforces.com/problemset/problem/1324/F）reroot_dp|dfs|down_to_up|up_to_down
337D（https://codeforces.com/problemset/problem/337/D）reroot_dp|dfs|down_to_up|up_to_down
1187E（https://codeforces.com/problemset/problem/1187/E）reroot_dp|dfs|down_to_up|up_to_down
600E（https://codeforces.com/problemset/problem/600/E）heuristic_merge
1676G（https://codeforces.com/contest/1676/problem/G）tree_dp
1822F（https://codeforces.com/contest/1822/problem/F）tree_dis|reroot_dp|down_to_up|up_to_down
219D（https://codeforces.com/contest/219/problem/D）reroot_dp|dfs_order|diff_array
1092F（https://codeforces.com/contest/1092/problem/F）tree_dis|reroot_dp
1472G（https://codeforces.com/contest/1472/problem/G）shortest_path|dfs|down_to_up|up_to_down|brain_teaser
1833G（https://codeforces.com/contest/1833/problem/G）tree_dp|construction
1881F（https://codeforces.com/contest/1881/problem/F）reroot_dp|tree_dp
1926G（https://codeforces.com/contest/1926/problem/G）tree_dp|classical
161D（https://codeforces.com/problemset/problem/161/D）tree_dp|counter
1923E（https://codeforces.com/contest/1923/problem/E）heuristic_merge|tree_dp|counter|classical
1984E（https://codeforces.com/contest/1984/problem/E）reroot_dp|mis|maximum_independent_set
1363E（https://codeforces.com/problemset/problem/1363/E）greedy|implemention|observation
1406C（https://codeforces.com/problemset/problem/1406/C）link_cut_centroids|tree_centroids|greedy|implemention|construction|classical
461B（https://codeforces.com/problemset/problem/461/B）classical|tree_dp|observation
1551F（https://codeforces.com/problemset/problem/1551/F）tree_dp|bag_dp|brute_force
486D（https://codeforces.com/problemset/problem/486/D）multiplication_method|tree_dp|contribution_method|brute_force
1988D（https://codeforces.com/problemset/problem/1988/D）tree_dp|classical|observation|data_range
1101D（https://codeforces.com/problemset/problem/1101/D）tree_dp|prime_factor|classical|observation
1997D（https://codeforces.com/problemset/problem/1997/D）tree_dp|greedy
1083A（https://codeforces.com/problemset/problem/1083/A）tree_dp|greedy|implemention|weighted_tree|classical
982C（https://codeforces.com/problemset/problem/982/C）tree_dp|greedy|classical
1856E1（https://codeforces.com/problemset/problem/1856/E1）tree_dp|greedy|down_to_up|classical
1153D（https://codeforces.com/problemset/problem/1153/D）tree_dp|greedy|brain_teaser
274B（https://codeforces.com/problemset/problem/274/B）tree_dp|greedy|brain_teaser|classical
369C（https://codeforces.com/contest/369/problem/C）tree_dp|greedy|classical
2033G（https://codeforces.com/contest/2033/problem/G）reroot_dp|tree_multiplication|build_graph|brain_teaser|implemention|dfs|back_trace|PointSetPointAddRangeMerge

=====================================AcWing=====================================
3760（https://www.acwing.com/problem/content/description/3763/）brain_teaser|tree_dp
4381（https://www.acwing.com/problem/content/description/4384/）reroot_dp|dfs|dfs_order|diff_array

"""
import math
from collections import Counter
from functools import lru_cache
from typing import List, Optional

from src.basis.tree_node.template import TreeNode
from src.math.prime_factor.template import PrimeFactor
from src.struct.list_node.template import ListNode
from src.struct.sorted_list.template import SortedList
from src.struct.zkw_segment_tree.template import PointSetPointAddRangeMerge
from src.tree.tree_dp.template import WeightedTree
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1676g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1676/problem/G
        tag: tree_dp
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                res = 0
                dp = [0] * self.n
                stack = [0]
                while stack:
                    i = stack.pop()
                    if i >= 0:
                        stack.append(~i)
                        ind = self.point_head[i]
                        while ind:
                            j = self.edge_to[ind]
                            stack.append(j)
                            ind = self.edge_next[ind]
                    else:
                        i = ~i
                        ind = self.point_head[i]
                        while ind:
                            j = self.edge_to[ind]
                            dp[i] += dp[j]
                            ind = self.edge_next[ind]
                        dp[i] += 1 if nums[i] == "B" else -1
                        res += dp[i] == 0
                return res

        for _ in range(ac.read_int()):
            n = ac.read_int()
            graph = Graph(n)
            p = ac.read_list_ints_minus_one()
            for x in range(n - 1):
                graph.add_directed_edge(p[x], x + 1, 1)
            nums = ac.read_str()
            ans = graph.tree_dp()
            ac.st(ans)
        return

    @staticmethod
    def lc_2003(parents: List[int], nums: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/smallest-missing-genetic-value-in-each-subtree/
        tag: tree_dp|heuristic_merge|classical
        """

        class Graph(WeightedTree):
            def heuristic_merge(self):
                dp = [0] * self.n
                sub = [set() for _ in range(self.n)]
                stack = [0]
                while stack:
                    u = stack.pop()
                    if u >= 0:
                        stack.append(~u)
                        ind = self.point_head[u]
                        while ind:
                            v = self.edge_to[ind]
                            stack.append(v)
                            ind = self.edge_next[ind]
                    else:
                        u = ~u
                        ind = self.point_head[u]
                        res = 1
                        pre = {nums[u]}
                        while ind:
                            v = self.edge_to[ind]
                            cur = sub[v]
                            if len(cur) > len(pre):
                                pre, cur = cur, pre
                            pre.update(cur)
                            sub[v] = set()
                            res = max(res, dp[v])
                            ind = self.edge_next[ind]
                        while res in pre:
                            res += 1
                        sub[u] = pre
                        dp[u] = res
                return dp

        n = len(parents)
        graph = Graph(n)
        for i in range(1, n):
            graph.add_directed_edge(parents[i], i, 1)
        ans = graph.heuristic_merge()
        return ans

    @staticmethod
    def cf_1388c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1388/C
        tag: tree_dp|implemention|recursion|down_to_up|up_to_down
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                res = 1
                pos = [0] * n
                neg = [0] * n
                self.parent = [-1] * n
                stack = [0]
                while stack and res:
                    i = stack.pop()
                    if i >= 0:
                        stack.append(~i)
                        ind = self.point_head[i]
                        while ind:
                            j = self.edge_to[ind]
                            if j != self.parent[i]:
                                self.parent[j] = i
                                stack.append(j)
                            ind = self.edge_next[ind]
                    else:
                        i = ~i
                        a = b = 0
                        ind = self.point_head[i]
                        while ind:
                            j = self.edge_to[ind]
                            if j != self.parent[i]:
                                a += pos[j]
                                b += neg[j]
                            ind = self.edge_next[ind]
                        if (h[i] + person[i] + b + a) % 2:
                            res = 0
                            break
                        good = (h[i] + person[i] + b + a) // 2
                        bad = person[i] + a + b - good
                        if good < 0 or bad < 0 or bad > person[i] + b:
                            res = 0
                            break
                        pos[i] = good
                        neg[i] = bad
                return res

        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            person = ac.read_list_ints()
            h = ac.read_list_ints()
            graph = Graph(n)
            for _ in range(n - 1):
                x, y = ac.read_list_ints_minus_one()
                graph.add_undirected_edge(x, y, 1)
            ans = graph.tree_dp()
            ac.st("YES" if ans else "NO")
        return

    @staticmethod
    def cf_161d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/161/D
        tag: tree_dp|counter
        """

        class Graph(WeightedTree):
            def tree_dp(self):

                def idx(ii, jj):
                    return ii * (k + 1) + jj

                dp = [0] * (k + 1) * n
                self.parent = [-1] * n
                stack = [0]
                res = 0
                while stack:
                    i = stack.pop()
                    if i >= 0:
                        stack.append(~i)
                        ind = self.point_head[i]
                        while ind:
                            j = self.edge_to[ind]
                            if j != self.parent[i]:
                                self.parent[j] = i
                                stack.append(j)
                            ind = self.edge_next[ind]
                    else:
                        i = ~i
                        ind = self.point_head[i]
                        while ind:
                            j = self.edge_to[ind]
                            if j != self.parent[i]:
                                for s in range(1, k + 1):
                                    dp[idx(i, s)] += dp[idx(j, s - 1)]
                            ind = self.edge_next[ind]
                        dp[idx(i, 0)] = 1
                        res += dp[idx(i, k)]
                        ind = self.point_head[i]
                        cur = 0
                        while ind:
                            j = self.edge_to[ind]
                            if j != self.parent[i]:
                                for s in range(1, k):
                                    cur += dp[idx(j, s - 1)] * (dp[idx(i, k - s)] - dp[idx(j, k - s - 1)])
                            ind = self.edge_next[ind]
                        res += cur // 2
                return res

        n, k = ac.read_list_ints()
        graph = Graph(n)
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(x, y, 1)
        ans = graph.tree_dp()
        ac.st(ans)
        return

    @staticmethod
    def cf_1324f(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1324/F
        tag: reroot_dp|dfs|down_to_up|up_to_down
        """

        class Graph(WeightedTree):
            def reroot_dp(self):
                dp = [0] * self.n
                self.parent = [-1] * self.n
                stack = [0]
                while stack:
                    u = stack.pop()
                    if u >= 0:
                        stack.append(~u)
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                self.parent[v] = u
                                stack.append(v)
                    else:
                        u = ~u
                        dp[u] = nums[u]
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                dp[u] += max(0, dp[v])

                ndp = dp[:]
                stack = [0]
                while stack:
                    u = stack.pop()
                    for v in self.get_to_nodes(u):
                        if v != self.parent[u]:
                            ndp[v] = max(0, ndp[u] - max(0, dp[v])) + dp[v]
                            stack.append(v)
                return ndp

        n = ac.read_int()
        nums = [x * 2 - 1 for x in ac.read_list_ints()]
        graph = Graph(n)
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(x, y, 1)
        ans = graph.reroot_dp()
        ac.lst(ans)
        return

    @staticmethod
    def cf_337d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/337/D
        tag: reroot_dp|dfs|down_to_up|up_to_down
        """

        class Graph(WeightedTree):
            def reroot_dp(self):
                dp = [0] * self.n
                self.parent = [-1] * self.n
                stack = [0]
                while stack:
                    u = stack.pop()
                    if u >= 0:
                        stack.append(~u)
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                self.parent[v] = u
                                stack.append(v)
                    else:
                        u = ~u
                        dp[u] = nums[u]
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                dp[u] += max(0, dp[v])

                ndp = dp[:]
                stack = [0]
                while stack:
                    u = stack.pop()
                    for v in self.get_to_nodes(u):
                        if v != self.parent[u]:
                            ndp[v] = max(0, ndp[u] - max(0, dp[v])) + dp[v]
                            stack.append(v)
                return ndp

        n = ac.read_int()
        nums = [x * 2 - 1 for x in ac.read_list_ints()]
        graph = Graph(n)
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(x, y, 1)
        ans = graph.reroot_dp()
        ac.lst(ans)
        return

    @staticmethod
    def cf_1092f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1092/problem/F
        tag: tree_dis|reroot_dp
        """
        n = ac.read_int()
        weights = ac.read_list_ints()
        graph = WeightedTree(n)
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(x, y, 1)
        ans = graph.reroot_dp_for_tree_dis_with_node_weights(weights)
        ac.st(max(ans))
        return

    @staticmethod
    def lc_968(root: Optional[TreeNode]) -> int:
        """
        url: https://leetcode.cn/problems/binary-tree-cameras/
        tag: tree_dp
        """

        def dfs(node):
            if not node:
                return [0, math.inf, 0]
            left = dfs(node.left)
            right = dfs(node.right)
            res = [math.inf, math.inf, math.inf]
            res[0] = min(left[1] + min(right[0], right[1]), right[1] + min(left[0], left[1]))
            res[1] = 1 + min(left) + min(right)
            res[2] = left[0] + right[0]
            return res

        ans = dfs(root)
        return min(ans[0], ans[1])

    @staticmethod
    def lc_1367(head: Optional[ListNode], root: Optional[TreeNode]) -> bool:
        """
        url: https://leetcode.cn/problems/linked-list-in-binary-tree/description/
        tag: classical|2-tree|linked_list|memory_dp
        """

        @lru_cache(None)
        def dfs(lst, node):
            if not lst:
                return True
            if not node:
                return False
            if lst.val == node.val:
                if dfs(lst.next, node.left) or dfs(lst.next, node.right):
                    return True
            return dfs(head, node.left) or dfs(head, node.right)

        return dfs(head, root)

    @staticmethod
    def cf_1187e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1187/E
        tag: reroot_dp|dfs|down_to_up|up_to_down
        """
        n = ac.read_int()
        graph = WeightedTree(n)
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(x, y, 1)
        ans = graph.reroot_dp_for_tree_dis_with_node_weights([1] * n)
        ac.st(max(ans) + n)
        return

    @staticmethod
    def cf_600e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/600/E
        tag: heuristic_merge
        """

        class Graph(WeightedTree):
            def heuristic_merge(self):
                dp = [0] * self.n
                sub = [None] * self.n
                cnt = [0] * self.n
                self.parent = [-1] * self.n
                stack = [0]
                while stack:
                    u = stack.pop()
                    if u >= 0:
                        stack.append(~u)
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                self.parent[v] = u
                                stack.append(v)
                    else:
                        u = ~u
                        pre = {nums[u]: 1}
                        cnt[u] = 1
                        dp[u] = nums[u]
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                cur = sub[v]
                                if len(pre) < len(cur):
                                    dp[u] = dp[v]
                                    cnt[u] = cnt[v]
                                    pre, cur = cur, pre
                                for color in cur:
                                    pre[color] = pre.get(color, 0) + cur[color]
                                    if pre[color] > cnt[u]:
                                        cnt[u] = pre[color]
                                        dp[u] = color
                                    elif pre[color] == cnt[u]:
                                        dp[u] += color
                                sub[v] = None
                        sub[u] = pre
                return dp

        n = ac.read_int()
        graph = Graph(n)
        nums = ac.read_list_ints()
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(x, y, 1)
        ans = graph.heuristic_merge()
        ac.lst(ans)
        return

    @staticmethod
    def lg_p1395_1(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1395
        tag: tree_dis|tree_centroid|reroot_dp|classical|up_to_down|down_to_up
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                # the smallest centroid of tree
                # equal the node with minimum of maximum subtree node cnt
                # equivalent to the node which has the shortest distance from all other nodes
                sub = [1] * self.n  # subtree size of i-th node rooted by 0
                ma = [0] * self.n  # maximum subtree node cnt or i-rooted
                ma[0] = n
                res = 0
                self.parent = [-1] * n
                stack = [0]
                while stack:
                    i = stack.pop()
                    if i >= 0:
                        stack.append(~i)
                        for j in self.get_to_nodes(i):
                            if j != self.parent[i]:
                                self.parent[j] = i
                                stack.append(j)
                    else:
                        i = ~i
                        for j in self.get_to_nodes(i):
                            if j != self.parent[i]:
                                sub[i] += sub[j]
                                ma[i] = ma[i] if ma[i] > sub[j] else sub[j]
                        # like re-rooted dp to check the maximum subtree size
                        ma[i] = ma[i] if ma[i] > n - sub[i] else n - sub[i]
                        if ma[i] < ma[res] or (ma[i] == ma[res] and i < res):
                            res = i
                return res

            def bfs_for_distance(self, src):
                stack = [src]
                dis = [0] * n
                self.parent = [-1] * n
                while stack:
                    u = stack.pop()
                    for v in self.get_to_nodes(u):
                        if v != self.parent[u]:
                            dis[v] = dis[u] + 1
                            self.parent[v] = u
                            stack.append(v)
                return sum(dis)

        n = ac.read_int()
        graph = Graph(n)
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(x, y, 1)
        center = graph.tree_dp()
        ans = graph.bfs_for_distance(center)
        ac.lst([center + 1, ans])
        return

    @staticmethod
    def lg_p1395_2(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1395
        tag: tree_dis|tree_centroid|reroot_dp|classical|up_to_down|down_to_up
        """
        n = ac.read_int()
        graph = WeightedTree(n)
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(x, y, 1)
        weights = [1] * n
        ans = graph.reroot_dp_for_tree_dis_with_node_weights(weights)
        ind = ans.index(min(ans))
        ac.lst([ind + 1, ans[ind]])
        return

    @staticmethod
    def cf_1822f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1822/problem/F
        tag: tree_dis|reroot_dp|down_to_up|up_to_down
        """
        for _ in range(ac.read_int()):
            n, k, c = ac.read_list_ints()
            graph = WeightedTree(n)
            for _ in range(n - 1):
                i, j = ac.read_list_ints_minus_one()
                graph.add_undirected_edge(i, j, 1)

            dis = graph.reroot_dp_for_tree_dis_with_node_weights_maximum([1] * n)

            ans = -math.inf
            stack = [0]
            parent = [-1] * n
            cost = [0] * n
            while stack:
                i = stack.pop()
                cur = dis[i] * k - cost[i]
                ans = max(ans, cur)
                for j in graph.get_to_nodes(i):
                    if j != parent[i]:
                        cost[j] = cost[i] + c
                        stack.append(j)
                        parent[j] = i
            ac.st(ans)
        return

    @staticmethod
    def lg_p1352(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1352
        tag: tree_dp|mis|maximum_independent_set
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                dp0 = [0] * self.n
                dp1 = [0] * self.n
                self.parent = [-1] * self.n
                stack = [root]
                while stack:
                    u = stack.pop()
                    if u >= 0:
                        stack.append(~u)
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                self.parent[v] = u
                                stack.append(v)
                    else:
                        u = ~u
                        ind = self.point_head[u]
                        x = max(nums[u], 0)
                        y = 0
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                x += dp0[v]
                                y += dp1[v]
                            ind = self.edge_next[ind]
                        dp0[u] = y
                        dp1[u] = max(x, y)
                return max(dp0[root], dp1[root])

        n = ac.read_int()
        graph = Graph(n)
        nums = [ac.read_int() for _ in range(n)]
        degree = [0] * n
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            graph.add_directed_edge(j, i, 1)
            degree[i] += 1
        root = [i for i in range(n) if not degree[i]][0]
        ans = graph.tree_dp()
        ac.st(ans)
        return

    @staticmethod
    def lg_p2015(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2015
        tag: tree_dp|tree_bag_dp
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                dp = [[0] * (q + 1) for _ in range(n)]
                self.parent = [-1] * self.n
                stack = [0]
                while stack:
                    u = stack.pop()
                    if u >= 0:
                        stack.append(~u)
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                self.parent[v] = u
                                stack.append(v)
                    else:
                        u = ~u
                        lst = self.get_to_nodes_weights(u)
                        lst = [(v, weight) for v, weight in lst if v != self.parent[u]]
                        if lst:
                            a, w1 = lst[0]
                            b, w2 = lst[1]
                            for p in range(1, q + 1):
                                cur = max(dp[a][p - 1] + w1, dp[b][p - 1] + w2)
                                for k in range(p - 1):
                                    cur = max(cur, dp[a][k] + dp[b][p - k - 2] + w1 + w2)
                                dp[u][p] = cur

                return dp[0][q]

        n, q = ac.read_list_ints()
        graph = Graph(n)
        for _ in range(n - 1):
            i, j, w = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, w + 1)
        ans = graph.tree_dp()
        ac.st(ans)
        return

    @staticmethod
    def lg_p2014(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2014
        tag: tree_dp|tree_bag_dp
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                dp = [[0] * (m + 2) for _ in range(n + 1)]
                stack = [0]
                while stack:
                    u = stack.pop()
                    if u >= 0:
                        stack.append(~u)
                        for v in self.get_to_nodes(u):
                            stack.append(v)
                    else:
                        u = ~u
                        dp[u][1] = nums[u]
                        for v in self.get_to_nodes(u):
                            cur = dp[u][:]
                            for x in range(1, m + 2):
                                for y in range(m + 2 - x):
                                    cur[x + y] = max(cur[x + y], dp[u][x] + dp[v][y])
                            dp[u] = cur[:]

                return dp[0][m + 1]

        n, m = ac.read_list_ints()
        graph = Graph(n + 1)
        nums = [0]
        for i in range(n):
            k, s = ac.read_list_ints()
            nums.append(s)
            graph.add_directed_edge(k, i + 1, 1)
        ans = graph.tree_dp()
        ac.st(ans)
        return

    @staticmethod
    def lg_p1351(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1351
        tag: brute_force
        """
        n = ac.read_int()
        graph = WeightedTree(n)
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, 1)
        nums = ac.read_list_ints()
        ceil = ans = 0
        mod = 10007
        for i in range(n):
            lst = [nums[j] for j in graph.get_to_nodes(i)]
            s = sum(lst)
            a = b = 0
            for num in lst:
                ans += num * (s - num)
                ans %= mod
                if num > a:
                    a, b = num, a
                elif num > b:
                    b = num
            ceil = max(ceil, a * b)
        ac.lst([ceil, ans])
        return

    @staticmethod
    def lg_3408(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3408
        tag: tree_dp
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                stack = [0]
                while stack:
                    u = stack.pop()
                    if u >= 0:
                        stack.append(~u)
                        for v in self.get_to_nodes(u):
                            stack.append(v)
                    else:
                        u = ~u
                        nodes = self.get_to_nodes(u)
                        if dp[u] > t or not nodes:
                            continue
                        m = len(nodes)
                        x = math.ceil(m * dp[u] / t)
                        lst = [dp[v] for v in nodes]
                        lst.sort()
                        dp[u] = sum(lst[:x])
                return dp[0]

        n, t, c = ac.read_list_ints()
        graph = Graph(n + 1)
        dp = [c]
        for i in range(n):
            b, a = ac.read_list_ints()
            graph.add_directed_edge(b, i + 1, 1)
            dp.append(a)
        ans = graph.tree_dp()
        ac.st(ans)
        return

    @staticmethod
    def lg_p3478(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3478
        tag: tree_centroid
        """
        n = ac.read_int()
        graph = WeightedTree(n)
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(x, y, 1)
        weights = [1] * n
        ans = graph.reroot_dp_for_tree_dis_with_node_weights(weights)
        ac.st(ans.index(max(ans)) + 1)
        return

    @staticmethod
    def lg_p3931(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3931
        tag: classical|tree_dp
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                stack = [root]
                dp = [inf] * self.n
                self.parent = [-1] * self.n
                while stack:
                    u = stack.pop()
                    if u >= 0:
                        stack.append(~u)
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                self.parent[v] = u
                                stack.append(v)
                    else:
                        u = ~u
                        nodes = self.get_to_nodes(u)
                        if u != root and len(nodes) == 1:
                            continue
                        res = 0
                        for v, weight in self.get_to_nodes_weights(u):
                            if v != self.parent[u]:
                                res += min(weight, dp[v])
                        dp[u] = res
                return dp[root]

        inf = 10 ** 12
        n, root = ac.read_list_ints()
        root -= 1
        graph = Graph(n)
        for _ in range(n - 1):
            i, j, w = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, w + 1)
        ans = graph.tree_dp()
        ac.st(ans)
        return

    @staticmethod
    def lg_p4395(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4395
        tag: tree_dp|greedy
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                ceil = int(math.log2(self.n)) + 1
                stack = [0]
                dp = [[ceil * n] * (ceil + 1) for _ in range(n)]
                self.parent = [-1] * self.n
                while stack:
                    u = stack.pop()
                    if u >= 0:
                        stack.append(~u)
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                self.parent[v] = u
                                stack.append(v)
                    else:
                        u = ~u
                        cur = [0] * (ceil + 1)
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                a = b = ceil * n
                                for c in dp[v][1:]:
                                    if c < a:
                                        a, b = c, a
                                    elif c < b:
                                        b = c
                                for x in range(1, ceil + 1):
                                    if dp[v][x] == a:
                                        cur[x] += b
                                    else:
                                        cur[x] += a
                        for x in range(1, ceil + 1):
                            dp[u][x] = x + cur[x]
                return min(dp[0][1:])

        n = ac.read_int()
        graph = Graph(n)
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, 1)
        ans = graph.tree_dp()
        ac.st(ans)
        return

    @staticmethod
    def lg_p8625(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8625
        tag: tree_dp|classical
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                stack = [0]
                self.parent = [-1] * self.n
                while stack:
                    u = stack.pop()
                    if u >= 0:
                        stack.append(~u)
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                self.parent[v] = u
                                stack.append(v)
                    else:
                        u = ~u
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                dp[u] += dp[v]
                        dp[u] = max(dp[u], 0)
                return max(dp)

        n = ac.read_int()
        dp = ac.read_list_ints()
        graph = Graph(n)
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(i, j, 1)
        ans = graph.tree_dp()
        ac.st(ans)
        return

    @staticmethod
    def ac_3760(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/3763/
        tag: brain_teaser|tree_dp
        """
        n = ac.read_int()
        w = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v, c = ac.read_list_ints()
            u -= 1
            v -= 1
            dct[u].append([v, c])
            dct[v].append([u, c])
        ans = 0

        stack = [[0, -1]]
        sub = [0 for _ in range(n)]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j, cc in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i

                d1, d2 = 0, 0
                for j, cc in dct[i]:
                    if j != fa:
                        d = sub[j] - cc
                        if d >= d1:
                            d1, d2 = d, d1
                        elif d >= d2:
                            d2 = d
                if d1 + d2 + w[i] > ans:
                    ans = d1 + d2 + w[i]
                sub[i] = d1 + w[i]
        ac.st(ans)
        return

    @staticmethod
    def ac_4381(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4384/
        tag: reroot_dp|dfs|dfs_order|diff_array
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            dct[x].append([y, 0])
            dct[y].append([x, 1])

        sub = [0] * n
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j, w in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                cur = 0
                for j, w in dct[i]:
                    if j != fa:
                        cur += sub[j] + w
                sub[i] = cur

        stack = [[0, -1, 0]]
        while stack:
            i, fa, pre = stack.pop()
            cur = sub[i]
            sub[i] += pre
            for j, w in dct[i]:
                if j != fa:
                    stack.append([j, i, pre + cur - sub[j] - w + 1 - w])

        x = min(sub)
        ac.st(x)
        res = [i + 1 for i in range(n) if sub[i] == x]
        ac.lst(res)
        return

    @staticmethod
    def lc_100041(n: int, edges: List[List[int]]) -> List[int]:

        dct = [[] for _ in range(n)]
        for x, y in edges:
            dct[x].append([y, 1])
            dct[y].append([x, 0])

        sub_cnt = [0] * n
        sub_one = [0] * n
        pre_cnt = [0] * n
        pre_one = [0] * n
        stack = [[0, -1]]
        while stack:
            x, fa = stack.pop()
            if x >= 0:
                stack.append([~x, fa])
                for y, w in dct[x]:
                    if y != fa:
                        pre_cnt[y] = pre_cnt[x] + 1
                        pre_one[y] = pre_one[x] + w
                        stack.append([y, x])
            else:
                x = ~x
                sub_cnt[x] = 1
                for y, w in dct[x]:
                    if y != fa:
                        sub_cnt[x] += sub_cnt[y]
                        sub_one[x] += sub_one[y] + w
        ans = [pre_one[i] + (sub_cnt[i] - sub_one[i]) + (
                n - 1 - pre_cnt[i] - sub_cnt[i] - (sub_one[0] - sub_one[i] - pre_one[i])) for i in range(n)]
        return ans

    @staticmethod
    def lc_2673(n: int, cost: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/make-costs-of-paths-equal-in-a-binary-tree/
        tag: tree_dp|greedy
        """
        ans = 0
        for i in range(n // 2, 0, -1):
            left = cost[i * 2 - 1]
            right = cost[i * 2]
            if left > right:
                cost[i - 1] += left
                ans += left - right
            else:
                cost[i - 1] += right
                ans += right - left
        return ans

    @staticmethod
    def cf_1926g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1926/problem/G
        tag: tree_dp|classical
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            p = [0] + ac.read_list_ints_minus_one()
            s = ac.read_str()
            dpp = [0] * n
            dps = [0] * n
            for i in range(n - 1, -1, -1):
                if s[i] == "P":
                    dps[i] = n
                if s[i] == "S":
                    dpp[i] = n
                pp = p[i]
                if i:
                    dpp[pp] += min(dpp[i], dps[i] + 1)
                    dps[pp] += min(dps[i], dpp[i] + 1)
            ac.st(min(dpp[0], dps[0]))
        return

    @staticmethod
    def cf_1923e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1923/problem/E
        tag: heuristic_merge|tree_dp|counter|classical
        """
        ac.get_random_seed()
        for _ in range(ac.read_int()):
            n = ac.read_int()
            c = ac.read_list_ints()
            dct = [[] for _ in range(n)]
            for _ in range(n - 1):
                i, j = ac.read_list_ints_minus_one()
                dct[i].append(j)
                dct[j].append(i)

            sub = [dict() for _ in range(n)]
            ind = list(range(n))
            stack = [(0, -1)]
            ans = 0
            while stack:
                x, fa = stack.pop()
                if x >= 0:
                    stack.append((~x, fa))
                    for y in dct[x]:
                        if y != fa:
                            stack.append((y, x))
                else:
                    x = ~x
                    xx = x
                    cc = c[x] ^ ac.random_seed
                    cur = ind[x]
                    for y in dct[x]:
                        if y != fa:
                            ans += sub[ind[y]].get(cc, 0)
                            if len(sub[ind[y]]) > len(sub[cur]):
                                for x in sub[cur]:
                                    if x != cc:
                                        ans += sub[cur][x] * sub[ind[y]].get(x, 0)
                                for x in sub[cur]:
                                    sub[ind[y]][x] = sub[ind[y]].get(x, 0) + sub[cur][x]
                                cur = ind[y]
                            else:
                                for x in sub[ind[y]]:
                                    if x != cc:
                                        ans += sub[ind[y]][x] * sub[cur].get(x, 0)
                                for x in sub[ind[y]]:
                                    sub[cur][x] = sub[cur].get(x, 0) + sub[ind[y]][x]
                    ind[xx] = cur
                    sub[cur][cc] = 1
            ac.st(ans)
        return

    @staticmethod
    def lg_p3047(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3047
        tag: reroot_dp|classical
        """
        n, k = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            dct[u].append(v)
            dct[v].append(u)
        c = [ac.read_int() for _ in range(n)]
        sub = [[0] * k for _ in range(n)]
        k += 1
        stack = [(0, -1)]
        while stack:
            x, fa = stack.pop()
            if x >= 0:
                stack.append((~x, fa))
                for y in dct[x]:
                    if y != fa:
                        stack.append((y, x))
            else:
                x = ~x
                cur = [0] * k
                for y in dct[x]:
                    if y != fa:
                        for j in range(k - 1):
                            cur[j + 1] += sub[y][j]
                cur[0] += c[x]
                sub[x] = cur[:]

        stack = [[0, -1, [0] * k]]
        while stack:
            x, fa, pre = stack.pop()
            for j in range(k):
                sub[x][j] += pre[j]
            for y in dct[x]:
                if y != fa:
                    for j in range(k - 1):
                        pre[j + 1] += sub[y][j]

            for y in dct[x]:
                if y != fa:
                    nex = pre[:]
                    for j in range(k - 1):
                        nex[j + 1] -= sub[y][j]
                    nex = [0] + nex[:-1]
                    nex[1] += c[x]
                    stack.append([y, x, nex])
        for ls in sub:
            ac.st(sum(ls))
        return

    @staticmethod
    def lg_u420033(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/U420033
        tag: reroot_dp|classical
        """

        def standard_procedure() -> int:
            dct = [[] for _ in range(n)]
            for i, j, w in edges:
                dct[i].append((j, w))
                dct[j].append((i, w))
            father = [-1] * n
            stack = [(0, -1)]
            sub = [[0, 0, 0] for _ in range(n)]
            dia = [0] * n
            while stack:
                x, fa = stack.pop()
                if x >= 0:
                    stack.append((~x, fa))
                    for y, w in dct[x]:
                        if y != fa:
                            stack.append((y, x))
                            father[y] = x
                else:
                    x = ~x
                    a = b = c = d = 0
                    for y, ww in dct[x]:
                        if y != fa:
                            for w in sub[y][:1]:
                                if w + ww >= a:
                                    a, b, c = w + ww, a, b
                                elif w + ww >= b:
                                    b, c = w + ww, b
                                elif w + ww >= c:
                                    c = w + ww
                            d = max(d, dia[y])
                    sub[x] = [a, b, c]
                    d = max(d, a + b)
                    dia[x] = d

            ans = math.inf
            stack = [(0, -1, 0, 0)]
            while stack:
                x, fa, pre, pre_dia = stack.pop()
                a, b, c = sub[x]
                aa = bb = -math.inf
                for y, _ in dct[x]:
                    if y != fa:
                        dd = dia[y]
                        if dd >= aa:
                            aa, bb = dd, aa
                        elif dd >= bb:
                            bb = dd
                for y, w in dct[x]:
                    if y != fa:
                        down = dia[y]
                        if sub[y][0] == a - w:
                            up = max(pre + b, b + c, pre_dia)
                            nex = max(pre, b) + w
                            nex_dia = max(pre_dia, pre + b, b + w, b + c)
                        elif sub[y][0] == b - w:
                            up = max(pre + a, a + c, pre_dia)
                            nex = max(pre, a) + w
                            nex_dia = max(pre_dia, pre + a, a + w, a + c)
                        else:
                            up = max(pre + a, a + b, pre_dia)
                            nex = max(pre, a) + w
                            nex_dia = max(pre_dia, pre + a, a + w, a + b)
                        if dia[y] == aa:
                            up = max(up, bb)
                            nex_dia = max(nex_dia, bb)
                        else:
                            up = max(up, aa)
                            nex_dia = max(nex_dia, aa)
                        ans = min(ans, abs(up - down))
                        stack.append((y, x, nex, nex_dia))
            return ans

        n = ac.read_int()
        edges = [ac.read_list_ints() for _ in range(n - 1)]
        ac.st(standard_procedure())
        return

    @staticmethod
    def abc_348e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc348/tasks/abc348_e
        tag: reroot_dp|classical
        """
        n = ac.read_int()
        graph = WeightedTree(n)
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(x, y, 1)
        weights = ac.read_list_ints()
        ans = graph.reroot_dp_for_tree_dis_with_node_weights(weights)
        ac.st(min(ans))
        return

    @staticmethod
    def abc_259f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc259/tasks/abc259_f
        tag: tree_dp|brain_teaser|greedy
        """
        n = ac.read_int()
        d = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for i in range(n - 1):
            u, v, w = ac.read_list_ints_minus_one()
            w += 1
            dct[u].append((v, w))
            dct[v].append((u, w))
        sub = [(0, 0) for _ in range(n)]
        stack = [(0, -1)]
        while stack:
            x, fa = stack.pop()
            if x >= 0:
                stack.append((~x, fa))
                for y, _ in dct[x]:
                    if y != fa:
                        stack.append((y, x))
            else:
                x = ~x
                pos = 0
                son = []
                for y, w in dct[x]:
                    if y != fa:
                        a, b = sub[y]
                        diff = w + b - a
                        pos += a
                        if diff > 0 and d[y]:
                            son.append(diff)
                son.sort(reverse=True)
                if d[x]:
                    a = pos + sum(son[:d[x]])
                    sub[x] = (a, a - son[d[x] - 1] if len(son) >= d[x] else a)
                else:
                    sub[x] = (pos, 0)
        ac.st(max(sub[0]))
        return

    @staticmethod
    def abc_222f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc222/tasks/abc222_f
        tag: reroot_dp|classical
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j, w = ac.read_list_ints()
            i -= 1
            j -= 1
            dct[i].append((j, w))
            dct[j].append((i, w))
        d = ac.read_list_ints()

        sub = [0] * n
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j, w in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                cur = 0
                for j, w in dct[i]:
                    if j != fa:
                        cur = max(cur, max(sub[j], d[j]) + w)
                sub[i] = cur

        ans = [0] * n
        stack = [[0, -1, 0]]
        while stack:
            i, fa, pre = stack.pop()

            ans[i] = max(pre, sub[i])
            aa = bb = -math.inf
            for j, w in dct[i]:
                if j != fa:
                    cur = max(sub[j], d[j]) + w
                    if cur > aa:
                        aa, bb = cur, aa
                    elif cur > bb:
                        bb = cur
            for j, w in dct[i]:
                if j != fa:
                    cur = max(sub[j], d[j]) + w
                    if cur == aa:
                        stack.append((j, i, max(pre, bb, d[i]) + w))
                    else:
                        stack.append((j, i, max(pre, aa, d[i]) + w))
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def abc_220f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc220/tasks/abc220_f
        tag: reroot_dp|classical
        """
        n = ac.read_int()
        graph = WeightedTree(n)
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(x, y, 1)
        weights = [1] * n
        ans = graph.reroot_dp_for_tree_dis_with_node_weights(weights)
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def abc_218f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc218/tasks/abc218_f
        tag: tree_dp|game_dp|implemention|dfs_order|median
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        depth = [0] * n

        def find_median():
            k = len(lst)
            if k % 2:
                return lst[k // 2]
            return (lst[k // 2] + lst[k // 2 - 1]) // 2

        lst = SortedList()
        stack = [(0, -1)]
        sub = [0] * n
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                lst.add(nums[i])
                stack.append((~i, fa))
                for j in dct[i]:
                    if j != fa:
                        depth[j] = depth[i] + 1
                        stack.append((j, i))
            else:
                i = ~i
                d = depth[i]
                nex = []
                for j in dct[i]:
                    if j != fa:
                        nex.append(sub[j])
                if not nex:
                    sub[i] = find_median()
                else:
                    if d % 2 == 0:
                        sub[i] = max(nex)
                    else:
                        sub[i] = min(nex)
                lst.discard(nums[i])
        ans = sub[0]
        ac.st(ans)
        return

    @staticmethod
    def cf_1984e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1984/problem/E
        tag: reroot_dp|mis|maximum_independent_set
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            dct = [[] for _ in range(n)]
            degree = [0] * n
            for _ in range(n - 1):
                x, y = ac.read_list_ints_minus_one()
                dct[y].append(x)
                dct[x].append(y)
                degree[x] += 1
                degree[y] += 1
            ans = [0] * n
            dp = [(0, 0) for _ in range(n)]  # not_include or include
            stack = [(0, -1)]
            while stack:
                i, fa = stack.pop()
                if i >= 0:
                    stack.append((~i, fa))
                    for j in dct[i]:
                        if j != fa:
                            stack.append((j, i))
                else:
                    i = ~i
                    x = 1
                    y = 0
                    for j in dct[i]:
                        if j != fa:
                            a, b = dp[j]
                            x += a
                            y += b
                    dp[i] = (y, max(x, y))
                    ans[i] = y + 1 if degree[i] == 1 else y

            stack = [(0, -1, 0, 0)]
            while stack:
                i, fa, pre_a, pre_b = stack.pop()
                ans[i] += pre_b
                lst = [j for j in dct[i] if j != fa]
                aa = sum(dp[j][0] for j in lst)
                bb = sum(dp[j][1] for j in lst)
                for j in lst:
                    nex_aa = aa - dp[j][0] + pre_a + 1
                    nex_bb = bb - dp[j][1] + pre_b
                    stack.append((j, i, nex_bb, max(nex_bb, nex_aa)))
            ac.st(max(ans))
        return

    @staticmethod
    def abc_359g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc359/tasks/abc359_g
        tag: heuristic_merge|classical
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            dct[u].append(v)
            dct[v].append(u)
        nums = ac.read_list_ints_minus_one()
        tot = Counter(nums)
        stack = [(0, -1)]
        ans = 0
        sub = [dict() for _ in range(n)]
        index = list(range(n))
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append((~i, fa))
                for j in dct[i]:
                    if j != fa:
                        stack.append((j, i))
            else:
                i = ~i
                ind = index[i]
                sub[ind][nums[i]] = (0, 1)
                for j in dct[i]:
                    if j != fa:
                        xj = index[j]
                        for p in sub[xj]:
                            if p in sub[ind]:
                                dis, cnt = sub[xj][p]
                                dis0, cnt0 = sub[ind].get(p, (0, 0))
                                ans += dis * cnt0 + dis0 * cnt
                        if len(sub[xj]) > len(sub[ind]):
                            nex = ind
                            ind = xj
                        else:
                            nex = xj
                        for p, (dis, cnt) in sub[nex].items():
                            if p in sub[ind]:
                                dis0, cnt0 = sub[ind].get(p, (0, 0))
                                sub[ind][p] = (dis + dis0, cnt + cnt0)
                            else:
                                sub[ind][p] = (dis, cnt)
                        sub[nex] = None
                for p in list(sub[ind].keys()):
                    if tot[p] == sub[ind][p][1]:
                        del sub[ind][p]
                    dis, cnt = sub[ind][p]
                    sub[ind][p] = dis + cnt, cnt

                index[i] = ind
        ac.st(ans)
        return

    @staticmethod
    def cf_1363e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1363/E
        tag: greedy|implemention|observation
        """
        n = ac.read_int()
        a = []
        b = []
        c = []
        for _ in range(n):
            x, y, z = ac.read_list_ints()
            a.append(x)
            b.append(y)
            c.append(z)
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        stack = [(0, -1)]
        sub = [0 for _ in range(n)]
        ans = 0
        while stack:
            x, fa = stack.pop()
            if x >= 0:
                stack.append((~x, fa))
                for y in dct[x]:
                    if y != fa:
                        a[y] = min(a[y], a[x])
                        stack.append((y, x))
            else:
                x = ~x
                pos = neg = 0
                if b[x] and not c[x]:
                    pos += 1
                elif not b[x] and c[x]:
                    neg += 1
                for y in dct[x]:
                    if y != fa:
                        if sub[y] > 0:
                            pos += sub[y]
                        else:
                            neg -= sub[y]
                ans += min(pos, neg) * a[x] * 2
                sub[x] = pos - neg
        ac.st(ans if sub[0] == 0 else -1)
        return

    @staticmethod
    def cf_1406c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1406/C
        tag: link_cut_centroids|tree_centroids|greedy|implemention|construction|classical
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            dct = [[] for _ in range(n)]
            for _ in range(n - 1):
                i, j = ac.read_list_ints_minus_one()
                dct[i].append(j)
                dct[j].append(i)
            stack = [(0, -1)]
            sub = [1] * n
            parent = [-1] * n
            dp = [0] * n
            depth = [0] * n
            while stack:
                x, fa = stack.pop()
                if x >= 0:
                    stack.append((~x, fa))
                    for y in dct[x]:
                        if y != fa:
                            stack.append((y, x))
                            parent[y] = x
                            depth[y] = depth[x] + 1
                else:
                    x = ~x
                    nex = 0
                    for y in dct[x]:
                        if y != fa:
                            sub[x] += sub[y]
                            nex = max(nex, sub[y])
                    dp[x] = max(nex, n - sub[x])
            floor = min(dp)
            ind = [i for i in range(n) if dp[i] == floor]
            if len(ind) == 1:
                ac.lst([1, dct[0][0] + 1])
                ac.lst([1, dct[0][0] + 1])
            else:
                assert len(ind) == 2
                x, y = ind[0], ind[1]
                if depth[x] > depth[y]:
                    x, y = y, x

                a, fa = y, parent[y]
                while True:
                    for b in dct[a]:
                        if b != fa:
                            a, fa = b, a
                            break
                    else:
                        break
                ac.lst([a + 1, parent[a] + 1])
                ac.lst([a + 1, x + 1])
        return

    @staticmethod
    def cf_461b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/461/B
        tag: classical|tree_dp|observation
        """

        n = ac.read_int()
        p = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for i in range(n - 1):
            dct[i + 1].append(p[i])
            dct[p[i]].append(i + 1)
        dp0 = [0] * n
        dp1 = [0] * n
        color = ac.read_list_ints()
        stack = [(0, -1)]
        mod = 10 ** 9 + 7
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append((~i, fa))
                for j in dct[i]:
                    if j != fa:
                        stack.append((j, i))
            else:
                i = ~i
                if color[i]:
                    dp1[i] = 1
                else:
                    dp0[i] = 1
                for j in dct[i]:
                    if j != fa:
                        dp1[i] = (dp1[i] * (dp0[j] + dp1[j]) + dp0[i] * dp1[j]) % mod
                        dp0[i] = dp0[i] * (dp0[j] + dp1[j]) % mod
        ac.st(dp1[0])
        return

    @staticmethod
    def cf_1551f(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1551/F
        tag: tree_dp|bag_dp|brute_force
        """
        mod = 10 ** 9 + 7
        for _ in range(ac.read_int()):
            ac.read_str()
            n, k = ac.read_list_ints()
            dct = [[] for _ in range(n)]
            for _ in range(n - 1):
                i, j = ac.read_list_ints_minus_one()
                dct[i].append(j)
                dct[j].append(i)
            if k == 2:
                ac.st(n * (n - 1) // 2)
                continue
            ans = 0
            for i in range(n):
                dis = [-1] * n
                parent = [-1] * n
                stack = [i]
                dis[i] = 0
                while stack:
                    nex = []
                    for x in stack:
                        for y in dct[x]:
                            if dis[y] == -1:
                                dis[y] = dis[x] + 1
                                nex.append(y)
                                if parent[x] != -1:
                                    parent[y] = parent[x]
                                else:
                                    parent[y] = y
                    stack = nex
                group = [[] for _ in range(n)]
                for x in range(n):
                    group[dis[x]].append(x)
                for d in range(1, n):
                    if group[d]:
                        cnt = Counter([parent[x] for x in group[d]])
                        dp = [0] * (k + 1)
                        dp[0] = 1
                        for w in cnt.values():
                            for x in range(k, 0, -1):
                                dp[x] += dp[x - 1] * w
                                dp[x] %= mod
                        ans += dp[-1]
                        ans %= mod
            ac.st(ans % mod)
        return

    @staticmethod
    def lc_3241(edges: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/time-taken-to-mark-all-nodes/
        tag: reroot_dp|classical
        """
        n = len(edges) + 1
        graph = WeightedTree(n)
        for i, j in edges:
            graph.add_undirected_edge(i, j, 1)

        weights = [2 if i % 2 == 0 else 1 for i in range(n)]
        ans = graph.reroot_dp_for_tree_dis_with_node_weights_maximum(weights)
        return ans

    @staticmethod
    def cf_1988d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1988/D
        tag: tree_dp|classical|observation|data_range
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            dct = [[] for _ in range(n)]
            for _ in range(n - 1):
                i, j = ac.read_list_ints_minus_one()
                dct[i].append(j)
                dct[j].append(i)
            dp = [[0] * 21 for _ in range(n)]

            stack = [0]
            while stack:
                val = stack.pop()
                if val >= 0:
                    x, fa = val // n, val % n
                    stack.append(~val)
                    for y in dct[x]:
                        if y != fa:
                            stack.append(y * n + x)
                else:
                    val = ~val
                    x, fa = val // n, val % n
                    for i in range(21):
                        dp[x][i] = (i + 1) * nums[x]
                    for y in dct[x]:
                        if y != fa:
                            aa = bb = math.inf
                            for j in range(21):
                                cur = dp[y][j]
                                if cur < aa:
                                    aa, bb = cur, aa
                                elif cur < bb:
                                    bb = cur

                            for i in range(21):
                                dp[x][i] += aa if aa != dp[y][i] else bb
            ans = min(dp[0])
            ac.st(ans)
        return

    @staticmethod
    def cf_1101d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1101/D
        tag: tree_dp|prime_factor|classical|observation
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        pf = PrimeFactor(2 * 10 ** 5 + 10)
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        dp = [dict() for _ in range(n)]
        ans = 0
        stack = [0]
        while stack:
            val = stack.pop()
            if val >= 0:
                x, fa = val // n, val % n
                stack.append(~val)
                for y in dct[x]:
                    if y != fa:
                        stack.append(y * n + x)
            else:
                val = ~val
                x, fa = val // n, val % n
                for p, _ in pf.prime_factor[nums[x]]:
                    aa = bb = 0
                    for y in dct[x]:
                        if y != fa:
                            if p in dp[y]:
                                if dp[y][p] >= aa:
                                    aa, bb = dp[y][p], aa
                                elif dp[y][p] > bb:
                                    bb = dp[y][p]
                    ans = max(ans, aa + bb + 1)
                    dp[x][p] = aa + 1
        ac.st(ans)
        return

    @staticmethod
    def cf_1997d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1997/D
        tag: tree_dp|greedy
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                ans = [0] * self.n
                stack = [0]
                res = nums[0]
                while stack:
                    i = stack.pop()
                    if i >= 0:
                        stack.append(~i)
                        ind = self.point_head[i]
                        while ind:
                            j = self.edge_to[ind]
                            stack.append(j)
                            ind = self.edge_next[ind]
                    else:
                        i = ~i
                        ind = self.point_head[i]
                        cur = math.inf
                        while ind:
                            j = self.edge_to[ind]
                            cur = min(cur, ans[j])
                            ind = self.edge_next[ind]
                        if i == 0:
                            res = max(res, nums[0] + cur)
                        if cur == math.inf:
                            ans[i] = nums[i]
                        elif nums[i] >= cur:
                            ans[i] = cur
                        else:
                            ans[i] = nums[i] + (cur - nums[i]) // 2
                return res

        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            tree = Graph(n)
            p = ac.read_list_ints_minus_one()
            for k in range(n - 1):
                tree.add_directed_edge(p[k], k + 1)
            final = tree.tree_dp()
            ac.st(final)
        return

    @staticmethod
    def cf_1083a(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1083/A
        tag: tree_dp|greedy|implemention|weighted_tree|classical
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                ans = [0] * self.n
                parent = [-1] * self.n
                stack = [0]
                res = max(nums)
                while stack:
                    i = stack.pop()
                    if i >= 0:
                        stack.append(~i)
                        ind = self.point_head[i]
                        while ind:
                            j = self.edge_to[ind]
                            if j != parent[i]:
                                parent[j] = i
                                stack.append(j)
                            ind = self.edge_next[ind]
                    else:
                        i = ~i
                        ind = self.point_head[i]
                        a = b = 0
                        while ind:
                            j = self.edge_to[ind]
                            if j != parent[i]:
                                cur = ans[j] - self.edge_weight[ind]
                                if cur > a:
                                    a, b = cur, a
                                elif cur > b:
                                    b = cur
                            ind = self.edge_next[ind]
                        res = max(res, a + b + nums[i])
                        ans[i] = a + nums[i]
                return res

        n = ac.read_int()
        graph = Graph(n)
        nums = ac.read_list_ints()
        for _ in range(n - 1):
            u, v, c = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(u, v, c + 1)
        final = graph.tree_dp()
        ac.st(final)
        return

    @staticmethod
    def cf_982c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/982/C
        tag: tree_dp|greedy|classical
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                ans = [1] * self.n
                parent = [-1] * self.n
                stack = [0]
                while stack:
                    i = stack.pop()
                    if i >= 0:
                        stack.append(~i)
                        ind = self.point_head[i]
                        while ind:
                            j = self.edge_to[ind]
                            if j != parent[i]:
                                parent[j] = i
                                stack.append(j)
                            ind = self.edge_next[ind]
                    else:
                        i = ~i
                        ind = self.point_head[i]
                        while ind:
                            j = self.edge_to[ind]
                            if j != parent[i]:
                                ans[i] += ans[j]
                            ind = self.edge_next[ind]
                res = sum(x % 2 == 0 for x in ans)
                return res - 1

        n = ac.read_int()
        graph = Graph(n)
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(u, v)
        if n % 2:
            ac.st(-1)
        else:
            ac.st(graph.tree_dp())
        return

    @staticmethod
    def cf_1856e1(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1856/E1
        tag: tree_dp|greedy|down_to_up|classical
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                ans = [1] * self.n
                parent = [-1] * self.n
                stack = [0]
                res = 0
                dp = [0] * (n + 1)
                while stack:
                    i = stack.pop()
                    if i >= 0:
                        stack.append(~i)
                        ind = self.point_head[i]
                        while ind:
                            j = self.edge_to[ind]
                            if j != parent[i]:
                                parent[j] = i
                                stack.append(j)
                            ind = self.edge_next[ind]
                    else:
                        i = ~i
                        ind = self.point_head[i]
                        lst = []
                        while ind:
                            j = self.edge_to[ind]
                            if j != parent[i]:
                                lst.append(ans[j])
                            ind = self.edge_next[ind]
                        s = sum(lst)
                        for x in range(s + 1):
                            dp[x] = 0
                        dp[0] = 1
                        for num in lst:
                            for x in range(s, num - 1, -1):
                                if dp[x - num]:
                                    dp[x] = 1
                        res += max(x * (s - x) for x in range(s + 1) if dp[x])
                        ans[i] += s
                return res

        n = ac.read_int()
        graph = Graph(n)
        p = ac.read_list_ints_minus_one()
        for u in range(1, n):
            graph.add_undirected_edge(p[u - 1], u)
        final = graph.tree_dp()
        ac.st(final)
        return

    @staticmethod
    def cf_1153d_1(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1153/D
        tag: tree_dp|greedy|brain_teaser
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                ans = [1] * self.n
                parent = [-1] * self.n
                stack = [0]
                leaf = 0
                while stack:
                    i = stack.pop()
                    if i >= 0:
                        stack.append(~i)
                        ind = self.point_head[i]
                        while ind:
                            j = self.edge_to[ind]
                            if j != parent[i]:
                                parent[j] = i
                                stack.append(j)
                            ind = self.edge_next[ind]
                    else:
                        i = ~i
                        ind = self.point_head[i]
                        lst = []
                        while ind:
                            j = self.edge_to[ind]
                            if j != parent[i]:
                                cur = ans[j]
                                lst.append(cur)
                            ind = self.edge_next[ind]
                        if lst:
                            if nums[i] == 0:
                                ans[i] = sum(lst)
                            else:
                                ans[i] = min(lst)
                        else:
                            leaf += 1
                return leaf - ans[0] + 1

        n = ac.read_int()
        graph = Graph(n)
        nums = ac.read_list_ints()
        p = ac.read_list_ints_minus_one()
        for u in range(1, n):
            graph.add_undirected_edge(p[u - 1], u)
        final = graph.tree_dp()
        ac.st(final)
        return

    @staticmethod
    def cf_1153d_2(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1153/D
        tag: tree_dp|greedy|brain_teaser
        """

        n = ac.read_int()
        dp = [0] * n
        arr = ac.read_list_ints()
        p = ac.read_list_ints_minus_one()
        ans = 0
        for u in range(n - 1, 0, -1):
            if dp[u] == 0:
                dp[u] = 1
                ans += 1
            f = p[u - 1]
            if arr[f] == 0:
                dp[f] += dp[u]
            else:
                dp[f] = dp[u] if dp[f] == 0 else min(dp[f], dp[u])
        ac.st(ans - dp[0] + 1)
        return

    @staticmethod
    def cf_274b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/274/B
        tag: tree_dp|greedy|brain_teaser|classical
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                f = [0] * self.n
                g = [0] * self.n
                parent = [-1] * self.n
                stack = [0]
                while stack:
                    i = stack.pop()
                    if i >= 0:
                        stack.append(~i)
                        ind = self.point_head[i]
                        while ind:
                            j = self.edge_to[ind]
                            if j != parent[i]:
                                parent[j] = i
                                stack.append(j)
                            ind = self.edge_next[ind]
                    else:
                        i = ~i
                        ind = self.point_head[i]
                        ff = gg = 0
                        while ind:
                            j = self.edge_to[ind]
                            if j != parent[i]:
                                ff = max(ff, f[j])
                                gg = max(gg, g[j])
                            ind = self.edge_next[ind]
                        k = nums[i] + ff - gg
                        f[i] = ff
                        g[i] = gg
                        if k > 0:
                            g[i] += k
                        else:
                            f[i] -= k
                return f[0] + g[0]

        n = ac.read_int()
        graph = Graph(n)
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(u, v)
        nums = ac.read_list_ints()
        final = graph.tree_dp()
        ac.st(final)
        return

    @staticmethod
    def cf_369c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/369/problem/C
        tag: tree_dp|greedy|classical
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                res = []
                ans = [0] * self.n
                parent = [-1] * self.n
                stack = [0]
                while stack:
                    i = stack.pop()
                    if i >= 0:
                        stack.append(~i)
                        ind = self.point_head[i]
                        while ind:
                            j = self.edge_to[ind]
                            if j != parent[i]:
                                parent[j] = i
                                stack.append(j)
                            ind = self.edge_next[ind]
                    else:
                        i = ~i
                        ind = self.point_head[i]
                        sub = 0
                        while ind:
                            j = self.edge_to[ind]
                            w = self.edge_weight[ind]
                            if j != parent[i]:
                                if w and not ans[j]:
                                    sub += 1
                                    res.append(j + 1)
                                ans[i] += ans[j]
                            ind = self.edge_next[ind]
                        ans[i] += sub
                return res

        n = ac.read_int()
        graph = Graph(n)
        for _ in range(n - 1):
            u, v, ww = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(u, v, ww)
        final = graph.tree_dp()
        ac.st(len(final))
        ac.lst(final)
        return

    @staticmethod
    def cf_2033g_1(ac=FastIO()):
        """
        url: https://codeforces.com/contest/2033/problem/G
        tag: reroot_dp|tree_multiplication|build_graph|brain_teaser|implemention|dfs|back_trace|PointSetPointAddRangeMerge
        """

        class Graph(WeightedTree):
            def reroot_dp(self):
                dp1 = [0] * self.n
                dp2 = [0] * self.n
                self.parent = [-1] * self.n
                stack = [self.root]
                depth = [0] * self.n
                while stack:
                    u = stack.pop()
                    if u >= 0:
                        stack.append(~u)
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                self.parent[v] = u
                                stack.append(v)
                                depth[v] = depth[u] + 1
                    else:
                        u = ~u
                        a = b = 0
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                if dp1[v] + 1 > a:
                                    a, b = dp1[v] + 1, a
                                elif dp1[v] + 1 > b:
                                    b = dp1[v] + 1
                        dp1[u] = a
                        dp2[u] = b
                        ind = self.point_head[u]
                        while ind:
                            v = self.edge_to[ind]
                            if v != self.parent[u]:
                                sub = dp1[u] if dp1[u] != dp1[v] + 1 else dp2[u]
                                self.edge_weight[ind] = depth[u] - sub
                            ind = self.edge_next[ind]

                self.multiplication_build_for_minimum_weights()
                for _ in range(q):
                    jj, k = ac.read_list_ints()
                    jj -= 1
                    if not k:
                        ans.append(dp1[jj])
                    else:
                        k = min(k, depth[jj])
                        lowest = self.multiplication_get_kth_ancestor_min_weights(jj, k)
                        ans.append(max(dp1[jj], depth[jj] - lowest))
                return

        for _ in range(ac.read_int()):
            n = ac.read_int()
            graph = Graph(n)
            for _ in range(n - 1):
                i, j = ac.read_list_ints_minus_one()
                graph.add_undirected_edge(i, j, 1)
            q = ac.read_int()
            ans = []
            graph.reroot_dp()
            ac.lst(ans)
        return

    @staticmethod
    def cf_2033g_2(ac=FastIO()):
        """
        url: https://codeforces.com/contest/2033/problem/G
        tag: reroot_dp|tree_multiplication|build_graph|brain_teaser|implemention|dfs|back_trace|PointSetPointAddRangeMerge
        """

        class Graph(WeightedTree):
            def reroot_dp(self):
                dp1 = [0] * self.n
                dp2 = [0] * self.n
                self.parent = [-1] * self.n
                stack = [self.root]
                depth = [0] * self.n
                while stack:
                    u = stack.pop()
                    if u >= 0:
                        stack.append(~u)
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                self.parent[v] = u
                                stack.append(v)
                                depth[v] = depth[u] + 1
                    else:
                        u = ~u
                        a = b = 0
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                if dp1[v] + 1 > a:
                                    a, b = dp1[v] + 1, a
                                elif dp1[v] + 1 > b:
                                    b = dp1[v] + 1
                        dp1[u] = a
                        dp2[u] = b

                stack = [(self.root, 0)]
                mono = []
                tree = PointSetPointAddRangeMerge(self.n, 32 * self.n, min)
                while stack:
                    u, pre = stack.pop()
                    if u >= 0:
                        stack.append((~u, -1))
                        if u:
                            mono.append(pre)
                            tree.point_set(len(mono) - 1, pre)
                        for val in queries[u]:
                            kk, ii = val // q, val % q
                            if kk == 0 or not u:
                                ans[ii] = dp1[u]
                            else:
                                ans[ii] = max(dp1[u],
                                              depth[u] - tree.range_merge(max(0, len(mono) - kk), len(mono) - 1))
                        for v in self.get_to_nodes(u):
                            if v != self.parent[u]:
                                sub = dp1[u] if dp1[u] != dp1[v] + 1 else dp2[u]
                                stack.append((v, depth[u] - sub))
                    else:
                        u = ~u
                        if u:
                            mono.pop()
                return

        for _ in range(ac.read_int()):
            n = ac.read_int()
            graph = Graph(n)
            for _ in range(n - 1):
                i, j = ac.read_list_ints_minus_one()
                graph.add_undirected_edge(i, j, 1)
            q = ac.read_int()
            queries = [list() for _ in range(n)]
            for i in range(q):
                j, k = ac.read_list_ints()
                queries[j - 1].append(k * q + i)
            ans = [-1] * q
            graph.reroot_dp()
            ac.lst(ans)
        return

    @staticmethod
    def lc_100430_1(parent: List[int], s: str) -> List[int]:
        """
        url: https://leetcode.cn/problems/find-subtree-sizes-after-changes/
        tag: tree_dp|build_graph
        """

        class Graph(WeightedTree):
            def tree_dp1(self):
                res = 0
                stack = [0]
                ind = [[] for _ in range(26)]

                while stack:
                    i = stack.pop()
                    if i >= 0:
                        stack.append(~i)
                        if ind[ord(s[i]) - ord("a")]:
                            graph2.add_directed_edge(ind[ord(s[i]) - ord("a")][-1], i, 1)
                        elif i:
                            graph2.add_directed_edge(parent[i], i, 1)
                        ind[ord(s[i]) - ord("a")].append(i)
                        for j in self.get_to_nodes(i):
                            stack.append(j)
                    else:
                        i = ~i
                        ind[ord(s[i]) - ord("a")].pop()
                return res

            def tree_dp2(self):
                dp = [0] * self.n
                stack = [0]
                while stack:
                    i = stack.pop()
                    if i >= 0:
                        stack.append(~i)
                        for j in self.get_to_nodes(i):
                            stack.append(j)
                    else:
                        i = ~i
                        for j in self.get_to_nodes(i):
                            dp[i] += dp[j]
                return dp

        n = len(parent)
        graph = Graph(n)
        graph2 = Graph(n)
        for x in range(1, n):
            graph.add_directed_edge(parent[x], x)
        graph.tree_dp1()
        return graph2.tree_dp2()

    @staticmethod
    def lc_100430_2(parent: List[int], s: str) -> List[int]:
        """
        url: https://leetcode.cn/problems/find-subtree-sizes-after-changes/
        tag: tree_dp|build_graph
        """

        class Graph(WeightedTree):
            def tree_dp(self):
                stack = [0]
                ind = [[] for _ in range(26)]
                sub = [1] * n
                while stack:
                    i = stack.pop()
                    if i >= 0:
                        stack.append(~i)
                        ind[ord(s[i]) - ord("a")].append(i)
                        for j in self.get_to_nodes(i):
                            stack.append(j)
                    else:
                        i = ~i
                        ind[ord(s[i]) - ord("a")].pop()
                        lst = set()
                        for j in self.get_to_nodes(i):
                            if i == parent[j]:
                                lst.add(j)
                        for j in lst:
                            sub[i] += sub[j]
                        if ind[ord(s[i]) - ord("a")]:
                            self.add_directed_edge(ind[ord(s[i]) - ord("a")][-1], i, 1)
                            parent[i] = ind[ord(s[i]) - ord("a")][-1]
                return sub

        n = len(parent)
        graph = Graph(n)
        for x in range(1, n):
            graph.add_directed_edge(parent[x], x)
        return graph.tree_dp()
