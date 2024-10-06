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
P2014（https://www.luogu.com.cn/problem/P2014）tree_dp
P4316（https://www.luogu.com.cn/problem/P4316）reverse_graph|topological_sort|dag_dp
P1351（https://www.luogu.com.cn/problem/P1351）tree_dp
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
600E（https://codeforces.com/problemset/problem/600/E）dfs_order|heuristic_merge
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

=====================================AcWing=====================================
3760（https://www.acwing.com/problem/content/description/3763/）brain_teaser|tree_dp
4381（https://www.acwing.com/problem/content/description/4384/）reroot_dp|dfs|dfs_order|diff_array

"""
import math
from collections import deque, Counter
from functools import lru_cache
from typing import List, Optional

from src.basis.tree_node.template import TreeNode
from src.data_structure.list_node.template import ListNode
from src.data_structure.sorted_list.template import SortedList
from src.dp.tree_dp.template import ReRootDP, WeightedTree
from src.mathmatics.prime_factor.template import PrimeFactor
from src.search.dfs.template import UnWeightedTree
from src.utils.fast_io import FastIO
from src.utils.fast_io import inf


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1676g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1676/problem/G
        tag: tree_dp
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            parent = ac.read_list_ints_minus_one()
            color = ac.read_str()
            dct = [[] for _ in range(n)]
            for i in range(n - 1):
                dct[parent[i]].append(i + 1)
            ans = 0
            sub = [0] * n
            stack = [0]
            while stack:
                i = stack.pop()
                if i >= 0:
                    stack.append(~i)
                    stack.extend(dct[i])
                else:
                    i = ~i
                    x = sum(sub[j] for j in dct[i])
                    x += 1 if color[i] == "B" else -1
                    sub[i] = x
                    ans += x == 0
            ac.st(ans)
        return

    @staticmethod
    def lc_2003(parents: List[int], nums: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/smallest-missing-genetic-value-in-each-subtree/
        tag: tree_dp|heuristic_merge|classical
        """
        n = len(nums)
        dct = [[] for _ in range(n)]
        for i in range(1, n):
            dct[parents[i]].append(i)

        sub = [set() for _ in range(n)]
        stack = [0]
        ans = [-1] * n
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                stack.extend(dct[i])
            else:
                i = ~i
                x = 1
                pre = {nums[i]}
                for j in dct[i]:
                    cur = sub[j]
                    if len(cur) > len(pre):
                        pre, cur = cur, pre
                    pre.update(cur)
                    sub[j] = set()
                    if ans[j] > x:
                        x = ans[j]
                while x in pre:
                    x += 1
                ans[i] = x
                sub[i] = pre
        return ans

    @staticmethod
    def cf_1388c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1388/C
        tag: tree_dp|implemention|recursion|down_to_up|up_to_down
        """
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            person = ac.read_list_ints()
            h = ac.read_list_ints()
            edge = [[] for _ in range(n)]
            for _ in range(n - 1):
                x, y = ac.read_list_ints_minus_one()
                edge[x].append(y)
                edge[y].append(x)

            ans = 1
            pos = [0] * n
            neg = [0] * n
            stack = [(0, -1)]
            while stack and ans:
                i, fa = stack.pop()
                if i >= 0:
                    stack.append((~i, fa))
                    for j in edge[i]:
                        if j != fa:
                            stack.append((j, i))
                else:
                    i = ~i
                    a = b = 0
                    for j in edge[i]:
                        if j != fa:
                            a += pos[j]
                            b += neg[j]
                    if (h[i] + person[i] + b + a) % 2:
                        ans = 0
                        break
                    good = (h[i] + person[i] + b + a) // 2
                    bad = person[i] + a + b - good
                    if good < 0 or bad < 0 or bad > person[i] + b:
                        ans = 0
                        break
                    pos[i] = good
                    neg[i] = bad

            ac.st("YES" if ans else "NO")
        return

    @staticmethod
    def cf_161d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/161/D
        tag: tree_dp|counter
        """
        n, k = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            edge[x].append(y)
            edge[y].append(x)

        def idx(ii, jj):
            return ii * (k + 1) + jj

        dp = [0] * (k + 1) * n
        stack = [(0, -1)]
        ans = 0
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append((~i, fa))
                for j in edge[i]:
                    if j != fa:
                        stack.append((j, i))
            else:
                i = ~i
                for j in edge[i]:
                    if j != fa:
                        for s in range(1, k + 1):
                            dp[idx(i, s)] += dp[idx(j, s - 1)]
                dp[idx(i, 0)] = 1
                ans += dp[idx(i, k)]
                cur = 0
                for j in edge[i]:
                    if j != fa:
                        for s in range(1, k):
                            cur += dp[idx(j, s - 1)] * (dp[idx(i, k - s)] - dp[idx(j, k - s - 1)])
                ans += cur // 2
        ac.st(ans)
        return

    @staticmethod
    def cf_1324f(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1324/F
        tag: reroot_dp|dfs|down_to_up|up_to_down
        """
        n = ac.read_int()
        a = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            edge[x].append(y)
            edge[y].append(x)
        sub = [0] * n
        stack = [(0, -1)]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append((~i, fa))
                for j in edge[i]:
                    if j != fa:
                        stack.append((j, i))
            else:
                i = ~i
                cur = 2 * a[i] - 1
                for j in edge[i]:
                    if j != fa:
                        cur += ac.max(sub[j], 0)
                sub[i] = cur

        ans = [0] * n
        stack = [(0, -1, 0)]
        while stack:
            i, fa, d = stack.pop()
            ans[i] = d + sub[i]
            for j in edge[i]:
                if j != fa:
                    if sub[j] > 0:
                        nex = sub[i] - sub[j] + d
                    else:
                        nex = sub[i] + d
                    nex = ac.max(nex, 2 * a[i] - 1)
                    stack.append((j, i, ac.max(0, nex)))
        ac.lst(ans)
        return

    @staticmethod
    def cf_337d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/337/D
        tag: reroot_dp|dfs|down_to_up|up_to_down
        """
        n, m, d = ac.read_list_ints()
        sub = [-inf] * n
        evil = [0] * n
        for i in ac.read_list_ints_minus_one():
            sub[i] = 0
            evil[i] = 1
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            edge[u].append(v)
            edge[v].append(u)

        stack = [(0, -1)]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append((~i, fa))
                for j in edge[i]:
                    if j != fa:
                        stack.append((j, i))
            else:
                i = ~i
                cur = -inf
                for j in edge[i]:
                    if j != fa:
                        cur = ac.max(cur, sub[j] + 1)
                sub[i] = ac.max(sub[i], cur)

        stack = [(0, -1, -inf)]
        while stack:
            i, fa, up = stack.pop()
            sub[i] = ac.max(sub[i], up)
            if evil[i]:
                up = ac.max(0, up)
            a = b = -inf
            for j in edge[i]:
                if j != fa:
                    if sub[j] > a:
                        a, b = sub[j], a
                    elif sub[j] > b:
                        b = sub[j]
            for j in edge[i]:
                if j != fa:
                    if sub[j] == a:
                        stack.append((j, i, ac.max(b + 2, up + 1)))
                    else:
                        stack.append((j, i, ac.max(a + 2, up + 1)))
        ac.st(sum(x <= d for x in sub))
        return

    @staticmethod
    def cf_1092f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1092/problem/F
        tag: tree_dis|reroot_dp
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            dct[x].append(y)
            dct[y].append(x)
        ans = ReRootDP().get_tree_distance_weight(dct, nums)
        ac.st(max(ans))
        return

    @staticmethod
    def lc_968(root: Optional[TreeNode]) -> int:
        """
        url: https://leetcode.cn/problems/binary-tree-cameras/
        tag: tree_dp
        """

        # tree_dp
        def dfs(node):
            # 不装被监控，装被监控，不装不被监控
            if not node:
                return [0, inf, 0]
            left = dfs(node.left)
            right = dfs(node.right)
            res = [inf, inf, inf]
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

        # classical二叉树与linked_list|比较的memory_searchDP

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
        # reroot_dp题最佳结果
        n = ac.read_int()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            edge[u].append(v)
            edge[v].append(u)

        @ac.bootstrap
        def dfs(i, fa):
            for j in edge[i]:
                if j != fa:
                    yield dfs(j, i)
                    down[i] += down[j]
                    son[i] += son[j]
            son[i] += 1
            down[i] += son[i]
            yield

        down = [0] * n
        son = [0] * n
        dfs(0, -1)

        @ac.bootstrap
        def dfs2(i, fa, pre):
            up[i] = pre
            res = sum(down[j] for j in edge[i] if j != fa)
            for j in edge[i]:
                if j != fa:
                    yield dfs2(j, i, (n - son[j]) + pre + (res - down[j]))
            yield

        up = [0] * n
        dfs2(0, -1, 0)
        ac.st(max(up[i] + (down[i] - son[i]) + n for i in range(n)))
        return

    @staticmethod
    def cf_600e_bfs(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/600/E
        tag: dfs_order|heuristic_merge
        """
        # 自下而上recursion的迭代写法，从小到大heuristic_merge
        n = ac.read_int()
        colors = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            edge[i].append(j)
            edge[j].append(i)
        # dfs_order自下而上以及父子信息获取
        stack = [[0, -1]]
        parent = [-1] * n
        down_to_up_order = []
        while stack:
            i, fa = stack.pop()
            down_to_up_order.append(i)
            for j in edge[i]:
                if j != fa:
                    stack.append([j, i])
                    parent[j] = i
        down_to_up_order.reverse()

        # 维护一个最大值的出现次数
        mx = [0] * n
        ans = [0] * n
        dp = [None] * n
        for i in down_to_up_order:
            dp[i] = Counter()
            dp[i][colors[i]] += 1
            mx[i] = 1
            ans[i] = colors[i]
            for j in edge[i]:
                if dp[j]:
                    if len(dp[j]) > len(dp[i]):
                        # 从小到大
                        dp[i], dp[j] = dp[j], dp[i]
                        mx[i] = mx[j]
                        ans[i] = ans[j]
                    for w in dp[j]:
                        # heuristic_merge
                        dp[i][w] += dp[j][w]
                        if dp[i][w] == mx[i]:
                            ans[i] += w
                        elif dp[i][w] > mx[i]:
                            mx[i] = dp[i][w]
                            ans[i] = w
                    # 及时清空
                    dp[j] = None
        ac.lst(ans)
        return

    @staticmethod
    def cf_600e_dfs(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/600/E
        tag: dfs_order|heuristic_merge
        """
        # 自下而上recursion的recursion写法，从小到大heuristic_merge
        n = ac.read_int()
        nums = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            edge[u].append(v)
            edge[v].append(u)

        @ac.bootstrap
        def dfs(i, fa):
            dp[i] = Counter()
            dp[i][nums[i]] += 1
            ceil[i] = 1
            cur = nums[i]
            for j in edge[i]:
                if j != fa:
                    yield dfs(j, i)
                    if len(dp[j]) > len(dp[i]):
                        dp[i], dp[j] = dp[j], dp[i]
                        cur = ans[j]
                        ceil[i] = ceil[j]
                    for num in dp[j]:
                        dp[i][num] += dp[j][num]
                        if dp[i][num] > ceil[i]:
                            ceil[i] = dp[i][num]
                            cur = num
                        elif dp[i][num] == ceil[i]:
                            cur += num
                    dp[j] = None
            ans[i] = cur
            yield

        dp = [None] * n
        ans = [0] * n
        ceil = [0] * n
        dfs(0, -1)
        ac.lst(ans)
        return

    @staticmethod
    def lg_p1395_1(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1395
        tag: tree_dis|tree_centroid|reroot_dp|classical|up_to_down|down_to_up
        """
        # tree_centroid为最大子树节点数最小
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints()
            dct[i - 1].append(j - 1)
            dct[j - 1].append(i - 1)

        root = ReRootDP().get_tree_centroid(dct)

        def bfs_diameter(src):
            ans = 0
            stack = [[src, 0]]
            parent = [-1] * n
            while stack:
                u, dis = stack.pop()
                ans += dis
                for v in dct[u]:
                    if v != parent[u]:
                        parent[v] = u
                        stack.append([v, dis + 1])
            return ans

        ac.lst([root + 1, bfs_diameter(root)])
        return

    @staticmethod
    def lg_p1395_2(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1395
        tag: tree_dis|tree_centroid|reroot_dp|classical|up_to_down|down_to_up
        """
        # tree_centroid为距离其余所有节点
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints()
            dct[i - 1].append(j - 1)
            dct[j - 1].append(i - 1)

        ans = ReRootDP().get_tree_distance(dct)
        dis = min(ans)
        ac.lst([ans.index(dis) + 1, dis])
        return

    @staticmethod
    def cf_1822f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1822/problem/F
        tag: tree_dis|reroot_dp|down_to_up|up_to_down
        """
        # 换根 DP 树中节点其余节点最大的距离
        for _ in range(ac.read_int()):
            n, k, c = ac.read_list_ints()
            dct = [[] for _ in range(n)]
            for _ in range(n - 1):
                i, j = ac.read_list_ints_minus_one()
                dct[i].append(j)
                dct[j].append(i)

            dis = ReRootDP().get_tree_distance_max(dct)

            ans = -inf
            stack = [[0, 0, -1]]
            while stack:
                i, d, fa = stack.pop()
                cur = dis[i] * k - d
                ans = ac.max(ans, cur)
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, d + c, i])
            ac.st(ans)
        return

    @staticmethod
    def lg_p1352(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1352
        tag: tree_dp|mis|maximum_independent_set
        """

        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]

        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            dct[y].append(x)
            degree[x] += 1

        root = [i for i in range(n) if not degree[i]][0]
        dp = [[0, 0] for _ in range(n)]
        stack = [[root, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                x = nums[i] if nums[i] > 0 else 0
                y = 0
                for j in dct[i]:
                    if j != fa:
                        a, b = dp[j]
                        x += a
                        y += b
                dp[i] = [y, ac.max(x, y)]
        ac.st(max(dp[root]))
        return

    @staticmethod
    def lg_p2015(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2015
        tag: tree_dp|tree_bag_dp
        """
        # tree_dp
        n, q = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(n - 1):
            x, y, z = ac.read_list_ints()
            dct[x - 1][y - 1] = z
            dct[y - 1][x - 1] = z
        dp = [[0] * (q + 1) for _ in range(n)]
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                if len(dct[i]) > 1:
                    a, b = [x for x in dct[i] if x != fa]
                    for j in range(1, q + 1):
                        cur = ac.max(dp[a][j - 1] + dct[i][a], dp[b][j - 1] + dct[i][b])
                        for k in range(j - 1):
                            cur = ac.max(cur, dp[a][k] + dp[b][j - k - 2] + dct[i][a] + dct[i][b])
                        dp[i][j] = cur
        ac.st(dp[0][q])
        return

    @staticmethod
    def lg_p2014(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2014
        tag: tree_dp
        """
        # tree_dp|bag_dp|
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n + 1)]
        nums = [0]
        for i in range(n):
            k, s = ac.read_list_ints()
            nums.append(s)
            dct[k].append(i + 1)
        dp = [[0] * (m + 2) for _ in range(n + 1)]
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                dp[i][1] = nums[i]
                for j in dct[i]:
                    if j != fa:
                        cur = dp[i][:]
                        for x in range(1, m + 2):
                            for y in range(m + 2 - x):
                                cur[x + y] = ac.max(cur[x + y], dp[i][x] + dp[j][y])
                        dp[i] = cur[:]
        ac.st(dp[0][m + 1])
        return

    @staticmethod
    def lg_p4316(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4316
        tag: reverse_graph|topological_sort|dag_dp
        """
        # reverse_graph|topological_sorting树形prob_dp
        n, m = ac.read_list_ints()
        dp = [0 for _ in range(n)]
        degree = [0] * n
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            a, b, w = ac.read_list_ints()
            a -= 1
            b -= 1
            dct[b][a] = w
            degree[a] += 1
        cnt = degree[:]

        stack = deque([n - 1])
        while stack:
            i = stack.popleft()
            len(dct[i])
            a = dp[i]
            for j in dct[i]:
                dp[j] += a + dct[i][j]
                degree[j] -= 1
                if not degree[j]:
                    dp[j] /= cnt[j]
                    stack.append(j)
        ans = "%.2f" % (dp[0])
        ac.st(ans)
        return

    @staticmethod
    def lg_p1351(ac=FastIO()):
        # tree_dp
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        nums = ac.read_list_ints()
        ceil = ans = 0
        mod = 10007
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            lst = []
            if fa != -1:
                lst.append(nums[fa])
            for j in dct[i]:
                if j != fa:
                    stack.append([j, i])
                    lst.append(nums[j])
            if lst:

                s = sum(lst)
                a = b = 0
                for num in lst:
                    ans += num * (s - num)
                    ans %= mod
                    if num > a:
                        a, b = num, a
                    elif num > b:
                        b = num
                ceil = ac.max(ceil, a * b)

        ac.lst([ceil, ans])
        return

    @staticmethod
    def lg_3408(ac=FastIO()):

        # tree_dp| implemention
        n, t, c = ac.read_list_ints()
        dct = [[] for _ in range(n + 1)]
        nums = [c]
        for i in range(n):
            b, a = ac.read_list_ints()
            dct[b].append(i + 1)
            nums.append(a)

        stack = [0]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in dct[i]:
                    stack.append(j)
            else:
                i = ~i
                if not dct[i]:
                    continue
                else:
                    # 收到子树下属的最少花费
                    if nums[i] > t:
                        continue

                    # 需要最少的 x 个下属花费
                    x = math.ceil(len(dct[i]) * nums[i] / t)
                    lst = []
                    for j in dct[i]:
                        lst.append(nums[j])
                    lst.sort()
                    nums[i] = sum(lst[:x])
        ac.st(nums[0])
        return

    @staticmethod
    def lg_p3478(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3478
        tag: tree_centroid
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        dis = ReRootDP().get_tree_distance(dct)
        ind = 0
        for i in range(1, n):
            if dis[i] > dis[ind]:
                ind = i
        ac.st(ind + 1)
        return

    @staticmethod
    def lg_p3931(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3931
        tag: classical|tree_dp
        """
        # tree_dp| implemention
        n, root = ac.read_list_ints()
        root -= 1
        dct = [dict() for _ in range(n)]
        for _ in range(n - 1):
            i, j, c = ac.read_list_ints_minus_one()
            c += 1
            dct[i][j] = dct[j][i] = c
        stack = [[root, -1]]
        sub = [inf] * n
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                if len(dct[i]) == 1 and i != root:
                    continue
                res = 0
                for j in dct[i]:
                    if j != fa:
                        res += ac.min(dct[i][j], sub[j])
                sub[i] = res
        ac.st(sub[root] if sub[root] < inf else 0)
        return

    @staticmethod
    def lg_p4395(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4395
        tag: tree_dp|greedy
        """
        # tree_dp| greedy标权值使得整棵树总价值最小
        n = ac.read_int()
        ceil = int(math.log2(n)) + 1
        sub = [[inf] * (ceil + 1) for _ in range(n)]
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)

        # 迭代写法
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                cur = [0] * (ceil + 1)
                for j in dct[i]:
                    if j != fa:
                        # 记录子树最小的两个值
                        a = b = inf
                        for c in sub[j][1:]:
                            if c < a:
                                a, b = c, a
                            elif c < b:
                                b = c
                        # 记录 i 赋值为 x 时的子树价值和
                        for x in range(1, ceil + 1):
                            if sub[j][x] == a:
                                cur[x] += b
                            else:
                                cur[x] += a
                for x in range(1, ceil + 1):
                    sub[i][x] = x + cur[x]
        ac.st(min(sub[0][1:]))
        return

    @staticmethod
    def lg_p8625(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8625
        tag: tree_dp|classical
        """
        # tree_dp| classical
        n = ac.read_int()
        nums = ac.read_list_ints()
        sub = [0] * n
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                res = 0
                for j in dct[i]:
                    if j != fa:
                        res += sub[j]
                res += nums[i]
                sub[i] = res if res > 0 else 0
        ac.st(max(sub))
        return

    @staticmethod
    def lc_1617_2(n: int, edges: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/count-subtrees-with-max-distance-between-cities/
        tag: brute_force|tree_diameter|tree_dp
        """
        # brute_forcetree_diameter端点与乘法原理tree_dp
        dct = [[] for _ in range(n)]
        for i, j in edges:
            i -= 1
            j -= 1
            dct[i].append(j)
            dct[j].append(i)

        dis = []
        for i in range(n):
            cur = [inf] * n
            cur[i] = 0
            stack = deque([i])
            while stack:
                i = stack.pop()
                for j in dct[i]:
                    if cur[j] == inf:
                        cur[j] = cur[i] + 1
                        stack.append(j)
            dis.append(cur[:])

        ans = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = dis[i][j]

                stack = [[i, -1]]
                sub = [0] * n
                while stack:
                    ii, fa = stack.pop()
                    if ii >= 0:
                        stack.append([~ii, fa])
                        for jj in dct[ii]:
                            if jj != fa:
                                if (dis[i][jj] < d or (dis[i][jj] == d and jj > j)) and \
                                        (dis[j][jj] < d or (dis[j][jj] == d and jj > i)):
                                    stack.append([jj, ii])
                    else:
                        ii = ~ii
                        sub[ii] = 1  # x是必点
                        for jj in dct[ii]:
                            if jj != fa:
                                if (dis[i][jj] < d or (dis[i][jj] == d and jj > j)) and \
                                        (dis[j][jj] < d or (dis[j][jj] == d and jj > i)):
                                    sub[ii] *= sub[jj]
                        if dis[i][ii] + dis[j][ii] > d:  # x 是可选点
                            sub[ii] += 1
                ans[d] += sub[i]

        return ans[1:]

    @staticmethod
    def ac_3760(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/3763/
        tag: brain_teaser|tree_dp
        """
        # brain_teaser转化为tree_dp迭代方式求解
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
        # 迭代法实现树形reroot_dp
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            dct[x].append([y, 0])
            dct[y].append([x, 1])

        # 第一遍DP子节点的影响，从上到下再从下到上累|
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

        # 第二遍DP祖先节点与兄弟节点的影响
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
        # 一遍DFS迭代实现树形reroot_dp
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
        # tree_dpgreedy
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

        def standard_procedure(n: int) -> int:
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

            ans = inf
            stack = [(0, -1, 0, 0)]
            while stack:
                x, fa, pre, pre_dia = stack.pop()
                a, b, c = sub[x]
                aa = bb = -inf
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
        ac.st(standard_procedure(n))
        return

    @staticmethod
    def abc_348e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc348/tasks/abc348_e
        tag: reroot_dp|classical
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            a, b = ac.read_list_ints_minus_one()
            dct[a].append(b)
            dct[b].append(a)
        weight = ac.read_list_ints()
        ans = ReRootDP().get_tree_distance_weight(dct, weight)
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
            aa = bb = -inf
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
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        ans = ReRootDP().get_tree_distance_weight(dct, [1] * n)
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
                    dp[i] = (y, ac.max(x, y))
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
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        weights = [2 if i % 2 == 0 else 1 for i in range(n)]
        ans = ReRootDP().get_tree_distance_max_weighted(dct, weights)
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
                            aa = bb = inf
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

        class Graph(UnWeightedTree):
            def tree_dp(self, nums):
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
                        cur = inf
                        while ind:
                            j = self.edge_to[ind]
                            cur = min(cur, ans[j])
                            ind = self.edge_next[ind]
                        if i == 0:
                            res = max(res, nums[0] + cur)
                        if cur == inf:
                            ans[i] = nums[i]
                        elif nums[i] >= cur:
                            ans[i] = cur
                        else:
                            ans[i] = nums[i] + (cur - nums[i]) // 2
                return res

        for _ in range(ac.read_int()):
            n = ac.read_int()
            arr = ac.read_list_ints()
            tree = Graph(n)
            p = ac.read_list_ints_minus_one()
            for k in range(n - 1):
                tree.add_directed_edge(p[k], k + 1)
            final = tree.tree_dp(arr)
            ac.st(final)
        return

    @staticmethod
    def cf_1083a(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1083/A
        tag: tree_dp|greedy|implemention|weighted_tree|classical
        """

        class Graph(WeightedTree):
            def tree_dp(self, nums):
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
        weights = ac.read_list_ints()
        for _ in range(n - 1):
            u, v, c = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(u, v, c + 1)
        final = graph.tree_dp(weights)
        ac.st(final)
        return

    @staticmethod
    def cf_982c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/982/C
        tag: tree_dp|greedy|classical
        """

        class Graph(UnWeightedTree):
            def tree_dp(self, nums):
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
            ac.st(graph.tree_dp([-1]))
        return
