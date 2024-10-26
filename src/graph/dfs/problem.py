"""

Algorithm：dfs|coloring_method|brute_force|back_trace|euler_order|dfs_order|prune|iteration
Description：back_trace|brute_force|dfs_order|up_to_down|down_to_up|heuristic_method|dsu_on_tree


====================================LeetCode====================================
473（https://leetcode.cn/problems/matchsticks-to-square/）dfs|back_trace
301（https://leetcode.cn/problems/remove-invalid-parentheses/）back_trace|dfs|prune
2581（https://leetcode.cn/problems/count-number-of-possible-root-nodes）dfs_order|diff_array|counter|reroot_dp
1059（https://leetcode.cn/problems/all-paths-from-source-lead-to-destination/）memory_search|dfs|back_trace
1718（https://leetcode.cn/problems/construct-the-lexicographically-largest-valid-sequence/）back_trace
2322（https://leetcode.cn/problems/minimum-score-after-removals-on-a-tree/）dfs_order|brute_force
1240（https://leetcode.cn/problems/tiling-a-rectangle-with-the-fewest-squares/）dfs|back_trace|prune
1239（https://leetcode.cn/problems/maximum-length-of-a-concatenated-string-with-unique-characters/）dfs|back_trace|2-base|brute_force
1080（https://leetcode.cn/problems/insufficient-nodes-in-root-to-leaf-paths/description/）dfs|up_to_down|down_to_up
2056（https://leetcode.cn/problems/number-of-valid-move-combinations-on-chessboard/description/）back_trace|brute_force
2458（https://leetcode.cn/problems/height-of-binary-tree-after-subtree-removal-queries）dfs_order|classical
2858（https://leetcode.cn/problems/minimum-edge-reversals-so-every-node-is-reachable/）reroot_dp|dfs|dfs_order|diff_array
3327（https://leetcode.cn/problems/check-if-dfs-strings-are-palindromes/）dfs_order|manacher|palindrome|classical

=====================================LuoGu======================================
P2383（https://www.luogu.com.cn/problem/P2383）dfs|back_trace
P1120（https://www.luogu.com.cn/problem/P1120）dfs|back_trace
P1692（https://www.luogu.com.cn/problem/P1692）dfs|brute_force|lexicographical_order
P1612（https://www.luogu.com.cn/problem/P1612）dfs|prefix_sum|binary_search
P1475（https://www.luogu.com.cn/problem/P1475）dfs
P2080（https://www.luogu.com.cn/problem/P2080）dfs|back_trace|prune
P2090（https://www.luogu.com.cn/problem/P2090）dfs|greed|back_trace|prune|euclidean_division|euclidean_minus
P2420（https://www.luogu.com.cn/problem/P2420）brain_teaser|dfs|shortest_path|xor_path|classical
P1473（https://www.luogu.com.cn/problem/P1473）dfs|brute_force
P1461（https://www.luogu.com.cn/problem/P1461）dfs|back_trace|brute_force
P1394（https://www.luogu.com.cn/problem/P1394）dfs
P1180（https://www.luogu.com.cn/problem/P1180）dfs|implemention
P1118（https://www.luogu.com.cn/problem/P1118）implemention|lexicographical_order|dfs
P3252（https://www.luogu.com.cn/problem/P3252）dfs|back_trace|prefix_sum|hash
P4913（https://www.luogu.com.cn/problem/P4913）dfs
P5118（https://www.luogu.com.cn/problem/P5118）dfs|back_trace|hash|implemention
P5197（https://www.luogu.com.cn/problem/P5197）tree_dp|implemention|coloring_method
P5198（https://www.luogu.com.cn/problem/P5198）union_find
P5318（https://www.luogu.com.cn/problem/P5318）bfs|topological_sort|dfs_order
P6691（https://www.luogu.com.cn/problem/P6691）coloring_method|bipartite_graph|specific_plan|counter
P7370（https://www.luogu.com.cn/problem/P7370）ancestor
P1036（https://www.luogu.com.cn/problem/P1036）back_trace|prune
P8578（https://www.luogu.com.cn/problem/P8578）greed|dfs_order
P8838（https://www.luogu.com.cn/problem/P8838）dfs|back_trace
P1444（https://www.luogu.com.cn/problem/P1444）dfs|back_trace|circle_check|brain_teaser|observation

===================================CodeForces===================================
570D（https://codeforces.com/contest/570/problem/D）dfs_order|binary_search|offline_query
208E（https://codeforces.com/contest/208/problem/E）dfs_order|lca|binary_search|counter
1006E（https://codeforces.com/contest/1006/problem/E）dfs_order|template
1702G2（https://codeforces.com/contest/1702/problem/G2）dfs_order|lca
1899G（https://codeforces.com/contest/1899/problem/G）dfs|inclusion_exclusion|classical|point_add_range_sum|heuristic_merge
1714G（https://codeforces.com/contest/1714/problem/G）dfs|binary_search|prefix_sum
1675F（https://codeforces.com/contest/1675/problem/F）dfs_order|greed
219D（https://codeforces.com/contest/219/problem/D）reroot_dp|dfs|dfs_order|diff_array
246E（https://codeforces.com/problemset/problem/246/E）tree_array|offline_query|range_unique|dfs_order
1076E（https://codeforces.com/problemset/problem/1076/E）tree_diff_array|dfs|classical
383C（https://codeforces.com/problemset/problem/383/C）dfs_order|odd_even|range_add|point_get
3C（https://codeforces.com/problemset/problem/3/C）dfs|back_trace|brute_force|implemention
459C（https://codeforces.com/problemset/problem/459/C）back_trace|brute_force|classical|implemention
1918F（https://codeforces.com/problemset/problem/1918/F）dfs_order|greed|tree_lca|implemention|observation|brain_teaser
1882D（https://codeforces.com/problemset/problem/1882/D）dfs_order|diff_array|contribution_method|greed
1009F（https://codeforces.com/problemset/problem/1009/F）heuristic_merge|classical
27E（https://codeforces.com/problemset/problem/27/E）prime_factor|brute_force|factor_dp

====================================AtCoder=====================================
ABC133F（https://atcoder.jp/contests/abc133/tasks/abc133_f）euler_order|online_tree_dis|binary_search|prefix_sum
ABC337G（https://atcoder.jp/contests/abc337/tasks/abc337_g）dfs_order|contribution_method|classical|tree_array
ABC328E（https://atcoder.jp/contests/abc328/tasks/abc328_e）dfs|back_trace|union_find|brute_force
ABC326D（https://atcoder.jp/contests/abc326/tasks/abc326_d）dfs|back_trace|brute_force
ABC322D（https://atcoder.jp/contests/abc322/tasks/abc322_d）dfs|back_trace|brute_force
ABC284E（https://atcoder.jp/contests/abc284/tasks/abc284_e）dfs|back_trace|classical
ABC268D（https://atcoder.jp/contests/abc268/tasks/abc268_d）dfs|back_trace|prune|classical
ABC244G（https://atcoder.jp/contests/abc244/tasks/abc244_g）construction|euler_order|brain_teaser|classical
ABC240F（https://atcoder.jp/contests/abc240/tasks/abc240_e）dfs_order|leaf|classical
ABC236D（https://atcoder.jp/contests/abc236/tasks/abc236_d）back_trace|prune|brute_force|classical
ABC213D（https://atcoder.jp/contests/abc213/tasks/abc213_d）euler_order|classical
ABC196D（https://atcoder.jp/contests/abc196/tasks/abc196_d）dfs|state_compression|bit_operation

=====================================AcWing=====================================
4313（https://www.acwing.com/problem/content/4313/）dfs_order|template
21（https://www.acwing.com/problem/content/description/21/）back_trace|template

"""

import bisect
import math
from bisect import bisect_right, bisect_left
from collections import defaultdict
from itertools import accumulate, permutations
from typing import List, Optional

from src.basis.diff_array.template import PreFixSumMatrix
from src.basis.tree_node.template import TreeNode
from src.graph.dfs.template import DFS, DfsEulerOrder
from src.graph.union_find.template import UnionFind
from src.string.manacher_palindrome.template import ManacherPlindrome
from src.structure.segment_tree.template import RangeAddPointGet
from src.structure.tree_array.template import PointAddRangeSum
from src.tree.tree_dp.template import WeightedTree
from src.tree.tree_lca.template import OfflineLCA
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_473_1(matchsticks: List[int]) -> bool:
        """
        url: https://leetcode.cn/problems/matchsticks-to-square/
        tag: dfs|back_trace|state_dp|classical
        """

        def dfs(i):
            nonlocal ans
            if ans:
                return
            if i == n:
                if len(pre) == 4:
                    ans = True
                return
            if len(pre) > 4:
                return
            for j in range(len(pre)):
                if pre[j] + matchsticks[i] <= m:
                    pre[j] += matchsticks[i]
                    dfs(i + 1)
                    pre[j] -= matchsticks[i]
            pre.append(matchsticks[i])
            dfs(i + 1)
            pre.pop()
            return

        n, s = len(matchsticks), sum(matchsticks)
        if s % 4 or max(matchsticks) > s // 4:
            return False
        matchsticks.sort(reverse=True)  # important!!!
        m = s // 4
        ans = False
        pre = []
        dfs(0)
        return ans

    @staticmethod
    def lc_473_2(matchsticks: List[int]) -> bool:
        """
        url: https://leetcode.cn/problems/matchsticks-to-square/
        tag: dfs|back_trace|state_dp|classical
        """
        n = len(matchsticks)
        tot = sum(matchsticks)
        if tot % 4:
            return False
        single = tot // 4
        dp = [-1] * (1 << n)
        dp[0] = 0
        for s in range(1, 1 << n):
            for i, x in enumerate(matchsticks):
                if not s & (1 << i):
                    continue
                pre = s ^ (1 << i)
                if dp[pre] >= 0 and dp[pre] + x <= single:
                    dp[s] = (dp[pre] + x) % single
                    break
        return dp[-1] == 0

    @staticmethod
    def lg_2383(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2383
        tag: dfs|back_trace|state_dp|classical
        """
        for _ in range(ac.read_int()):
            matchsticks = ac.read_list_ints()[1:]
            n = len(matchsticks)
            tot = sum(matchsticks)
            if tot % 4:
                ac.no()
                continue
            single = tot // 4
            dp = [-1] * (1 << n)
            dp[0] = 0
            for s in range(1, 1 << n):
                for i, x in enumerate(matchsticks):
                    if not s & (1 << i):
                        continue
                    pre = s ^ (1 << i)
                    if dp[pre] >= 0 and dp[pre] + x <= single:
                        dp[s] = (dp[pre] + x) % single
                        break
            ac.st("yes" if dp[-1] == 0 else "no")
        return

    @staticmethod
    def lc_2858(n: int, edges: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/minimum-edge-reversals-so-every-node-is-reachable/）
        tag：reroot_dp|dfs|dfs_order|diff_array
        """
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)
        start, end = DFS().gen_bfs_order_iteration(dct)
        diff = [0] * n
        for i, j in edges:
            if start[i] < start[j]:
                a, b = start[j], end[j]
                diff[a] += 1
                if b + 1 < n:
                    diff[b + 1] -= 1
            else:
                a, b = start[i], end[i]
                if 0 <= a - 1:
                    diff[0] += 1
                    diff[a] -= 1
                if b + 1 <= n - 1:
                    diff[b + 1] += 1
        diff = list(accumulate(diff))
        return [diff[start[i]] for i in range(n)]

    @staticmethod
    def cf_1006e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1006/problem/E
        tag: dfs_order|template
        """

        n, q = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        p = ac.read_list_ints_minus_one()
        for i in range(n - 1):
            dct[p[i]].append(i + 1)
        for i in range(n):
            dct[i].reverse()
        dfs = DfsEulerOrder(dct)
        for _ in range(q):
            u, k = ac.read_list_ints()
            u -= 1
            x = dfs.start[u]
            if n - x < k or dfs.start[dfs.order_to_node[x + k - 1]] > dfs.end[u]:
                ac.st(-1)
            else:
                ac.st(dfs.order_to_node[x + k - 1] + 1)
        return

    @staticmethod
    def cf_1899g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1899/problem/G
        tag: dfs|inclusion_exclusion|classical|point_add_range_sum
        """
        for _ in range(ac.read_int()):
            n, q = ac.read_list_ints()
            dct = [[] for _ in range(n)]
            for _ in range(n - 1):
                i, j = ac.read_list_ints_minus_one()
                dct[i].append(j)
                dct[j].append(i)
            p = ac.read_list_ints_minus_one()

            ind = [-1] * n
            for i in range(n):
                ind[p[i]] = i

            qs = [[] for _ in range(n)]
            ans = [0] * q
            for i in range(q):
                ll, rr, xx = ac.read_list_ints_minus_one()
                qs[xx].append((ll, rr, i))

            tree = PointAddRangeSum(n)
            stack = [(0, -1)]
            while stack:
                i, fa = stack.pop()
                if i >= 0:
                    stack.append((~i, fa))
                    for ll, rr, xx in qs[i]:
                        ans[xx] -= tree.range_sum(ll + 1, rr + 1)
                    tree.point_add(ind[i] + 1, 1)
                    for j in dct[i]:
                        if j != fa:
                            stack.append((j, i))
                else:
                    i = ~i
                    for ll, rr, xx in qs[i]:
                        ans[xx] += tree.range_sum(ll + 1, rr + 1)
            for a in ans:
                ac.st("YES" if a > 0 else "NO")
        return

    @staticmethod
    def lc_301(s):
        """
        url: https://leetcode.cn/problems/remove-invalid-parentheses/
        tag: back_trace|dfs|prune
        """

        def dfs(i):
            nonlocal ans, pre, left, right
            if i == n:
                if left == right:
                    if len(pre) == len(ans[-1]):
                        ans.append("".join(pre))
                    elif len(pre) > len(ans[-1]):
                        ans = ["".join(pre)]
                return
            if right > left or right + n - i < left or len(pre) + n - i < len(ans[-1]):
                return

            dfs(i + 1)
            left += int(s[i] == "(")
            right += int(s[i] == ")")
            pre.append(s[i])
            dfs(i + 1)
            left -= int(s[i] == "(")
            right -= int(s[i] == ")")
            pre.pop()
            return

        n = len(s)
        ans = [""]
        left = right = 0
        pre = []
        dfs(0)
        return list(set(ans))

    @staticmethod
    def lg_p5318(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5318
        tag: bfs|topological_sort|dfs_order
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            dct[x].append(y)
        for i in range(n):
            dct[i].sort()

        edge = [ls[::-1] for ls in dct]
        stack = [0]
        dfs = []
        visit = [0] * n
        while stack:
            x = stack.pop()
            if not visit[x]:
                dfs.append(x)
                visit[x] = 1
            while edge[x]:
                y = edge[x].pop()
                if not visit[y]:
                    stack.append(x)
                    stack.append(y)
                    break
        ac.lst([x + 1 for x in dfs])

        stack = [0]
        visit = [0] * n
        bfs = []
        visit[0] = 1
        while stack:
            bfs.extend(stack)
            nex = []
            for i in stack:
                for j in dct[i]:
                    if not visit[j]:
                        nex.append(j)
                        visit[j] = 1
            stack = nex
        ac.lst([x + 1 for x in bfs])
        return

    @staticmethod
    def lc_1080(root: Optional[TreeNode], limit: int) -> Optional[TreeNode]:
        """
        url: https://leetcode.cn/problems/insufficient-nodes-in-root-to-leaf-paths/description/
        tag: dfs|up_to_down|down_to_up
        """

        def dfs(node, lmt):
            if not node:
                return
            if not node.left and not node.right:
                if node.val < lmt:
                    return
                return node
            left = dfs(node.left, lmt - node.val)
            right = dfs(node.right, lmt - node.val)
            if not left and not right:
                return
            node.left = left
            node.right = right
            return node

        return dfs(root, limit)

    @staticmethod
    def lc_1239(arr: List[str]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-length-of-a-concatenated-string-with-unique-characters/
        tag: dfs|back_trace|2-base|brute_force
        """

        ans = 0
        arr = [word for word in arr if len(set(word)) == len(word)]
        n = len(arr)

        def dfs(i):
            nonlocal ans, pre
            if i == n:
                if len(pre) > ans:
                    ans = len(pre)
                return
            if not set(arr[i]).intersection(pre):
                pre |= set(arr[i])
                dfs(i + 1)
                pre -= set(arr[i])
            dfs(i + 1)
            return

        pre = set()
        dfs(0)
        return ans

    @staticmethod
    def lc_1240(n: int, m: int) -> int:
        """
        url: https://leetcode.cn/problems/tiling-a-rectangle-with-the-fewest-squares/
        tag: dfs|back_trace|prune
        """

        def dfs():
            nonlocal cnt, ans
            if cnt >= ans:  # prune
                return

            pre = PreFixSumMatrix([g[:] for g in grid])
            if pre.query(0, 0, m - 1, n - 1) == m * n:
                # prune
                ans = ans if ans < cnt else cnt
                return

            for i in range(m):
                for j in range(n):
                    if not grid[i][j]:
                        ceil = m - i
                        if n - j < ceil:
                            ceil = n - j
                        # brute_force left_up point
                        for x in range(ceil, 0, -1):
                            if pre.query(i, j, i + x - 1, j + x - 1) == 0 and cnt + 1 < ans:
                                for a in range(i, i + x):
                                    for b in range(j, j + x):
                                        grid[a][b] = 1
                                cnt += 1
                                dfs()
                                cnt -= 1
                                for a in range(i, i + x):
                                    for b in range(j, j + x):
                                        grid[a][b] = 0
                        return

            return

        grid = [[0] * n for _ in range(m)]
        ans = m * n
        cnt = 0
        dfs()
        return ans

    @staticmethod
    def lc_2056(pieces: List[str], positions: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/number-of-valid-move-combinations-on-chessboard/description/
        tag: back_trace|brute_force
        """

        dct = dict()
        dct["rook"] = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        dct["queen"] = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, 1], [1, 1], [1, -1], [-1, -1]]
        dct["bishop"] = [[-1, 1], [1, 1], [1, -1], [-1, -1]]

        ans = 0
        n = len(pieces)

        def dfs(i):
            nonlocal ans
            if i == n:
                ans += 1
                return
            x, y = positions[i]
            cnt = 0
            for a, b in dct[pieces[i]]:
                for step in range(8):
                    if step == 0 and cnt:
                        continue
                    cnt += 1
                    if not (1 <= x + a * step <= 8 and 1 <= y + b * step <= 8):
                        break
                    lst = [(x + a * s, y + b * s) for s in range(step + 1)]
                    while len(lst) < 8:
                        lst.append(lst[-1])
                    for ii, w in enumerate(lst):
                        if w in pre[ii]:
                            break
                    else:
                        for ii, w in enumerate(lst):
                            pre[ii].add(w)
                        dfs(i + 1)
                        for ii, w in enumerate(lst):
                            pre[ii].discard(w)

            return

        pre = [set() for _ in range(8)]
        dfs(0)
        return ans

    @staticmethod
    def lc_2322(nums: List[int], edges: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-score-after-removals-on-a-tree/
        tag: dfs_order|brute_force
        """

        n = len(nums)
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        order = 0
        start = [-1] * n
        end = [-1] * n
        parent = [-1] * n
        stack = [(0, -1)]
        sub = [0] * n
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                start[i] = order
                end[i] = order
                order += 1
                stack.append((~i, fa))
                for j in dct[i]:
                    if j != fa:
                        parent[j] = i
                        stack.append((j, i))
            else:
                i = ~i
                sub[i] = nums[i]
                for j in dct[i]:
                    if j != fa:
                        sub[i] ^= sub[j]
                if parent[i] != -1:
                    end[parent[i]] = end[i]

        total = sub[0]

        ans = math.inf
        for i in range(n - 1):
            x, y = edges[i]
            if parent[x] == y:
                x, y = y, x
            for j in range(i + 1, n - 1):
                a, b = edges[j]
                if parent[a] == b:
                    a, b = b, a
                yy = sub[y]
                bb = sub[b]
                if start[y] <= start[b] <= end[y]:
                    yy ^= bb
                if start[b] <= start[y] <= end[b]:
                    bb ^= yy
                cur = [yy, bb, total ^ yy ^ bb]
                ans = min(ans, max(cur) - min(cur))
        return ans

    @staticmethod
    def lc_2458(root: Optional[TreeNode], queries: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/height-of-binary-tree-after-subtree-removal-queries/
        tag: dfs|tree_dp|up_to_down|down_to_up|dfs
        """
        ans = defaultdict(int)
        stack = [(root, 0)]
        x = 0
        while stack:
            node, h = stack.pop()
            ans[node.val] = x
            if h > x:
                x = h
            if node.right:
                stack.append((node.right, h + 1))
            if node.left:
                stack.append((node.left, h + 1))

        stack = [(root, 0)]
        x = 0
        while stack:
            node, h = stack.pop()
            if x > ans[node.val]:
                ans[node.val] = x
            if h > x:
                x = h
            if node.left:
                stack.append((node.left, h + 1))
            if node.right:
                stack.append((node.right, h + 1))

        return [ans[i] for i in queries]

    @staticmethod
    def lc_2581(edges: List[List[int]], guesses: List[List[int]], k: int) -> int:
        """
        url: https://leetcode.cn/problems/count-number-of-possible-root-nodes
        tag: dfs_order|diff_array|counter|reroot_dp
        """

        n = len(edges) + 1
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        visit, interval = DFS().gen_bfs_order_iteration(dct)
        diff = [0] * n
        for u, v in guesses:
            if visit[u] <= visit[v]:
                a, b = interval[v]
                lst = [[0, a - 1], [b + 1, n - 1]]
            else:
                a, b = interval[u]
                lst = [[a, b]]

            for x, y in lst:
                if x <= y:
                    diff[x] += 1
                    if y + 1 < n:
                        diff[y + 1] -= 1

        for i in range(1, n):
            diff[i] += diff[i - 1]
        return sum(x >= k for x in diff)

    @staticmethod
    def cf_219d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/219/problem/D
        tag: reroot_dp|dfs|dfs_order|diff_array
        """

        n = ac.read_int()
        edges = [ac.read_list_ints_minus_one() for _ in range(n - 1)]

        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)
        start, end = DFS().gen_bfs_order_iteration(dct)
        diff = [0] * n
        for i, j in edges:
            if start[i] < start[j]:
                a, b = start[j], end[j]
                diff[a] += 1
                if b + 1 < n:
                    diff[b + 1] -= 1
            else:
                a, b = start[i], end[i]
                if 0 <= a - 1:
                    diff[0] += 1
                    diff[a] -= 1
                if b + 1 <= n - 1:
                    diff[b + 1] += 1
        diff = ac.accumulate(diff)[1:]
        res = [diff[start[i]] for i in range(n)]
        low = min(res)
        ac.st(low)
        ac.lst([i + 1 for i in range(n) if res[i] == low])
        return

    @staticmethod
    def cf_570d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/570/problem/D
        tag: dfs_order|binary_search|offline_query|classical
        """

        n, m = ac.read_list_ints()
        parent = ac.read_list_ints_minus_one()
        edge = [[] for _ in range(n)]
        for i in range(n - 1):
            edge[parent[i]].append(i + 1)
        s = ac.read_str()

        queries = [[] for _ in range(n)]
        for i in range(m):
            v, h = ac.read_list_ints()
            queries[v - 1].append([h - 1, i])

        ans = [0] * m
        depth = [0] * n

        stack = [(0, 1, 0)]
        while stack:
            i, state, height = stack.pop()
            if state:
                for h, j in queries[i]:
                    ans[j] ^= depth[h]
                depth[height] ^= 1 << (ord(s[i]) - ord("a"))
                stack.append((i, 0, height))
                for j in edge[i]:
                    stack.append((j, 1, height + 1))
            else:
                for h, j in queries[i]:
                    ans[j] ^= depth[h]
        for a in ans:
            ac.st("Yes" if a & (a - 1) == 0 else "No")
        return

    @staticmethod
    def cf_208e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/208/problem/E
        tag: dfs_order|lca|binary_search|counter
        """

        n = ac.read_int()
        parent = ac.read_list_ints()

        graph = WeightedTree(n + 1)
        for i in range(n):
            graph.add_directed_edge(parent[i], i + 1)
        graph.dfs_order()
        graph.lca_build_with_multiplication()

        dct = [[] for _ in range(n + 1)]
        for i in range(n + 1):
            dct[graph.depth[graph.order_to_node[i]]].append(i)

        ans = []
        for _ in range(ac.read_int()):
            v, p = ac.read_list_ints()
            if graph.depth[v] - 1 < p:
                ans.append(0)
                continue
            u = graph.lca_get_kth_ancestor(v, p)
            low, high = graph.start[u], graph.end[u]
            cur = bisect.bisect_right(dct[graph.depth[v]], high)
            cur -= bisect.bisect_left(dct[graph.depth[v]], low)
            ans.append(max(cur - 1, 0))
        ac.lst(ans)
        return

    @staticmethod
    def lg_p8838(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8838
        tag: dfs|back_trace
        """
        n, k = ac.read_list_ints()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        if n < k:
            ac.st(-1)
            return
        for item in permutations(list(range(n)), k):
            if all(a[item[i]] >= b[i] for i in range(k)):
                ac.lst([x + 1 for x in item])
                break
        else:
            ac.st(-1)
        return

    @staticmethod
    def ac_4313(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4313/
        tag: dfs_order|template
        """
        n, q = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        nums = ac.read_list_ints_minus_one()
        for i in range(n - 1):
            dct[nums[i]].append(i + 1)
        for i in range(n):
            dct[i].sort(reverse=True)

        start, end = DFS().gen_bfs_order_iteration(dct)
        ind = {num: i for i, num in enumerate(start)}
        for _ in range(q):
            u, k = ac.read_list_ints()
            u -= 1
            if end[u] - start[u] + 1 < k:
                ac.st(-1)
            else:
                ac.st(ind[start[u] + k - 1] + 1)
        return

    @staticmethod
    def abc_133f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc133/tasks/abc133_f
        tag: euler_order|online_tree_dis|binary_search|prefix_sum
        """
        n, q = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        graph = WeightedTree(n)
        for _ in range(n - 1):
            a, b, c, d = ac.read_list_ints()
            a -= 1
            b -= 1
            dct[a][b] = [c, d]
            dct[b][a] = [c, d]
            graph.add_undirected_edge(a, b, 1)
        graph.lca_build_with_multiplication()
        dis = [0] * n
        stack = [[0, -1]]
        while stack:
            x, fa = stack.pop()
            for y in dct[x]:
                if y != fa:
                    c, d = dct[x][y]
                    dct[x][y] = [c, d]
                    dct[y][x] = [-c, d]
                    dis[y] = dis[x] + d
                    stack.append([y, x])
        graph.dfs_euler_order()
        m = len(graph.euler_order)
        euler_ind = [-1] * n
        color_pos_ind = defaultdict(list)
        color_neg_ind = defaultdict(list)
        color_pos_pre = defaultdict(lambda: [0])
        color_neg_pre = defaultdict(lambda: [0])
        for i in range(m):
            if euler_ind[graph.euler_order[i]] == -1:
                euler_ind[graph.euler_order[i]] = i
            if i:
                a, b = graph.euler_order[i - 1], graph.euler_order[i]
                c, d = dct[a][b]
                if c > 0:
                    color_pos_ind[c].append(i)
                    color_pos_pre[c].append(color_pos_pre[c][-1] + d)
                else:
                    color_neg_ind[-c].append(i)
                    color_neg_pre[-c].append(color_neg_pre[-c][-1] + d)

        for _ in range(q):
            x, y, u, v = ac.read_list_ints()
            u -= 1
            v -= 1
            ancestor = graph.lca_get_lca_between_nodes(u, v)
            if euler_ind[u] > euler_ind[v]:
                u, v = v, u
            cur_dis = dict()
            for w in [u, v, ancestor]:
                start, end = euler_ind[0] + 1, euler_ind[w]
                pos_range = [bisect_left(color_pos_ind[x], start), bisect_right(color_pos_ind[x], end)]
                neg_range = [bisect_left(color_neg_ind[x], start), bisect_right(color_neg_ind[x], end)]

                pre_color = color_pos_pre[x][pos_range[1]] - color_pos_pre[x][pos_range[0]]
                pre_color -= color_neg_pre[x][neg_range[1]] - color_neg_pre[x][neg_range[0]]

                post_color_cnt = pos_range[1] - pos_range[0] - (neg_range[1] - neg_range[0])
                cur_dis[w] = dis[w] - pre_color + post_color_cnt * y
            ac.st(cur_dis[u] + cur_dis[v] - 2 * cur_dis[ancestor])
        return

    @staticmethod
    def ac_23(matrix, string):
        """
        url: https://www.acwing.com/problem/content/description/21/
        tag: back_trace|template
        """
        if not matrix:
            return False

        m, n = len(matrix), len(matrix[0])

        def dfs(ind, x, y):
            nonlocal ans
            if ind == k or ans:
                ans = True
                return
            for a, b in [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]:
                if 0 <= a < m and 0 <= b < n and not visit[a][b] and matrix[a][b] == string[ind]:
                    visit[a][b] = 1
                    dfs(ind + 1, a, b)
                    visit[a][b] = 0
            return

        ans = False
        visit = [[0] * n for _ in range(m)]
        k = len(string)
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == string[0]:
                    visit[i][j] = 1
                    dfs(1, i, j)
                    visit[i][j] = 0
                    if ans:
                        return True
        return False

    @staticmethod
    def abc_337g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc337/tasks/abc337_g
        tag: dfs_order|contribution_method|classical|tree_array
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        start, end = DFS().gen_bfs_order_iteration(dct, 0)

        diff = [0] * (n + 1)

        tree = PointAddRangeSum(n)
        for i in range(n):
            v = i
            for j in dct[i]:
                if start[j] > start[i]:
                    x = tree.range_sum(start[j] + 1, end[j] + 1)
                    if 0 <= start[j] - 1:
                        diff[0] += x  # [0, start[j]-1]
                        diff[start[j]] -= x
                    if end[j] + 1 <= n - 1:
                        diff[end[j] + 1] += x  # [end[j]+1, n-1]
                        diff[n] -= x
                    v -= x
            # [start[i], end[i]]
            diff[start[i]] += v
            diff[end[i] + 1] -= v
            tree.point_add(start[i] + 1, 1)

        for i in range(1, n):
            diff[i] += diff[i - 1]
        ac.lst([diff[start[i]] for i in range(n)])
        return

    @staticmethod
    def cf_246e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/246/E
        tag: tree_array|offline_query|range_unique|dfs_order
        """
        n = ac.read_int()
        n += 1
        names = [""] * n
        parent = [-1] * n
        dct = [[] for _ in range(n)]
        for i in range(1, n):
            name, fa = ac.read_list_strs()
            fa = int(fa)
            names[i] = name
            parent[i] = fa
            dct[fa].append(i)
        ind = {name: i for i, name in enumerate(list(set(names)))}
        names = [ind[x] for x in names]
        del ind

        order = 0
        start = [-1] * n
        end = [-1] * n
        stack = [(0, -1)]
        depth = [0] * n
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                start[i] = order
                end[i] = order
                order += 1
                stack.append((~i, fa))
                for j in dct[i]:
                    depth[j] = depth[i] + 1
                    stack.append((j, i))
            else:
                i = ~i
                if parent[i] != -1:
                    end[parent[i]] = end[i]

        m = ac.read_int()
        queries = [ac.read_list_ints() for _ in range(m)]
        dct = [[] for _ in range(n)]
        for i in range(m):
            v, k = queries[i]
            if depth[v] + k <= n - 1:
                dct[depth[v] + k].append((start[v], end[v], i))
        nodes = [[] for _ in range(n)]
        for i in range(n):
            nodes[depth[i]].append(i)

        ans = [0] * m
        pre = [-1] * n
        for d in range(n):
            lst = nodes[d]

            lst.sort(key=lambda it: start[it])
            dfs_order = [start[x] for x in lst]
            check = [(bisect.bisect_left(dfs_order, s), bisect.bisect_right(dfs_order, e) - 1, i) for s, e, i in dct[d]]
            for x in lst:
                pre[names[x]] = -1
            k = len(lst)
            tree = PointAddRangeSum(k)
            check.sort(key=lambda it: it[1])

            i = 0
            for ll, rr, ii in check:
                while i <= rr:
                    num = names[lst[i]]
                    if pre[num] != -1:
                        tree.point_add(pre[num] + 1, -1)
                    pre[num] = i
                    tree.point_add(i + 1, 1)
                    i += 1
                ans[ii] = tree.range_sum(ll + 1, rr + 1)
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def cf_1076e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1076/E
        tag: tree_diff_array|dfs|classical
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            dct[x].append(y)
            dct[y].append(x)
        ops = [[] for _ in range(n)]
        for _ in range(ac.read_int()):
            v, d, x = ac.read_list_ints()
            v -= 1
            ops[v].append((d, x))
        pre = [0] * (n + 1)
        ans = [0] * n
        stack = [(0, -1)]
        depth = [0] * n
        while stack:
            x, fa = stack.pop()
            if x >= 0:
                stack.append((~x, fa))
                if depth[x]:
                    pre[depth[x]] += pre[depth[x] - 1]
                for d, v in ops[x]:
                    pre[depth[x]] += v
                    if depth[x] + d + 1 < n:
                        pre[depth[x] + d + 1] -= v
                ans[x] = pre[depth[x]]
                for y in dct[x]:
                    if y != fa:
                        stack.append((y, x))
                        depth[y] = depth[x] + 1
            else:
                x = ~x
                if depth[x]:
                    pre[depth[x]] -= pre[depth[x] - 1]
                for d, v in ops[x]:
                    pre[depth[x]] -= v
                    if depth[x] + d + 1 < n:
                        pre[depth[x] + d + 1] += v
        ac.lst(ans)
        return

    @staticmethod
    def abc_328e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc328/tasks/abc328_e
        tag: dfs|back_trace|union_find|brute_force
        """
        n, m, k = ac.read_list_ints()
        edges = [ac.read_list_ints() for _ in range(m)]

        def dfs(i):
            if uf.part == 1:
                ans[0] = min(ans[0], pre[0] % k)
                return
            if i == m:
                return
            dfs(i + 1)
            u, v, w = edges[i]
            if not uf.is_connected(u - 1, v - 1):
                cur = uf.root_or_size[:]
                uf.union(u - 1, v - 1)
                pre[0] += w
                dfs(i + 1)
                pre[0] -= w
                uf.root_or_size[:] = cur[:]
                uf.part += 1
            return

        ans = [k + 1]
        pre = [0]
        uf = UnionFind(n)
        dfs(0)
        ac.st(ans[0])
        return

    @staticmethod
    def abc_284e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc284/tasks/abc284_e
        tag: dfs|back_trace|classical
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        ans = [0]
        visit = [0] * n
        visit[0] = 1
        ceil = 10 ** 6

        @ac.bootstrap
        def dfs(x):
            if ans[0] > ceil:
                yield
            ans[0] += 1
            for y in dct[x]:
                if not visit[y]:
                    visit[y] = 1
                    yield dfs(y)
                    visit[y] = 0
            yield

        dfs(0)
        ac.st(min(ans[0], ceil))
        return

    @staticmethod
    def abc_268d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc268/tasks/abc268_d
        tag: dfs|back_trace|prune|classical
        """
        n, m = ac.read_list_ints()
        words = [ac.read_str() for _ in range(n)]
        forbid = set([ac.read_str() for _ in range(m)])

        ans = ""
        for lst in permutations(words, n):
            pre_len = ac.accumulate([len(w) for w in lst])

            if ans:
                break

            def dfs(i):
                nonlocal pre, ans
                if ans:
                    return
                if i == n:
                    if 3 <= len(pre) <= 16 and pre not in forbid:
                        ans = pre
                    return
                if len(pre) + n - i + pre_len[-1] - pre_len[i] > 16:
                    return

                for c in range(1, 20):
                    if len(pre) + c + pre_len[-1] - pre_len[i] + n - i - 1 <= 16:
                        tmp = pre
                        pre += "_" * c + lst[i]
                        dfs(i + 1)
                        pre = tmp
                    else:
                        break
                return

            pre = lst[0]
            dfs(1)
        ac.st(ans if ans else -1)
        return

    @staticmethod
    def abc_244g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc244/tasks/abc244_g
        tag: construction|euler_order|brain_teaser|classical
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        uf = UnionFind(n)
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            if uf.union(i, j):
                dct[i].append(j)
                dct[j].append(i)
        s = [int(w) for w in ac.read_str()]
        t = [0] * n
        euler_order = []
        root = 0
        stack = [(root, -1)]
        parent = [-1] * n
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                euler_order.append(i)
                t[i] ^= 1
                stack.append((~i, fa))
                for j in dct[i]:
                    if j != fa:
                        parent[j] = i
                        stack.append((j, i))
            else:
                i = ~i
                if i != root:
                    euler_order.append(parent[i])
                    t[parent[i]] ^= 1
                    if t[i] != s[i]:
                        euler_order.append(i)
                        t[i] ^= 1
                        euler_order.append(parent[i])
                        t[parent[i]] ^= 1
                elif t[i] != s[i]:
                    euler_order.append(dct[i][0])
                    euler_order.append(i)
                    t[i] ^= 1
                    euler_order.append(dct[i][0])
        assert t == s
        ac.st(len(euler_order))
        ac.lst([x + 1 for x in euler_order])
        return

    @staticmethod
    def abc_240e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc240/tasks/abc240_e
        tag: dfs_order|leaf|classical
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        start = [0] * n
        end = [0] * n
        order = 0
        stack = [(0, -1)]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append((~i, fa))
                cnt = 0
                for j in dct[i]:
                    if j != fa:
                        stack.append((j, i))
                        cnt += 1
                if not cnt:
                    start[i] = order
                    end[i] = order
                    order += 1
                    stack.pop()
            else:
                i = ~i
                s = math.inf
                e = -math.inf
                for j in dct[i]:
                    if j != fa:
                        s = min(s, start[j])
                        e = max(e, end[j])
                start[i] = s
                end[i] = e
        for i in range(n):
            ac.lst([start[i] + 1, end[i] + 1])
        return

    @staticmethod
    def abc_236d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc236/tasks/abc236_d
        tag: back_trace|prune|brute_force|classical
        """
        n = ac.read_int()
        grid = [[0] * 2 * n for _ in range(2 * n)]
        for i in range(2 * n - 1):
            lst = ac.read_list_ints()
            for j in range(i + 1, 2 * n):
                grid[i][j] = lst[j - i - 1]

        def dfs(x):
            if x == n:
                ans[0] = max(ans[0], pre[0])
                return
            ind = [x for x in range(2 * n) if not visit[x]]
            a = ind[0]
            for b in ind[1:]:
                pre[0] ^= grid[a][b]
                visit[a] = visit[b] = 1
                dfs(x + 1)
                pre[0] ^= grid[a][b]
                visit[a] = visit[b] = 0
            return

        ans = [-math.inf]
        pre = [0]
        visit = [0] * 2 * n
        dfs(0)
        ac.st(ans[0])
        return

    @staticmethod
    def cf_383c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/383/C
        tag: dfs_order|odd_even|range_add|point_get
        """
        n, q = ac.read_list_ints()
        nums = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        res = DFS().gen_bfs_order_iteration(dct, 0)
        start, end, depth, parent = res
        dfn = [0] * n
        for i in range(n):
            dfn[start[i]] = i

        odd_tree = RangeAddPointGet(n)
        odd_tree.build([nums[dfn[i]] for i in range(n)])
        even_tree = RangeAddPointGet(n)
        even_tree.build([nums[dfn[i]] for i in range(n)])

        for _ in range(q):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                v, c = lst[1], lst[2]
                v -= 1
                if depth[v] % 2:
                    even_tree.range_add(start[v], end[v], -c)
                    odd_tree.range_add(start[v], end[v], c)
                else:
                    even_tree.range_add(start[v], end[v], c)
                    odd_tree.range_add(start[v], end[v], -c)
            else:
                v = lst[1]
                v -= 1
                if depth[v] % 2:
                    ans = odd_tree.point_get(start[v])
                else:
                    ans = even_tree.point_get(start[v])
                ac.st(ans)
        return

    @staticmethod
    def cf_3c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/3/C
        tag: dfs|back_trace|brute_force|implemention
        """

        def check():
            for i in range(3):
                if all(grid[i][j] == "X" for j in range(3)):
                    return "the first player won"
                if all(grid[i][j] == "0" for j in range(3)):
                    return "the second player won"
            for j in range(3):
                if all(grid[i][j] == "X" for i in range(3)):
                    return "the first player won"
                if all(grid[i][j] == "0" for i in range(3)):
                    return "the second player won"

            if all(grid[i][i] == "X" for i in range(3)):
                return "the first player won"
            if all(grid[i][i] == "0" for i in range(3)):
                return "the second player won"
            if all(grid[i][2 - i] == "X" for i in range(3)):
                return "the first player won"
            if all(grid[i][2 - i] == "0" for i in range(3)):
                return "the second player won"
            if all(grid[i][j] != "." for i in range(3) for j in range(3)):
                return "draw"
            return ""

        def dfs():
            if state in ans:
                return
            cur_state = "".join("".join(ls) for ls in grid)
            if cur_state in ans:
                return
            res = check()
            if res:
                ans[cur_state] = res
                return
            order[0] = 1 - order[0]
            ans[cur_state] = "first" if order[0] == 0 else "second"
            for i in range(3):
                for j in range(3):
                    if grid[i][j] == "." and st[order[0]] == cur[i][j]:
                        grid[i][j] = st[order[0]]
                        dfs()
                        grid[i][j] = "."
            order[0] = 1 - order[0]
            return

        grid = [["."] * 3 for _ in range(3)]
        st = "X0"
        order = [1]
        cur = [ac.read_str() for _ in range(3)]
        state = "".join(cur)
        ans = dict()
        dfs()
        if state not in ans:
            ac.st("illegal")
        else:
            ac.st(ans[state])
        return

    @staticmethod
    def cf_459c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/459/C
        tag: back_trace|brute_force|classical|implemention
        """
        n, k, d = ac.read_list_ints()

        pre = []
        res = []

        def dfs():
            if len(res) == n:
                return
            if len(pre) == d:
                res.append(pre[:])
                return

            for xx in range(1, k + 1):
                pre.append(xx)
                dfs()
                pre.pop()
                if len(res) == n:
                    break
            return

        dfs()
        if len(res) < n:
            ac.st(-1)
            return
        ans = [[0] * n for _ in range(d)]
        for ls in res:
            if not n:
                break
            n -= 1
            for x in range(d):
                ans[x][n] = ls[x]
        for ls in ans:
            ac.lst(ls)
        return

    @staticmethod
    def cf_1918f(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1918/F
        tag: dfs_order|greed|tree_lca|implemention|observation|brain_teaser
        """
        n, k = ac.read_list_ints()
        k += 1
        parent = [-1] + ac.read_list_ints_minus_one()
        dct = [[] for _ in range(n)]
        for i in range(1, n):
            dct[parent[i]].append(i)
        depth = [0] * n
        sub = [0] * n
        stack = [0]
        while stack:
            x = stack.pop()
            if x >= 0:
                stack.append(~x)
                for y in dct[x]:
                    depth[y] = depth[x] + 1
                    stack.append(y)
            else:
                x = ~x
                sub[x] = depth[x]
                for y in dct[x]:
                    sub[x] = max(sub[x], sub[y])
        for i in range(1, n):
            dct[i].sort(key=lambda it: -sub[it])

        order = 0
        start = [-1] * n
        stack = [0]
        while stack:
            i = stack.pop()
            start[i] = order
            order += 1
            for j in dct[i]:
                parent[j] = i
                stack.append(j)

        leaf = [i for i in range(n) if not dct[i]]
        leaf.sort(key=lambda it: start[it])
        m = len(leaf)
        queries = [(leaf[i], leaf[i + 1]) for i in range(m - 1)]
        ancestor = OfflineLCA().bfs_iteration(dct, queries)
        ans = 2 * (n - 1)
        dis = [depth[leaf[-1]]]
        for i in range(m - 1):
            cur = max(0, depth[leaf[i]] - 2 * depth[ancestor[i]])
            dis.append(cur)
        dis.sort(reverse=True)
        ans -= sum(dis[:k])
        ac.st(ans)
        return

    @staticmethod
    def lg_p1444(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1444
        tag: dfs|back_trace|circle_check|brain_teaser|observation
        """
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        edges = []
        dct = defaultdict(list)
        for i, (x, y) in enumerate(nums):
            dct[y].append((x, i))
        for y in dct:
            dct[y].sort()
            k = len(dct[y])
            for i in range(k - 1):
                edges.append((dct[y][i][1], dct[y][i + 1][1]))
        nums.sort(key=lambda it: (it[1], it[0]))

        def check():
            walk = [-1] * n
            skip = [-1] * n
            for xx, yy in pre:
                skip[xx] = yy
                skip[yy] = xx
            for xx, yy in edges:
                if skip[xx] != yy:
                    walk[xx] = yy
                else:
                    return 1

            for s in range(n):
                cur = {s}
                while s != -1:
                    if walk[s] == -1:
                        break
                    s = skip[walk[s]]
                    if s in cur:
                        return 1
                    cur.add(s)
            return 0

        visit = [0] * n

        def dfs():
            if len(pre) * 2 == n:
                ans[0] += check()
                return

            for ind in range(n):
                if not visit[ind]:
                    visit[ind] = 1
                    for nex in range(ind + 1, n):
                        if not visit[nex]:
                            visit[nex] = 1
                            pre.append((ind, nex))
                            dfs()
                            visit[nex] = 0
                            pre.pop()
                    visit[ind] = 0
                    break
            return

        ans = [0]
        pre = []
        dfs()
        ac.st(ans[0])
        return

    @staticmethod
    def cf_1882d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1882/D
        tag: dfs_order|diff_array|contribution_method|greed
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            dct = [[] for _ in range(n)]
            edges = [ac.read_list_ints_minus_one() for _ in range(n - 1)]
            for i, j in edges:
                dct[i].append(j)
                dct[j].append(i)

            order = 0
            start = [-1] * n
            end = [-1] * n
            parent = [-1] * n
            stack = [0]
            # depth of every original node
            depth = [0] * n
            # index is dfs order and value is original node
            order_to_node = [-1] * n
            sub = [1] * n
            diff = [0] * n
            while stack:
                val = stack.pop()
                if val >= 0:
                    i, fa = val // n, val % n
                    stack.append(~val)
                    start[i] = order
                    order_to_node[order] = i
                    end[i] = order
                    order += 1
                    for j in dct[i]:
                        # the order of son nodes can be assigned for lexicographical order
                        if j != fa:
                            parent[j] = i
                            depth[j] = depth[i] + 1
                            stack.append(j * n + i)
                else:
                    val = ~val
                    i, fa = val // n, val % n
                    if parent[i] != -1:
                        end[parent[i]] = end[i]
                    for j in dct[i]:
                        # the order of son nodes can be assigned for lexicographical order
                        if j != fa:
                            sub[i] += sub[j]
                            cur = nums[i] ^ nums[j]
                            s, e = start[j], end[j]
                            diff[0] += sub[j] * cur
                            diff[s] += (n - 2 * sub[j]) * cur
                            if e + 1 < n:
                                diff[e + 1] -= (n - 2 * sub[j]) * cur
            for i in range(1, n):
                diff[i] += diff[i - 1]
            ac.lst([diff[start[i]] for i in range(n)])
        return

    @staticmethod
    def cf_1009f(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1009/F
        tag: heuristic_merge|classical
        """
        n = ac.read_int()
        graph = WeightedTree(n)
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            graph.add_undirected_edge(x, y)
        ans = graph.heuristic_merge()
        ac.flatten(ans)
        return

    @staticmethod
    def lc_3327(parent: List[int], s: str) -> List[bool]:
        """
        url: https://leetcode.cn/problems/check-if-dfs-strings-are-palindromes/
        tag: dfs_order|manacher|palindrome|classical
        """
        n = len(parent)
        graph = WeightedTree(n)
        for i in range(n - 1, 0, -1):
            graph.add_directed_edge(parent[i], i)
        graph.dfs_order()
        ss = [s[i] for i in graph.order_to_node]
        t = "#" + "#".join(list(ss)) + "#"
        arm = ManacherPlindrome().manacher(t)
        ans = []
        for i in range(n):
            a, b = graph.start[i], graph.end[i]
            a = 2 * a + 1
            b = 2 * b + 1
            mid = a + (b - a) // 2
            ans.append(mid - arm[mid] + 1 <= a)
        return ans
