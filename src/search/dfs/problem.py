"""

Algorithm：dfs|coloring_method|brute_force|back_track|euler_order|dfs_order|prune|itertation
Description：back_track|brute_force|dfs_order|up_to_down|down_to_up


====================================LeetCode====================================
473（https://leetcode.cn/problems/matchsticks-to-square/）dfs|back_track
301（https://leetcode.cn/problems/remove-invalid-parentheses/）back_track|dfs|prune
2581（https://leetcode.cn/problems/count-number-of-possible-root-nodes）dfs_order|diff_array|counter|reroot_dp
1059（https://leetcode.cn/problems/all-paths-from-source-lead-to-destination/）memory_search|dfs|back_track
1718（https://leetcode.cn/problems/construct-the-lexicographically-largest-valid-sequence/）back_track
2322（https://leetcode.cn/problems/minimum-score-after-removals-on-a-tree/）dfs_order|brute_force
1240（https://leetcode.cn/problems/tiling-a-rectangle-with-the-fewest-squares/）dfs|back_track|prune
1239（https://leetcode.cn/problems/maximum-length-of-a-concatenated-string-with-unique-characters/）dfs|back_track|2-base|brute_force
1080（https://leetcode.cn/problems/insufficient-nodes-in-root-to-leaf-paths/description/）dfs|up_to_down|down_to_up
2056（https://leetcode.cn/problems/number-of-valid-move-combinations-on-chessboard/description/）back_track|brute_force
2458（https://leetcode.cn/problems/height-of-binary-tree-after-subtree-removal-queries）dfs_order|classical
2858（https://leetcode.cn/problems/minimum-edge-reversals-so-every-node-is-reachable/）reroot_dp|dfs|dfs_order|diff_array

=====================================LuoGu======================================
P2383（https://www.luogu.com.cn/problem/P2383）dfs|back_track
P1120（https://www.luogu.com.cn/problem/P1120）dfs|back_track
P1692（https://www.luogu.com.cn/problem/P1692）dfs|brute_force|lexicographical_order
P1612（https://www.luogu.com.cn/problem/P1612）dfs|prefix_sum|binary_search
P1475（https://www.luogu.com.cn/problem/P1475）dfs
P2080（https://www.luogu.com.cn/problem/P2080）dfs|back_track|prune
P2090（https://www.luogu.com.cn/problem/P2090）dfs|greedy|back_track|prune|euclidean_division|euclidean_minus
P2420（https://www.luogu.com.cn/problem/P2420）brain_teaser|dfs|shortest_path|xor_path|classical
P1473（https://www.luogu.com.cn/problem/P1473）dfs|brute_force
P1461（https://www.luogu.com.cn/problem/P1461）dfs|back_track|brute_force
P1394（https://www.luogu.com.cn/problem/P1394）dfs
P1180（https://www.luogu.com.cn/problem/P1180）dfs|implemention
P1118（https://www.luogu.com.cn/problem/P1118）implemention|lexicographical_order|dfs
P3252（https://www.luogu.com.cn/problem/P3252）dfs|back_track|prefix_sum|hash
P4913（https://www.luogu.com.cn/problem/P4913）dfs
P5118（https://www.luogu.com.cn/problem/P5118）dfs|back_track|hash|implemention
P5197（https://www.luogu.com.cn/problem/P5197）tree_dp|implemention|coloring_method
P5198（https://www.luogu.com.cn/problem/P5198）union_find
P5318（https://www.luogu.com.cn/problem/P5318）bfs|topological_sort|dfs_order
P6691（https://www.luogu.com.cn/problem/P6691）coloring_method|bipartite_graph|specific_plan|counter
P7370（https://www.luogu.com.cn/problem/P7370）ancestor
P1036（https://www.luogu.com.cn/problem/P1036）back_track|prune
P8578（https://www.luogu.com.cn/problem/P8578）greedy|dfs_order
P8838（https://www.luogu.com.cn/problem/P8838）dfs|back_track

===================================CodeForces===================================
570D（https://codeforces.com/contest/570/problem/D）dfs_order|binary_search|offline_query
208E（https://codeforces.com/contest/208/problem/E）dfs_order|lca|binary_search|counter
1006E（https://codeforces.com/contest/1006/problem/E）dfs_order|template
1702G2（https://codeforces.com/contest/1702/problem/G2）dfs_order|lca
1899G（https://codeforces.com/contest/1899/problem/G）dfs|inclusion_exclusion|classical|point_add_range_sum|heuristic_merge
1714G（https://codeforces.com/contest/1714/problem/G）dfs|binary_search|prefix_sum
1675F（https://codeforces.com/contest/1675/problem/F）dfs_order|greedy
219D（https://codeforces.com/contest/219/problem/D）reroot_dp|dfs|dfs_order|diff_array
246E（https://codeforces.com/problemset/problem/246/E）tree_array|offline_query|range_unique|dfs_order
1076E（https://codeforces.com/problemset/problem/1076/E）tree_diff_array|dfs|classical

====================================AtCoder=====================================
ABC133F（https://atcoder.jp/contests/abc133/tasks/abc133_f）euler_order|online_tree_dis|binary_search|prefix_sum
ABC337G（https://atcoder.jp/contests/abc337/tasks/abc337_g）dfs_order|contribution_method|classical|tree_array

=====================================AcWing=====================================
4313（https://www.acwing.com/problem/content/4313/）dfs_order|template
21（https://www.acwing.com/problem/content/description/21/）back_track|template

"""

import bisect
from bisect import bisect_right, bisect_left
from collections import defaultdict
from itertools import accumulate, permutations
from typing import List, Optional

from src.basis.diff_array.template import PreFixSumMatrix
from src.basis.tree_node.template import TreeNode
from src.data_structure.tree_array.template import PointAddRangeSum
from src.graph.tree_lca.template import TreeAncestor
from src.search.dfs.template import DFS, DfsEulerOrder
from src.utils.fast_io import FastIO
from src.utils.fast_io import inf


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_473_1(matchsticks: List[int]) -> bool:
        """
        url: https://leetcode.cn/problems/matchsticks-to-square/
        tag: dfs|back_track|state_dp|classical
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
        tag: dfs|back_track|state_dp|classical
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
        tag: dfs|back_track|state_dp|classical
        """
        for _ in range(ac.read_int()):
            matchsticks = ac.read_list_ints()[1:]
            n = len(matchsticks)
            tot = sum(matchsticks)
            if tot % 4:
                ac.st("no")
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
        tag: back_track|dfs|prune
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
        tag: dfs|back_track|2-base|brute_force
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
        tag: dfs|back_track|prune
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
        tag: back_track|brute_force
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

        ans = inf
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

        edge = [[] for _ in range(n + 1)]
        for i in range(n):
            edge[parent[i]].append(i + 1)
            edge[i + 1].append(parent[i])
        del parent

        tree = TreeAncestor(edge)
        start, end = DFS().gen_bfs_order_iteration(edge)

        dct = [[] for _ in range(n + 1)]
        for i in range(n + 1):
            dct[tree.depth[i]].append(start[i])
        for i in range(n + 1):
            dct[i].sort()

        ans = []
        for _ in range(ac.read_int()):
            v, p = ac.read_list_ints()
            if tree.depth[v] - 1 < p:
                ans.append(0)
                continue
            u = tree.get_kth_ancestor(v, p)
            low, high = start[u], end[u]
            cur = bisect.bisect_right(
                dct[tree.depth[v]], high) - bisect.bisect_left(dct[tree.depth[v]], low)
            ans.append(ac.max(cur - 1, 0))
        ac.lst(ans)
        return

    @staticmethod
    def lg_p8838(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8838
        tag: dfs|back_track
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
        edges = [[] for _ in range(n)]
        for _ in range(n - 1):
            a, b, c, d = ac.read_list_ints()
            a -= 1
            b -= 1
            dct[a][b] = [c, d]
            dct[b][a] = [c, d]
            edges[a].append(b)
            edges[b].append(a)
        tree = TreeAncestor(edges)

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

        euler_order = DfsEulerOrder(edges).euler_order[:]
        m = len(euler_order)
        euler_ind = [-1] * n
        color_pos_ind = defaultdict(list)
        color_neg_ind = defaultdict(list)
        color_pos_pre = defaultdict(lambda: [0])
        color_neg_pre = defaultdict(lambda: [0])
        for i in range(m):
            if euler_ind[euler_order[i]] == -1:
                euler_ind[euler_order[i]] = i
            if i:
                a, b = euler_order[i - 1], euler_order[i]
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
            ancestor = tree.get_lca(u, v)
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
        tag: back_track|template
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
