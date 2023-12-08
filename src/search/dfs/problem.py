"""

Algorithm：深度优先搜索、染色法、brute_forceback_track、欧拉序、dfs_order|
Function：常与back_trackbrute_force结合，比较的还有DFS序


====================================LeetCode====================================
473（https://leetcode.com/problems/matchsticks-to-square/）搜索木棍拼接组成正方形
301（https://leetcode.com/problems/remove-invalid-parentheses/）深搜back_track与剪枝
2581（https://leetcode.com/contest/biweekly-contest-99/problems/count-number-of-possible-root-nodes/）dfs_order|差分counter
1059（https://leetcode.com/problems/all-paths-from-source-lead-to-destination/）记忆化搜索DFS深搜且back_track
1718（https://leetcode.com/problems/construct-the-lexicographically-largest-valid-sequence/）back_track
2322（https://leetcode.com/problems/minimum-score-after-removals-on-a-tree/）dfs_orderdfs_order|brute_force
1240（https://leetcode.com/problems/tiling-a-rectangle-with-the-fewest-squares/）DFSback_track与剪枝
1239（https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/）DFSback_track二进制brute_force
1080（https://leetcode.com/problems/insufficient-nodes-in-root-to-leaf-paths/description/）dfs自上而下后又自下而上
2056（https://leetcode.com/problems/number-of-valid-move-combinations-on-chessboard/description/）back_trackbrute_force
2458（https://leetcode.com/contest/weekly-contest-317/problems/height-of-binary-tree-after-subtree-removal-queries/）dfs_order|模板题目

=====================================LuoGu======================================
2383（https://www.luogu.com.cn/problem/P2383）搜索木棍拼接组成正方形
1120（https://www.luogu.com.cn/problem/P1120）把数组分成和相等的子数组
1692（https://www.luogu.com.cn/problem/P1692）搜索brute_forcelexicographical_order最大可行的连通块
1612（https://www.luogu.com.cn/problem/P1612）dfs记录路径的prefix_sum并binary_search确定最长链条
1475（https://www.luogu.com.cn/problem/P1475）深搜确定可以控制的公司对
2080（https://www.luogu.com.cn/problem/P2080）深搜back_track与剪枝
2090（https://www.luogu.com.cn/problem/P2090）深搜greedyback_track剪枝与辗转相减法
2420（https://www.luogu.com.cn/problem/P2420）brain_teaser深搜确定到根路径的异或结果以及异或特性获得任意两点之间最短路的异或结果
1473（https://www.luogu.com.cn/problem/P1473）深搜brute_force符号数
1461（https://www.luogu.com.cn/problem/P1461）汉明距离与深搜back_trackbrute_force
1394（https://www.luogu.com.cn/problem/P1394）深搜可达性确认
1180（https://www.luogu.com.cn/problem/P1180）深搜implemention
1118（https://www.luogu.com.cn/problem/P1118）单位矩阵implemention杨辉三角的系数，再暴搜寻找最小lexicographical_order结果
3252（https://www.luogu.com.cn/problem/P3252）深搜back_track|prefix_sumhash
4913（https://www.luogu.com.cn/problem/P4913）深搜确定深度
5118（https://www.luogu.com.cn/problem/P5118）深搜back_track与hash记录implemention
5197（https://www.luogu.com.cn/problem/P5197）树形DPimplemention与染色法，利用父亲与自己的染色确定儿子们的染色
5198（https://www.luogu.com.cn/problem/P5198）连通块的周长与面积
5318（https://www.luogu.com.cn/problem/P5318）广搜topological_sorting与dfs_order生成与获取
6691（https://www.luogu.com.cn/problem/P6691）染色法，bipartite_graph可行性方案counter与最大最小染色
7370（https://www.luogu.com.cn/problem/P7370）所有可能的祖先节点，注意特别情况没有任何祖先节点则自身可达
1036（https://www.luogu.com.cn/problem/P1036）back_track剪枝
8578（https://www.luogu.com.cn/problem/P8578）greedydfs_order
8838（https://www.luogu.com.cn/problem/P8838）深度优先搜索与back_track


===================================CodeForces===================================
570D（https://codeforces.com/contest/570/problem/D）dfs_order|与binary_search，也可以offline_query
208E（https://codeforces.com/contest/208/problem/E）dfs_order|LCA|binary_searchcounter
1006E（https://codeforces.com/contest/1006/problem/E）dfs_order|模板题
1702G2（https://codeforces.com/contest/1702/problem/G2）dfs_order|与lca组合判断是否为简单路径集合
1899G（https://codeforces.com/contest/1899/problem/G）dfs with tolerance and exclusion by PointAddRangeSum

====================================AtCoder=====================================
F - Colorful Tree（https://atcoder.jp/contests/abc133/tasks/abc133_f）欧拉序在线查找树上距离，结合binary_search与prefix_sum变化情况

=====================================AcWing=====================================
4310（https://www.acwing.com/problem/content/4313/）dfs_order模板题
23（https://www.acwing.com/problem/content/description/21/）back_track模板题

"""

import bisect
from bisect import bisect_right, bisect_left
from collections import defaultdict
from functools import reduce
from itertools import accumulate
from math import inf
from operator import xor
from typing import List, Optional

from src.basis.diff_array.template import PreFixSumMatrix
from src.basis.tree_node.template import TreeNode
from src.data_structure.tree_array.template import PointAddRangeSum
from src.graph.tree_lca.template import TreeAncestor
from src.search.dfs.template import DFS, DfsEulerOrder
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_473(matchsticks: List[int]) -> bool:
        # 模板: 深搜|back_track判断能否将数组分成正方形
        n, s = len(matchsticks), sum(matchsticks)
        if s % 4 or max(matchsticks) > s // 4:
            return False

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

        matchsticks.sort(reverse=True)  # 优化点
        m = s // 4
        ans = False
        pre = []
        dfs(0)
        return ans

    @staticmethod
    def lg_2383(ac=FastIO()):
        n = ac.read_int()
        for _ in range(n):
            nums = ac.read_list_ints()[1:]
            ans = Solution().lc_473(nums)
            if not ans:
                ac.st("no")
            else:
                ac.st("yes")
        return

    @staticmethod
    def lc_100041(n: int, edges: List[List[int]]) -> List[int]:
        # 迭代法实现树形换根DP，或者一遍DFS或者dfs_order||差分
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
        # dfs_order|模板题
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
        # 深搜back_track删除最少数量的无效括号使得子串合法有效

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
        # 深搜与广搜序获取
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n + 1)]
        degree = [0] * (n + 1)
        for _ in range(m):
            x, y = ac.read_list_ints()
            dct[x].append(y)
            degree[y] += 1
        for i in range(1, n + 1):
            dct[i].sort()

        # dfs_order值获取
        @ac.bootstrap
        def dfs(a):
            ans.append(a)
            visit[a] = 1
            for z in dct[a]:
                if not visit[z]:
                    yield dfs(z)
            yield

        visit = [0] * (n + 1)
        ans = []
        dfs(1)
        ac.lst(ans)

        # topological_sorting广搜
        ans = []
        stack = [1]
        visit = [0] * (n + 1)
        visit[1] = 1
        while stack:
            ans.extend(stack)
            nex = []
            for i in stack:
                for j in dct[i]:
                    if not visit[j]:
                        nex.append(j)
                        visit[j] = 1
            stack = nex
        ac.lst(ans)
        return

    @staticmethod
    def add_to_n(n):

        # 将 [1, 1] 通过 [a, b] 到 [a, a+b] 或者 [a+b, a] 的方式最少次数变成 a == n or b == n
        if n == 1:
            return 0

        def gcd_minus(a, b, c):
            nonlocal ans
            if c >= ans or not b:
                return
            assert a >= b
            if b == 1:
                ans = ans if ans < c + a - 1 else c + a - 1
                return

            # reverse_thinking保证使 b 减少到 a 以下
            gcd_minus(b, a % b, c + a // b)
            return

        ans = n - 1
        for i in range(1, n):
            gcd_minus(n, i, 0)
        return ans

    @staticmethod
    def lc_1080(root: Optional[TreeNode], limit: int) -> Optional[TreeNode]:

        # dfs自上而下后又自下而上
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
        # DFSback_track二进制brute_force
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
        # DFSback_track与剪枝

        def dfs():
            nonlocal cnt, ans
            if cnt >= ans:  # 超过最小值剪枝
                return

            pre = PreFixSumMatrix([g[:] for g in grid])
            if pre.query(0, 0, m - 1, n - 1) == m * n:
                # 全部填满剪枝
                ans = ans if ans < cnt else cnt
                return

            for i in range(m):
                for j in range(n):
                    if not grid[i][j]:
                        ceil = m - i
                        if n - j < ceil:
                            ceil = n - j
                        # brute_force此时左上端点正方形的长度
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
                        return  # 在第一个左上角端点剪枝

            return

        grid = [[0] * n for _ in range(m)]
        ans = m * n
        cnt = 0
        dfs()
        return ans

    @staticmethod
    def lc_2056(pieces: List[str], positions: List[List[int]]) -> int:
        # back_trackbrute_force
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
        # dfs_orderdfs_order|brute_force
        n = len(nums)
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        parent = [-1] * n
        start, end = DFS().gen_bfs_order_iteration(dct)

        # 预处理子树分数
        def dfs(xx, fa):
            res = nums[xx]
            for yyy in dct[xx]:
                if yyy != fa:
                    dfs(yyy, xx)
                    parent[yyy] = xx
                    res ^= sub[yyy]
            sub[xx] = res
            return

        total = reduce(xor, nums)
        sub = [0] * n
        dfs(0, -1)
        ans = inf
        # brute_force边对
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
                cur_ = max(cur) - min(cur)
                if cur_ < ans:
                    ans = cur_
        return ans

    @staticmethod
    def lc_2458(root: Optional[TreeNode], queries: List[int]) -> List[int]:
        # dfs_order|模板题目

        def dfs(node):
            nonlocal n
            if not node:
                return
            x = node.val - 1
            if x + 1 > n:
                n = x + 1
            if node.left:
                dct[x].append(node.left.val - 1)
                dfs(node.left)
            if node.right:
                dct[x].append(node.right.val - 1)
                dfs(node.right)
            return

        dct = defaultdict(list)
        n = 0
        dfs(root)
        edge = [dct[i] for i in range(n)]
        r = root.val - 1
        dfs_order = DfsEulerOrder(edge, r)
        pre = [0] + list(accumulate(dfs_order.order_depth[:], func=max))
        post = list(accumulate(dfs_order.order_depth[::-1], func=max))[::-1] + [0]

        ans = []
        for i in queries:
            i -= 1
            s, e = dfs_order.start[i], dfs_order.end[i]
            a = pre[s]
            b = post[e + 1]
            ans.append(a if a > b else b)
        return ans

    @staticmethod
    def lc_2581(edges: List[List[int]], guesses: List[List[int]], k: int) -> int:
        # dfs_order确定猜测的查询范围，并diff_array|counter
        n = len(edges) + 1
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        visit, interval = DFS().gen_bfs_order_iteration(dct)
        # 也可以
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
        # 迭代法实现树形换根DP，或者一遍DFS或者dfs_order||差分
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
    def cf_570d_1(ac=FastIO()):
        # dfs_order|与binary_searchcounter统计（超时）
        n, m = ac.read_list_ints()
        parent = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        for i in range(n - 1):
            edge[parent[i] - 1].append(i + 1)
        del parent
        s = ac.read_str()

        # 生成dfs_order即 dfs 序以及对应子树编号区间
        @ac.bootstrap
        def dfs(x, h):
            nonlocal order, ceil
            ceil = ac.max(ceil, h)
            start = order
            order += 1
            while len(dct) < h + 1:
                dct.append(defaultdict(list))
            dct[h][s[x]].append(order - 1)
            for y in edge[x]:
                yield dfs(y, h + 1)
            interval[x] = [start, order - 1]
            yield

        # 高度与深搜区间
        order = 0
        ceil = 0
        # 存储字符对应的高度以及dfs_order|
        dct = []
        interval = [[] for _ in range(n)]
        dfs(0, 1)
        del s
        del edge

        for _ in range(m):
            v, he = ac.read_list_ints_minus_one()
            he += 1
            if he > ceil:
                ac.st("Yes")
                continue
            low, high = interval[v]
            odd = 0
            for w in dct[he]:
                cur = bisect.bisect_right(dct[he][w], high) - bisect.bisect_left(dct[he][w], low)
                odd += cur % 2
                if odd >= 2:
                    break
            ac.st("Yes" if odd <= 1 else "No")
        return

    @staticmethod
    def cf_570d_2(ac=FastIO()):
        # 迭代顺序实现dfs_order，利用异或和来判断是否能形成回文

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

        # 迭代方式同时更新深度与状态
        stack = [[0, 1, 0]]
        while stack:
            i, state, height = stack.pop()
            if state:
                for h, j in queries[i]:
                    ans[j] ^= depth[h]
                depth[height] ^= 1 << (ord(s[i]) - ord("a"))
                stack.append([i, 0, height])
                for j in edge[i]:
                    stack.append([j, 1, height + 1])
            else:
                for h, j in queries[i]:
                    ans[j] ^= depth[h]
        for a in ans:
            ac.st("Yes" if a & (a - 1) == 0 else "No")
        return

    @staticmethod
    def cf_208e(ac=FastIO()):
        # dfs_order|LCA|binary_searchcounter
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
        # 深度优先搜索与back_track
        n, k = ac.read_list_ints()
        a = ac.read_list_ints()
        b = ac.read_list_ints()

        def dfs(i):
            nonlocal ans, pre
            if ans:
                return
            if i == k:
                ans = pre[:]
                return
            for j in range(n):
                if a[j] >= b[i]:
                    x = a[j]
                    a[j] = -1
                    pre.append(j + 1)
                    dfs(i + 1)
                    a[j] = x
                    pre.pop()
            return

        pre = []
        ans = []
        dfs(0)
        if not ans:
            ans = [-1]
        ac.lst(ans)
        return

    @staticmethod
    def ac_4310(ac=FastIO()):
        # dfs_order模板题
        n, q = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        nums = ac.read_list_ints_minus_one()
        for i in range(n - 1):
            dct[nums[i]].append(i + 1)
        for i in range(n):
            # 注意遍历顺序
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
        # 欧拉序在线查找树上距离，结合binary_search与prefix_sum变化情况
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
        # 最近公共祖先
        tree = TreeAncestor(edges)
        # 初始距离
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
        # 欧拉序
        euler_order = DfsEulerOrder(edges).euler_order[:]
        m = len(euler_order)
        euler_ind = [-1] * n
        # 预处理欧拉序所经过的路径prefix_sum
        color_pos_ind = defaultdict(list)  # 从上往下
        color_neg_ind = defaultdict(list)  # 从下往上
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
                # 欧拉序维护距离的变化
                start, end = euler_ind[0] + 1, euler_ind[w]
                pos_range = [bisect_left(color_pos_ind[x], start), bisect_right(color_pos_ind[x], end)]
                neg_range = [bisect_left(color_neg_ind[x], start), bisect_right(color_neg_ind[x], end)]

                # 颜色对应的路径和
                pre_color = color_pos_pre[x][pos_range[1]] - color_pos_pre[x][pos_range[0]]
                pre_color -= color_neg_pre[x][neg_range[1]] - color_neg_pre[x][neg_range[0]]
                # 变更代价后的路径和
                post_color_cnt = pos_range[1] - pos_range[0] - (neg_range[1] - neg_range[0])
                cur_dis[w] = dis[w] - pre_color + post_color_cnt * y
            # 最近公共祖先距离任意两点之间的距离
            ac.st(cur_dis[u] + cur_dis[v] - 2 * cur_dis[ancestor])
        return

    @staticmethod
    def ac_23(matrix, string):
        # back_track模板题
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