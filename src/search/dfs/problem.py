"""

算法：深度优先搜索、染色法、枚举回溯、欧拉序、dfs序
功能：常与回溯枚举结合使用，比较经典的还有DFS序
题目：欧拉序是在dfs序的基础上增加了边的回溯，可以使用区间修改来在线维护树上距离

参考：基于欧拉序的维护树上距离的在线算法（https://zhuanlan.zhihu.com/p/84236967）

===================================力扣===================================
473. 火柴拼正方形（https://leetcode.cn/problems/matchsticks-to-square/）暴力搜索木棍拼接组成正方形
301. 删除无效的括号（https://leetcode.cn/problems/remove-invalid-parentheses/）深搜回溯与剪枝
2581. 统计可能的树根数目（https://leetcode.cn/contest/biweekly-contest-99/problems/count-number-of-possible-root-nodes/）深搜序加差分计数
1059. 从始点到终点的所有路径（https://leetcode.cn/problems/all-paths-from-source-lead-to-destination/）记忆化搜索DFS深搜且回溯
1718. 构建字典序最大的可行序列（https://leetcode.cn/problems/construct-the-lexicographically-largest-valid-sequence/）经典回溯
2065. 最大化一张图中的路径价值（https://leetcode.cn/problems/maximum-path-quality-of-a-graph/）经典回溯，正解使用Dijkstra跑最短路剪枝
2322. 从树中删除边的最小分数（https://leetcode.cn/problems/minimum-score-after-removals-on-a-tree/）使用深搜序dfs序枚举
1240. 铺瓷砖（https://leetcode.cn/problems/tiling-a-rectangle-with-the-fewest-squares/）经典DFS回溯与剪枝
1239. 串联字符串的最大长度（https://leetcode.cn/problems/maximum-length-of-a-concatenated-string-with-unique-characters/）经典DFS回溯进行二进制枚举
1080. 根到叶路径上的不足节点（https://leetcode.cn/problems/insufficient-nodes-in-root-to-leaf-paths/description/）经典dfs自上而下后又自下而上
2056. 棋盘上有效移动组合的数目（https://leetcode.cn/problems/number-of-valid-move-combinations-on-chessboard/description/）经典回溯枚举
100041. 可以到达每一个节点的最少边反转次数（https://www.acwing.com/problem/content/description/4384/）迭代法实现树形换根DP计算，或者一遍DFS或者dfs序加差分
2458. 移除子树后的二叉树高度（https://leetcode.cn/contest/weekly-contest-317/problems/height-of-binary-tree-after-subtree-removal-queries/）dfs序模板题目

===================================洛谷===================================
P2383 狗哥玩木棒（https://www.luogu.com.cn/problem/P2383）暴力搜索木棍拼接组成正方形
P1120 小木棍（https://www.luogu.com.cn/problem/P1120）把数组分成和相等的子数组
P1692 部落卫队（https://www.luogu.com.cn/problem/P1692）暴力搜索枚举字典序最大可行的连通块
P1612 [yLOI2018] 树上的链（https://www.luogu.com.cn/problem/P1612）使用dfs记录路径的前缀和并使用二分确定最长链条
P1475 [USACO2.3]控制公司 Controlling Companies（https://www.luogu.com.cn/problem/P1475）深搜确定可以控制的公司对
P2080 增进感情（https://www.luogu.com.cn/problem/P2080）深搜回溯与剪枝
P2090 数字对（https://www.luogu.com.cn/problem/P2090）深搜贪心回溯剪枝与辗转相减法
P2420 让我们异或吧（https://www.luogu.com.cn/problem/P2420）脑筋急转弯使用深搜确定到根路径的异或结果以及异或特性获得任意两点之间最短路的异或结果
P1473 [USACO2.3]零的数列 Zero Sum（https://www.luogu.com.cn/problem/P1473）深搜枚举符号数
P1461 [USACO2.1]海明码 Hamming Codes（https://www.luogu.com.cn/problem/P1461）汉明距离计算与深搜回溯枚举
P1394 山上的国度（https://www.luogu.com.cn/problem/P1394）深搜进行可达性确认
P1180 驾车旅游（https://www.luogu.com.cn/problem/P1180）深搜进行模拟
P1118 [USACO06FEB]Backward Digit Sums G/S（https://www.luogu.com.cn/problem/P1118）使用单位矩阵模拟计算杨辉三角的系数，再进行暴搜寻找最小字典序结果
P3252 [JLOI2012]树（https://www.luogu.com.cn/problem/P3252）深搜回溯加前缀和哈希
P4913 【深基16.例3】二叉树深度（https://www.luogu.com.cn/problem/P4913）深搜确定深度
P5118 [USACO18DEC]Back and Forth B（https://www.luogu.com.cn/problem/P5118）深搜回溯与哈希记录进行模拟
P5197 [USACO19JAN]Grass Planting S（https://www.luogu.com.cn/problem/P5197）树形DP模拟与染色法，利用父亲与自己的染色确定儿子们的染色
P5198 [USACO19JAN]Icy Perimeter S（https://www.luogu.com.cn/problem/P5198）经典计算连通块的周长与面积
P5318 【深基18.例3】查找文献（https://www.luogu.com.cn/problem/P5318）经典广搜拓扑排序与深搜序生成与获取
P6691 选择题（https://www.luogu.com.cn/problem/P6691）染色法，进行二分图可行性方案计数与最大最小染色
P7370 [COCI2018-2019#4] Wand（https://www.luogu.com.cn/problem/P7370）所有可能的祖先节点，注意特别情况没有任何祖先节点则自身可达
P1036 [NOIP2002 普及组] 选数（https://www.luogu.com.cn/problem/P1036）回溯剪枝
P8578 [CoE R5] So What Do We Do Now?（https://www.luogu.com.cn/problem/P8578）贪心使用深搜序
P8838 [传智杯 #3 决赛] 面试（https://www.luogu.com.cn/problem/P8838）深度优先搜索与回溯


================================CodeForces================================
D. Tree Requests（https://codeforces.com/contest/570/problem/D）dfs序与二分查找，也可以使用离线查询
E. Blood Cousins（https://codeforces.com/contest/208/problem/E）深搜序加LCA加二分查找计数
D. Choosing Capital for Treeland（https://codeforces.com/contest/219/problem/D）迭代法实现树形换根DP计算，或者一遍DFS或者dfs序加差分
E. Military Problem（https://codeforces.com/contest/1006/problem/E）经典dfs序模板题
G2. Passable Paths (hard version)（https://codeforces.com/contest/1702/problem/G2）使用dfs序与lca组合判断是否为简单路径集合

================================AtCoder================================
F - Colorful Tree（https://atcoder.jp/contests/abc133/tasks/abc133_f）欧拉序在线查找树上距离，结合二分与前缀和计算变化情况

================================AcWing================================
4310. 树的DFS（https://www.acwing.com/problem/content/4313/）经典深搜序模板题
23. 矩阵中的路径（https://www.acwing.com/problem/content/description/21/）回溯模板题

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
from src.graph.tree_lca.template import TreeAncestor
from src.search.dfs.template import DFS, DfsEulerOrder
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_473(matchsticks: List[int]) -> bool:
        # 模板: 深搜加回溯判断能否将数组分成正方形
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
        # 模板：迭代法实现树形换根DP计算，或者一遍DFS或者dfs序加差分
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
        # 模板：经典dfs序模板题
        n, q = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        p = ac.read_list_ints_minus_one()
        for i in range(n-1):
            dct[p[i]].append(i+1)
        for i in range(n):
            dct[i].reverse()
        dfs = DfsEulerOrder(dct)
        for _ in range(q):
            u, k = ac.read_list_ints()
            u -= 1
            x = dfs.start[u]
            if n-x < k or dfs.start[dfs.order_to_node[x+k-1]] > dfs.end[u]:
                ac.st(-1)
            else:
                ac.st(dfs.order_to_node[x+k-1] + 1)
        return

    @staticmethod
    def lc_301(s):
        # 模板：深搜回溯计算删除最少数量的无效括号使得子串合法有效

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
        # 模板：深搜与广搜序获取
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n + 1)]
        degree = [0] * (n + 1)
        for _ in range(m):
            x, y = ac.read_list_ints()
            dct[x].append(y)
            degree[y] += 1
        for i in range(1, n + 1):
            dct[i].sort()

        # 深搜序值获取
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

        # 拓扑排序广搜
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

        # 计算将 [1, 1] 通过 [a, b] 到 [a, a+b] 或者 [a+b, a] 的方式最少次数变成 a == n or b == n
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

            # 逆向思维计算保证使 b 减少到 a 以下
            gcd_minus(b, a % b, c + a // b)
            return

        ans = n - 1
        for i in range(1, n):
            gcd_minus(n, i, 0)
        return ans

    @staticmethod
    def lc_1080(root: Optional[TreeNode], limit: int) -> Optional[TreeNode]:

        # 模板：经典dfs自上而下后又自下而上
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
        # 模板：经典DFS回溯进行二进制枚举
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
                dfs(i+1)
                pre -= set(arr[i])
            dfs(i+1)
            return

        pre = set()
        dfs(0)
        return ans

    @staticmethod
    def lc_1240(n: int, m: int) -> int:
        # 模板：经典DFS回溯与剪枝

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
                        # 枚举此时左上端点正方形的长度
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
        # 模板：经典回溯枚举
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
        # 模板：使用深搜序dfs序枚举
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
        # 枚举边对
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
        # 模板：dfs序模板题目

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
        # 模板：使用深搜序确定猜测的查询范围，并使用差分数组计数
        n = len(edges) + 1
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        visit, interval = DFS().gen_bfs_order_iteration(dct)
        # 也可以使用
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
        # 模板：迭代法实现树形换根DP计算，或者一遍DFS或者dfs序加差分
        n = ac.read_int()
        edges = [ac.read_list_ints_minus_one() for _ in range(n-1)]

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
        # 模板：使用dfs序与二分进行计数统计（超时）
        n, m = ac.read_list_ints()
        parent = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        for i in range(n - 1):
            edge[parent[i] - 1].append(i + 1)
        del parent
        s = ac.read_str()

        # 模板：生成深搜序即 dfs 序以及对应子树编号区间
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

        # 计算高度与深搜区间
        order = 0
        ceil = 0
        # 存储字符对应的高度以及dfs序
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
        # 模板：使用迭代顺序实现深搜序，利用异或和来判断是否能形成回文

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
        # 模板：深搜序加LCA加二分查找计数
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
        # 模板：深度优先搜索与回溯
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
        # 模板：经典深搜序模板题
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
        # 模板：欧拉序在线查找树上距离，结合二分与前缀和计算变化情况
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
        # 预处理欧拉序所经过的路径前缀和
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
                # 使用欧拉序维护距离的变化
                start, end = euler_ind[0] + 1, euler_ind[w]
                pos_range = [bisect_left(color_pos_ind[x], start), bisect_right(color_pos_ind[x], end)]
                neg_range = [bisect_left(color_neg_ind[x], start), bisect_right(color_neg_ind[x], end)]

                # 颜色对应的路径和
                pre_color = color_pos_pre[x][pos_range[1]] - color_pos_pre[x][pos_range[0]]
                pre_color -= color_neg_pre[x][neg_range[1]] - color_neg_pre[x][neg_range[0]]
                # 变更代价后的路径和
                post_color_cnt = pos_range[1] - pos_range[0] - (neg_range[1] - neg_range[0])
                cur_dis[w] = dis[w] - pre_color + post_color_cnt * y
            # 使用最近公共祖先距离计算任意两点之间的距离
            ac.st(cur_dis[u] + cur_dis[v] - 2 * cur_dis[ancestor])
        return

    @staticmethod
    def ac_23(matrix, string):
        # 模板：回溯模板题
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
