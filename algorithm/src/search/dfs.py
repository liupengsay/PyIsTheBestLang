
import unittest
import bisect
from collections import defaultdict, deque

from typing import List, Optional

from algorithm.src.fast_io import FastIO
from algorithm.src.graph.lca import TreeAncestor

"""

算法：深度优先搜索、染色法
功能：常与回溯枚举结合使用，比较经典的还有DFS序
题目：

===================================力扣===================================
473. 火柴拼正方形（https://leetcode.cn/problems/matchsticks-to-square/）暴力搜索木棍拼接组成正方形
301. 删除无效的括号（https://leetcode.cn/problems/remove-invalid-parentheses/）深搜回溯与剪枝
2581. 统计可能的树根数目（https://leetcode.cn/contest/biweekly-contest-99/problems/count-number-of-possible-root-nodes/）深搜序加差分计数


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

================================CodeForces================================
D. Tree Requests（https://codeforces.com/contest/570/problem/D）dfs序与二分查找，也可以使用离线查询
E. Blood Cousins（https://codeforces.com/contest/208/problem/E）深搜序加LCA加二分查找计数

参考：OI WiKi（xx）
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Order:
    def __init__(self):
        # 二叉树或者N叉树的前、中、后序写法
        return

    @staticmethod
    def post_order(root: Optional[TreeNode]) -> List[int]:
        ans = []
        stack = [[root, 1]] if root else []
        while stack:
            node, state = stack.pop()
            if state:
                # 后序
                stack.append([node, 0])
                if node.right:
                    stack.append([node.right, 1])
                if node.left:
                    stack.append([node.left, 1])
            else:
                ans.append(node.val)
        return ans

    @staticmethod
    def pre_order(root: Optional[TreeNode]) -> List[int]:
        ans = []
        stack = [[root, 1]] if root else []
        while stack:
            node, state = stack.pop()
            if state:
                if node.right:
                    stack.append([node.right, 1])
                if node.left:
                    stack.append([node.left, 1])
                # 前序
                stack.append([node, 0])
            else:
                ans.append(node.val)
        return ans

    @staticmethod
    def in_order(root: Optional[TreeNode]) -> List[int]:
        ans = []
        stack = [[root, 1]] if root else []
        while stack:
            node, state = stack.pop()
            if state:
                if node.right:
                    stack.append([node.right, 1])
                # 中序
                stack.append([node, 0])
                if node.left:
                    stack.append([node.left, 1])
            else:
                ans.append(node.val)
        return ans


class DFS:
    def __init__(self):
        return

    @staticmethod
    def gen_dfs_order_recursion(dct):
        # 模板：生成深搜序即 dfs 序以及对应子树编号区间
        def dfs(x):
            nonlocal order
            start[x] = order
            order += 1
            for y in dct[x]:
                if start[y] == -1:
                    dfs(y)
            end[x] = order - 1
            return

        n = len(dct)
        order = 0
        start = [-1] * n
        end = [-1]*n
        dfs(0)
        return start, end

    @staticmethod
    def gen_dfs_order_iteration(dct):
        # 模板：生成深搜序即 dfs 序以及对应子树编号区间
        n = len(dct)
        order = 0
        start = [-1] * n
        end = [-1]*n
        parent = [-1]*n
        stack = [[0, 1, -1, 0]]
        depth = [0]*n
        while stack:
            i, state, fa, d = stack.pop()
            if state:
                start[i] = order
                end[i] = order
                depth[i] = d
                order += 1
                stack.append([i, 0, fa, d])
                for j in dct[i]:
                    if j != fa:
                        parent[j] = i
                        stack.append([j, 1, i, d+1])
            else:
                if parent[i] != -1:
                    end[parent[i]] = end[i]

        return start, end


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
        n, m = ac.read_ints()
        dct = [[] for _ in range(n + 1)]
        degree = [0] * (n + 1)
        for _ in range(m):
            x, y = ac.read_ints()
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
    def lc_2581(edges: List[List[int]], guesses: List[List[int]], k: int) -> int:
        # 模板：使用深搜序确定猜测的查询范围，并使用差分数组计数
        n = len(edges) + 1
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        visit, interval = DFS().gen_dfs_order_recursion(dct)
        # 也可以使用 DFS().gen_dfs_order_iteration(dct)
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
            v, he = ac.read_ints_minus_one()
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

        n, m = ac.read_ints()
        parent = ac.read_list_ints_minus_one()
        edge = [[] for _ in range(n)]
        for i in range(n - 1):
            edge[parent[i]].append(i + 1)
        s = ac.read_str()

        queries = [[] for _ in range(n)]
        for i in range(m):
            v, h = ac.read_ints()
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
        start, end = DFS().gen_dfs_order_iteration(edge)

        dct = [[] for _ in range(n + 1)]
        for i in range(n + 1):
            dct[tree.depth[i]].append(start[i])
        for i in range(n + 1):
            dct[i].sort()

        ans = []
        for _ in range(ac.read_int()):
            v, p = ac.read_ints()
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


class TestGeneral(unittest.TestCase):

    def test_dfs(self):
        dfs = DFS()
        dct = [[1, 2], [0, 3], [0, 4], [1], [2]]
        start, end = dfs.gen_dfs_order_recursion(dct)
        assert start == [x-1 for x in [1, 2, 4, 3, 5]]
        assert end == [b-1 for _, b in [[1, 5], [2, 3], [4, 5], [3, 3], [5, 5]]]

        dfs = DFS()
        dct = [[1, 2], [0, 3], [0, 4], [1], [2]]
        start, end = dfs.gen_dfs_order_iteration([d[::-1] for d in dct])
        assert start == [x - 1 for x in [1, 2, 4, 3, 5]]
        assert end == [b-1 for _, b in [[1, 5], [2, 3], [4, 5], [3, 3], [5, 5]]]
        return


if __name__ == '__main__':
    unittest.main()
