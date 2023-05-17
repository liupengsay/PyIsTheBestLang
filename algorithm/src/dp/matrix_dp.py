import unittest
from bisect import bisect_left
from collections import defaultdict
from math import inf

from typing import List
from types import GeneratorType
from algorithm.src.fast_io import FastIO


"""
算法：矩阵DP、二维DP、记忆化搜索（记忆化形式的DP，可以自顶向下也可以自底向上，就是另一种写法的DP）
功能：在二维矩阵上进行转移的DP，经典的有矩阵前缀和，矩阵区间和，正方形最大边长或面积，编辑距离，公共子序列，最长回文子串
题目：

===================================力扣===================================
2478. 完美分割的方案数（https://leetcode.cn/problems/number-of-beautiful-partitions/）
2463. 最小移动总距离（https://leetcode.cn/problems/minimum-total-distance-traveled/）
2435. 矩阵中和能被 K 整除的路径（https://leetcode.cn/problems/paths-in-matrix-whose-sum-is-divisible-by-k/）利用模 K 的特点进行路径计算
2088. 统计农场中肥沃金字塔的数目（https://leetcode.cn/problems/count-fertile-pyramids-in-a-land/）类似求正方形的边长和面积进行矩阵DP
221. 最大正方形（https://leetcode.cn/problems/maximal-square/）求全为 1 的最大正方形面积
72. 编辑距离（https://leetcode.cn/problems/edit-distance/）矩阵DP
329. 矩阵中的最长递增路径（https://leetcode.cn/problems/longest-increasing-path-in-a-matrix/）二维矩阵DP
1478. 安排邮筒（https://leetcode.cn/problems/allocate-mailboxes/）二维DP与一个计算不带权中位数距离的区间DP
6363. 找出对应 LCP 矩阵的字符串（https://leetcode.cn/problems/find-the-string-with-lcp/）贪心构造符合条件的字符串，并通过计算LCP进行确认
2328. 网格图中递增路径的数目（https://leetcode.cn/problems/number-of-increasing-paths-in-a-grid/）计算严格递增的路径数量
2312. 卖木头块（https://leetcode.cn/problems/selling-pieces-of-wood/）自顶向下搜索最佳方案
2267. 检查是否有合法括号字符串路径（https://leetcode.cn/problems/check-if-there-is-a-valid-parentheses-string-path/）记忆化搜索合法路径
1092. 最短公共超序列（https://leetcode.cn/problems/shortest-common-supersequence/）经典从后往前动态规划加从前往后构造，计算最长公共子序列，并构造包含两个字符串的最短公共超序列
1143. 最长公共子序列（https://leetcode.cn/problems/longest-common-subsequence/）使用LIS的方法求LCS
1035. 不相交的线（https://leetcode.cn/problems/uncrossed-lines/）使用LIS的方法求LCS

===================================洛谷===================================
P2701 [USACO5.3]巨大的牛棚Big Barn（https://www.luogu.com.cn/problem/P2701）求全为 "." 的最大正方形面积，如果不要求实心只能做到O(n^3)复杂度
P2049 魔术棋子（https://www.luogu.com.cn/problem/P2049）求左上角到右下角所有路径的乘积取模数
P2138 小Z的关系距离（https://www.luogu.com.cn/problem/P2138）最长公共子序列
P1681 最大正方形II（https://www.luogu.com.cn/problem/P1681）求黑白格子相间的最大正方形面积
P2268 [HNOI2002]DNA分子的最佳比对（https://www.luogu.com.cn/problem/P2268）类似编辑距离
P2301 就是干！（https://www.luogu.com.cn/problem/P2301）矩阵DP，注意最小值的更新处理
P2364 胖男孩（https://www.luogu.com.cn/problem/P2364）三维DP求最长公共子序列LCS并且输出LCS
P2543 [AHOI2004]奇怪的字符串（https://www.luogu.com.cn/problem/P2543）二维DP求最长公共子序列LCS长度
P2513 [HAOI2009]逆序对数列（https://www.luogu.com.cn/record/list?user=739032&status=12&page=2）二维矩阵DP加前缀和优化
P1434 [SHOI2002] 滑雪（https://www.luogu.com.cn/problem/P1434）二维矩阵DP计算最长上升的路径
P1140 相似基因（https://www.luogu.com.cn/problem/P1140）二维矩阵DP
P1057 [NOIP2008 普及组] 传球游戏（https://www.luogu.com.cn/problem/P1057）二维DP可做成转移的
P8825 [传智杯 #3 初赛] 运气（https://www.luogu.com.cn/problem/P8825）结合取模进行滚动更新计算
P2758 编辑距离（https://www.luogu.com.cn/problem/P2758）二维DP编辑距离
P2803 学校选址 II（https://www.luogu.com.cn/problem/P2803）二维DP与一个计算带权中位数距离的区间DP
P2946 [USACO09MAR]Cow Frisbee Team S（https://www.luogu.com.cn/problem/P2946）计算何为某个数字倍数的连续子序列个数
P2427 Wave（https://www.luogu.com.cn/problem/P2427）以矩阵中点为正方形中心的最大正方形边长，使用左上、左下、右上和右下的四个DP
P7074 [CSP-J2020] 方格取数（https://www.luogu.com.cn/problem/P7074）经典DP，三个方向进行转移更新
P7160 「dWoi R1」Sixth Monokuma's Son（https://www.luogu.com.cn/problem/P7160）三个维度DP的枚举计数
P7266 [BalticOI 2000] Honeycomb Problem（https://www.luogu.com.cn/problem/P7266）蜂窝形状的矩阵DP
P3399 丝绸之路（https://www.luogu.com.cn/problem/P3399）二维矩阵DP
P2516 [HAOI2010]最长公共子序列（https://www.luogu.com.cn/problem/P2516）经典DP最长公共子序列以及最长公共子序列的长度
P1544 三倍经验（https://www.luogu.com.cn/problem/P1544）三维矩阵DP
P1004 [NOIP2000 提高组] 方格取数（https://www.luogu.com.cn/problem/P1004）经典DP，三个方向转移
P1006 [NOIP2008 提高组] 传纸条（https://www.luogu.com.cn/problem/P1006）经典DP，三个方向转移
P1107 [BJWC2008]雷涛的小猫（https://www.luogu.com.cn/problem/P1107）二维DP加前缀最值优化
P1279 字串距离（https://www.luogu.com.cn/problem/P1279）经典编辑距离DP的变形
P1353 [USACO08JAN]Running S（https://www.luogu.com.cn/problem/P1353）矩阵DP
P1410 子序列（https://www.luogu.com.cn/problem/P1410）二维DP
P1799 数列（https://www.luogu.com.cn/problem/P1799）矩阵二维DP
P1854 花店橱窗布置（https://www.luogu.com.cn/problem/P1854）矩阵DP，并输出匹配方案
P2140 小Z的电力管制（https://www.luogu.com.cn/problem/P2140）矩阵四维DP，可以使用记忆化与迭代计算
P2217 [HAOI2007]分割矩阵（https://www.luogu.com.cn/problem/P2217）矩阵四维DP，可以使用记忆化与迭代计算
P1436 棋盘分割（https://www.luogu.com.cn/problem/P1436）矩阵四维DP，可以使用记忆化与迭代计算
P5752 [NOI1999] 棋盘分割（https://www.luogu.com.cn/problem/P5752）矩阵四维DP，可以使用记忆化与迭代计算
P2380 狗哥采矿（https://www.luogu.com.cn/problem/P2380）矩阵DP
P2401 不等数列（https://www.luogu.com.cn/problem/P2401）二维DP
P2528 [SHOI2001]排序工作量之新任务（https://www.luogu.com.cn/problem/P2528）逆序对矩阵 DP 与模拟构造
P2733 [USACO3.3]家的范围 Home on the Range（https://www.luogu.com.cn/problem/P2733）经典DP通过边长与差分数组计算正方形子矩阵的个数
P2736 [USACO3.4]“破锣摇滚”乐队 Raucous Rockers（https://www.luogu.com.cn/problem/P2736）矩阵DP
P2769 猴子上树（https://www.luogu.com.cn/problem/P2769）矩阵 DP 注意初始化条件


================================CodeForces================================
https://codeforces.com/problemset/problem/1446/B（最长公共子序列LCS变形问题，理解贡献）
https://codeforces.com/problemset/problem/429/B（四个方向的矩阵DP）
D. Colored Rectangles（https://codeforces.com/problemset/problem/1398/D）三维DP，选取两个不同数组的数乘积，计算最大总和
B. The least round way（https://codeforces.com/problemset/problem/2/B）矩阵DP，计算路径上乘积最少的后缀0个数，经典题目
B. Unmerge（https://codeforces.com/problemset/problem/1381/B）二维矩阵DP加单调栈优化

参考：OI WiKi（xx）
"""


class LcsLis:
    def __init__(self):
        return

    def longest_common_subsequence(self, s1, s2) -> int:
        # 使用LIS的办法求LCS
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        m = len(s2)
        mapper = defaultdict(list)
        for i in range(m-1, -1, -1):
            mapper[s2[i]].append(i)
        nums = []
        for c in s1:
            if c in mapper:
                nums.extend(mapper[c])
        return self.longest_increasing_subsequence(nums)

    @staticmethod
    def longest_increasing_subsequence(nums: List[int]) -> int:
        # 使用贪心二分求LIS
        stack = []
        for x in nums:
            idx = bisect_left(stack, x)
            if idx < len(stack):
                stack[idx] = x
            else:
                stack.append(x)
        # 还可以返回stack获得最长公共子序列
        return len(stack)


class MatrixDP:
    def __init__(self):
        return

    @staticmethod
    def lcp(s, t):
        # 模板：最长公共前缀模板s[i:]和t[j:]
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if s[i] == t[j]:
                    dp[i][j] = dp[i + 1][j + 1] + 1
        return dp

    @staticmethod
    def min_distance(word1: str, word2: str):
        m, n = len(word1), len(word2)
        dp = [[float("inf")] * (n + 1) for _ in range(m + 1)]
        # 编辑距离注意起始开头的边界条件
        for i in range(m + 1):
            dp[i][n] = m - i
        for j in range(n + 1):
            dp[m][j] = n - j
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                # 删除，插入，替换
                dp[i][j] = min(dp[i + 1][j] + 1, dp[i][j + 1] + 1,
                               dp[i + 1][j + 1] + int(word1[i] != word2[j]))
        return dp[0][0]

    @staticmethod
    def path_mul_mod(m, n, k, grid):
        # 求矩阵左上角到右下角的乘积取模数
        dp = [[set() for _ in range(n)] for _ in range(m)]
        dp[0][0].add(grid[0][0] % k)
        for i in range(1, m):
            x = grid[i][0]
            for p in dp[i - 1][0]:
                dp[i][0].add((p * x) % k)
        for j in range(1, n):
            x = grid[0][j]
            for p in dp[0][j - 1]:
                dp[0][j].add((p * x) % k)

        for i in range(1, m):
            for j in range(1, n):
                x = grid[i][j]
                for p in dp[i][j - 1]:
                    dp[i][j].add((p * x) % k)
                for p in dp[i - 1][j]:
                    dp[i][j].add((p * x) % k)
        ans = sorted(list(dp[-1][-1]))
        return ans

    @staticmethod
    def maximal_square(matrix: List[List[str]]) -> int:

        # 求全为 1 的最大正方形面积
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        ans = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == "1":
                    # 转移公式
                    dp[i + 1][j + 1] = min(dp[i][j], dp[i + 1][j], dp[i][j + 1]) + 1
                    if dp[i + 1][j + 1] > ans:
                        ans = dp[i + 1][j + 1]
        return ans ** 2

    @staticmethod
    def longest_common_sequence(s1, s2, s3) -> str:
        # 模板: 最长公共子序列 LCS 可扩展到三维四维
        m, n, k = len(s1), len(s2), len(s3)
        # 记录 LCS 的长度
        dp = [[[0] * (k + 1) for _ in range(n + 1)] for _ in range(m + 1)]
        # 记录 LCS 的子串
        res = [[[""] * (k + 1) for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                for p in range(k):
                    if s1[i] == s2[j] == s3[p]:
                        if dp[i + 1][j + 1][p + 1] < dp[i][j][p] + 1:
                            dp[i + 1][j + 1][p + 1] = dp[i][j][p] + 1
                            res[i + 1][j + 1][p + 1] = res[i][j][p] + s1[i]
                    else:
                        for a, b, c in [[1, 1, 0], [0, 1, 1], [1, 0, 1]]:
                            if dp[i + 1][j + 1][p +
                                                1] < dp[i + a][j + b][p + c]:
                                dp[i + 1][j + 1][p + 1] = dp[i + a][j + b][p + c]
                                res[i + 1][j + 1][p + 1] = res[i + a][j + b][p + c]
        return res[m][n][k]


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1305(nums1: List[int], nums2: List[int]) -> int:
        # 模板：使用LIS的办法求LCS
        return LcsLis().longest_common_subsequence(nums1, nums2)

    @staticmethod
    def lc_1143(s1: str, s2: str) -> int:
        # 模板：使用LIS的办法求LCS
        return LcsLis().longest_common_subsequence(s1, s2)

    @staticmethod
    def lc_1092(str1: str, str2: str) -> str:
        # 模板：计算最长公共子序列，并构造包含两个字符串的最短公共超序列
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                a, b = dp[i + 1][j], dp[i][j + 1]
                a = a if a > b else b
                if str1[i] == str2[j]:
                    b = dp[i + 1][j + 1] + 1
                    a = a if a > b else b
                dp[i][j] = a

        i = j = 0
        ans = ""
        while i < m and j < n:
            if str1[i] == str2[j]:
                ans += str1[i]
                i += 1
                j += 1
            elif dp[i + 1][j + 1] == dp[i + 1][j]:
                ans += str2[j]
                j += 1
            else:
                ans += str1[i]
                i += 1
        ans += str1[i:] + str2[j:]
        return ans

    @staticmethod
    def lc_2435(grid: List[List[int]], k: int) -> int:
        # 模板：标准矩阵 DP 左上到右下的状态转移
        mod = 10 ** 9 + 7
        m, n = len(grid), len(grid[0])
        dp = [[[0] * k for _ in range(n)] for _ in range(m)]
        dp[0][0][grid[0][0] % k] = 1

        pre = grid[0][0]
        for j in range(1, n):
            pre += grid[0][j]
            dp[0][j][pre % k] = 1

        pre = grid[0][0]
        for i in range(1, m):
            pre += grid[i][0]
            dp[i][0][pre % k] = 1

        for i in range(1, m):
            for j in range(1, n):
                for x in range(k):
                    y = (x - grid[i][j]) % k
                    dp[i][j][x] = (dp[i - 1][j][y] + dp[i][j - 1][y]) % mod
        return dp[-1][-1][0]

    @staticmethod
    def lc_6363(lcp: List[List[int]]) -> str:
        # 模板：根据 LCP 矩阵生成字典序最小的符合条件的字符串
        n = len(lcp)
        ans = [""] * n
        ind = 0
        for i in range(n):
            if ans[i]:
                continue
            if ind == 26:
                return ""
            w = chr(ind + ord("a"))
            ans[i] = w
            ind += 1
            for j in range(i + 1, n):
                if lcp[i][j]:
                    ans[j] = w

        ans = "".join(ans)
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if ans[i] == ans[j]:
                    if i + 1 < n and j + 1 < n:
                        x = lcp[i + 1][j + 1] + 1
                    else:
                        x = 1
                else:
                    x = 0
                if x != lcp[i][j]:
                    return ""
        return ans

    @staticmethod
    def cf_2b(ac, n, grid):
        # 模板：计算乘积后缀0最少的个数以及对应的路径
        def f_2(num):
            if not num:
                return 1
            res = 0
            while num and num % 2 == 0:
                num //= 2
                res += 1
            return res

        def f_5(num):
            if not num:
                return 1
            res = 0
            while num and num % 5 == 0:
                num //= 5
                res += 1
            return res

        def check(fun):
            dp = [[inf] * n for _ in range(n)]
            dp[0][0] = fun(grid[0][0])
            f = [[-1] * n for _ in range(n)]
            for j in range(1, n):
                f[0][j] = j - 1
                dp[0][j] = dp[0][j - 1] + fun(grid[0][j]) if grid[0][j] else 1
            for i in range(1, n):
                f[i][0] = (i - 1) * n
                dp[i][0] = dp[i - 1][0] + fun(grid[i][0]) if grid[i][0] else 1
                for j in range(1, n):
                    if grid[i][j] == 0:
                        dp[i][j] = 1
                    else:
                        c = fun(grid[i][j])
                        dp[i][j] = ac.min(dp[i - 1][j], dp[i][j - 1]) + c
                    f[i][j] = (i - 1) * n + j if dp[i - 1][j] < dp[i][j - 1] else i * n + j - 1
            cnt = dp[-1][-1]
            path = ""
            x = (n - 1) * n + n - 1
            while f[x // n][x % n] != -1:
                i, j = x // n, x % n
                p = f[i][j]
                if i == p // n:
                    path += "R"
                else:
                    path += "D"
                x = p
            return cnt, path[::-1]

        inf = float("inf")
        c1, path1 = check(f_2)
        c2, path2 = check(f_5)
        if c1 <= c2:
            ans = [c1, path1]
        else:
            ans = [c2, path2]

        # 考虑 0 的存在影响
        zero = False
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 0:
                    zero = True
        if not zero:
            return ans

        if ans[0] > 1:
            for i in range(n):
                for j in range(n):
                    if grid[i][j] == 0:
                        cur = "D" * i + "R" * j + "D" * (n - 1 - i) + "R" * (n - 1 - j)
                        ans = [1, cur]
                        return ans
        return ans

    @staticmethod
    def cf_1398d(ac, r, g, b, lst):
        # 模板：三维DP，选取两个不同数组的数乘积，计算最大总和
        @ac.bootstrap
        def dfs(i, j, k):
            if dp[i][j][k] != -1:
                yield
            res = 0
            if i < r and j < g:
                yield dfs(i + 1, j + 1, k)
                res = ac.max(res, dp[i + 1][j + 1][k] + lst[0][i] * lst[1][j])
            if i < r and k < b:
                yield dfs(i + 1, j, k + 1)
                res = ac.max(res, dp[i + 1][j][k + 1] + lst[0][i] * lst[2][k])
            if j < g and k < b:
                yield dfs(i, j + 1, k + 1)
                res = ac.max(res, dp[i][j + 1][k + 1] + lst[2][k] * lst[1][j])
            dp[i][j][k] = res
            yield

        dp = [[[-1] * (b + 1) for _ in range(g + 1)] for _ in range(r + 1)]
        dfs(0, 0, 0)
        return dp[0][0][0]

    @staticmethod
    def lc_2478(s: str, k: int, min_length: int) -> int:
        mod = 10**9 + 7
        # 模板：前缀和优化二维矩阵DP
        start = set("2357")
        if s[0] not in start:
            return 0
        n = len(s)
        dp = [[0] * n for _ in range(k)]
        for i in range(n):
            if i + 1 >= min_length and s[i] not in start:
                dp[0][i] = 1

        for j in range(1, k):
            pre = 0
            x = 0
            for i in range(n):
                while x <= i - min_length and s[x]:
                    if s[x] not in start and s[x + 1] in start:
                        pre += dp[j - 1][x]
                        pre %= mod
                    x += 1
                if s[i] not in start:
                    dp[j][i] = pre
        return dp[-1][-1]

    @staticmethod
    def lc_2463(robot, factory):
        # 模板：两个数组使用指针移动方向与前缀和优化求解
        robot.sort()
        factory.sort()
        m, n = len(factory), len(robot)
        dp = [[float("inf")] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        for i in range(m):
            for j in range(n + 1):
                if dp[i][j] < float("inf"):
                    dp[i + 1][j] = min(dp[i + 1][j], dp[i][j])
                    cost = 0
                    for k in range(1, factory[i][1] + 1):
                        if j + k - 1 < n:
                            cost += abs(factory[i][0] - robot[j + k - 1])
                            dp[i + 1][j + k] = min(dp[i + 1][j + k], dp[i][j] + cost)
                        else:
                            break
        return dp[-1][-1]

    @staticmethod
    def lg_p2516(ac=FastIO()):
        # 模板：最长公共子序列的长度以及个数DP计算
        s = ac.read_str()[:-1]
        t = ac.read_str()[:-1]
        m, n = len(s), len(t)
        mod = 10 ** 8
        # 使用滚动数组进行优化
        dp = [[0] * (n + 1) for _ in range(2)]
        cnt = [[0] * (n + 1) for _ in range(2)]
        pre = 0
        for j in range(n + 1):
            cnt[pre][j] = 1
        for i in range(m):
            cur = 1 - pre
            dp[cur][0] = 0
            cnt[cur][0] = 1
            for j in range(n):
                dp[cur][j + 1] = 0
                cnt[cur][j + 1] = 0
                # 长度更长
                if s[i] == t[j]:
                    dp[cur][j + 1] = dp[pre][j] + 1
                    cnt[cur][j + 1] = cnt[pre][j]

                # 左面的转移
                if dp[cur][j] > dp[cur][j + 1]:
                    dp[cur][j + 1] = dp[cur][j]
                    cnt[cur][j + 1] = cnt[cur][j]
                elif dp[cur][j] == dp[cur][j + 1]:
                    cnt[cur][j + 1] += cnt[cur][j]

                # 上面的转移
                if dp[pre][j + 1] > dp[cur][j + 1]:
                    dp[cur][j + 1] = dp[pre][j + 1]
                    cnt[cur][j + 1] = cnt[pre][j + 1]
                elif dp[pre][j + 1] == dp[cur][j + 1]:
                    cnt[cur][j + 1] += cnt[pre][j + 1]

                # 长度未变则设计重复计算
                if dp[pre][j] == dp[cur][j + 1]:
                    cnt[cur][j + 1] -= cnt[pre][j]
                cnt[cur][j + 1] %= mod
            pre = cur

        ac.st(dp[pre][-1])
        ac.st(cnt[pre][-1])
        return

    @staticmethod
    def lg_p1544(ac=FastIO()):
        # 模板：三维矩阵DP
        n, k = ac.read_ints()
        dp = [[[-inf]*(k+1) for _ in range(n)] for _ in range(2)]
        nums = []
        while len(nums) < n*(n+1)//2:
            nums.extend(ac.read_list_ints())

        pre = 0
        num = nums[0]
        dp[pre][0][0] = num
        dp[pre][0][1] = num*3
        s = 1
        for i in range(1, n):
            lst = nums[s:s+i+1]
            s += i+1
            cur = 1-pre
            dp[cur] = [[-inf]*(k+1) for _ in range(n)]
            for j in range(i+1):
                for p in range(k+1):
                    if j and p:
                        a = ac.max(dp[pre][j][p], dp[pre][j-1][p]) + lst[j]
                        b = ac.max(dp[pre][j][p-1], dp[pre][j-1][p-1]) + lst[j]*3
                        dp[cur][j][p] = ac.max(a, b)
                    elif j:
                        dp[cur][j][p] = ac.max(dp[pre][j][p], dp[pre][j-1][p]) + lst[j]
                    elif p:
                        dp[cur][j][p] = ac.max(dp[pre][j][p]+lst[j], dp[pre][j][p-1]+lst[j]*3)
                    else:
                        dp[cur][j][p] = dp[pre][j][p]+lst[j]
            pre = cur
        ac.st(max(max(d) for d in dp[pre]))
        return

    @staticmethod
    def lg_p1004(ac=FastIO()):
        # 模板：经典取数四维转三维DP，路径可以有交叠
        n = ac.read_int()
        grid = [[0] * n for _ in range(n)]
        while True:
            lst = ac.read_list_ints()
            if lst == [0, 0, 0]:
                break
            x, y, z = lst
            grid[x - 1][y - 1] = z

        dp = [[[0] * n for _ in range(n)] for _ in range(n)]
        for x1 in range(n - 1, -1, -1):
            for y1 in range(n - 1, -1, -1):
                high = ac.min(n-1, x1+y1)
                low = ac.max(0, x1+y1-(n-1))
                for x2 in range(high, low-1, -1):
                    y2 = x1 + y1 - x2
                    post = 0
                    for a, b in [[x1 + 1, y1], [x1, y1 + 1]]:
                        for c, d in [[x2 + 1, y2], [x2, y2 + 1]]:
                            if 0 <= a < n and 0 <= b < n and 0 <= c < n and 0 <= d < n:
                                post = ac.max(post, dp[a][b][c])
                    dp[x1][y1][x2] = post + grid[x1][y1] + grid[x2][y2]
                    if x1 == x2:
                        dp[x1][y1][x2] -= grid[x1][y1]
        ac.st(dp[0][0][0])
        return

    @staticmethod
    def lg_p1006(ac=FastIO()):
        # 模板：经典取数四维转三维DP，路径不能有交叠
        m, n = ac.read_ints()
        grid = [ac.read_list_ints() for _ in range(m)]

        dp = [[[0] * m for _ in range(n)] for _ in range(m)]
        for x1 in range(m - 1, -1, -1):
            for y1 in range(n - 1, -1, -1):
                high = ac.min(m-1, x1+y1)
                low = ac.max(0, x1+y1-(n-1))
                for x2 in range(high, low-1, -1):
                    y2 = x1 + y1 - x2
                    post = 0
                    for a, b in [[x1 + 1, y1], [x1, y1 + 1]]:
                        for c, d in [[x2 + 1, y2], [x2, y2 + 1]]:
                            if 0 <= a < m and 0 <= b < n and 0 <= c < m and 0 <= d < n:
                                post = ac.max(post, dp[a][b][c])
                    dp[x1][y1][x2] = post + grid[x1][y1] + grid[x2][y2]
                    if x1 == x2 and y1 == y2 and [x1, y1] not in [[0, 0], [m-1, n-1]]:
                        dp[x1][y1][x2] = -inf
        ac.st(dp[0][0][0])
        return

    @staticmethod
    def lg_p1107(ac=FastIO()):
        # 模板：矩阵DP加前缀数组最值优化
        n, h, d = ac.read_ints()
        cnt = [[0]*(h+1) for _ in range(n)]
        for i in range(n):
            lst = ac.read_list_ints()
            for j in lst[1:]:
                cnt[i][j] += 1

        ceil = [0]*(h+1)
        dp = [[0]*n for _ in range(2)]
        pre = 0
        for i in range(h, -1, -1):
            cur = 1-pre
            for j in range(n):
                dp[cur][j] = dp[pre][j]+cnt[j][i]
                if i+d <= h and ceil[i+d] > dp[pre][j]:
                    dp[cur][j] = ceil[i+d]+cnt[j][i]
            pre = cur
            ceil[i] = max(dp[pre])
        ac.st(ceil[0])
        return

    @staticmethod
    def lg_p1279(ac=FastIO()):
        # 模板：编辑距离 DP 变形
        s = ac.read_str()
        t = ac.read_str()
        k = ac.read_int()
        m, n = len(s), len(t)
        dp = [[inf]*(n+1) for _ in range(m+1)]
        dp[0][0] = 0
        for j in range(n):
            dp[0][j+1] = dp[0][j]+k
        for i in range(m):
            dp[i+1][0] = dp[i][0]+k
            for j in range(n):
                dp[i+1][j+1] = min(dp[i][j]+abs(ord(s[i])-ord(t[j])), dp[i+1][j]+k, dp[i][j+1]+k)
        ac.st(dp[-1][-1])
        return

    @staticmethod
    def lg_p1353(ac=FastIO()):
        # 模板：矩阵DP
        n, m = ac.read_ints()
        nums = [ac.read_int() for _ in range(n)]
        dp = [[-inf]*(m+1) for _ in range(n+1)]
        dp[0][0] = 0
        for i in range(n):
            dp[i+1][0] = dp[i][0]
            for j in range(1, min(i+2, m+1)):
                dp[i+1][0] = ac.max(dp[i+1][0], dp[i+1-j][j])
            for j in range(1, m+1):
                dp[i+1][j] = dp[i][j-1]+nums[i]
        ac.st(dp[n][0])
        return

    @staticmethod
    def lg_p1854(ac=FastIO()):
        # 模板：矩阵DP，并输出匹配方案
        m, n = ac.read_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        dp = [[-inf] * (n + 1) for _ in range(m + 1)]
        dp[0] = [0] * (n + 1)
        for i in range(m):
            for j in range(i, n):
                for k in range(i, j + 1):
                    if dp[i + 1][j + 1] < dp[i][k] + grid[i][j]:
                        dp[i + 1][j + 1] = dp[i][k] + grid[i][j]
        res = max(dp[-1])
        ac.st(res)

        # 倒序寻找最后一朵花插入的花瓶
        ans = [n]
        x = res
        for i in range(m - 1, -1, -1):
            for j in range(ans[-1] - 1, -1, -1):
                if dp[i + 1][j + 1] == x:
                    ans.append(j)
                    x -= grid[i][j]
                    break
        ans.reverse()
        ac.lst([x + 1 for x in ans[:-1]])
        return

    @staticmethod
    def lg_p2140(ac=FastIO()):
        # 模板：矩阵四维DP，可以使用记忆化与迭代计算
        m, n, u = ac.read_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        m, n = len(grid), len(grid[0])
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                pre[i + 1][j + 1] = pre[i][j + 1] + pre[i + 1][j] - pre[i][j] + grid[i][j]
        s = pre[-1][-1]

        # @lru_cache(None)
        # def dfs(xa, ya, xb, yb):
        #     w = pre[xb + 1][yb + 1] - pre[xb + 1][ya] - pre[xa][yb + 1] + pre[xa][ya]
        #     res = [-inf, -inf]
        #     if w >= s-u:
        #         res = [1, u-(s-w)]
        #     else:
        #         return res
        #
        #     for xx in range(xa, xb):
        #         nex1 = dfs(xa, ya, xx, yb)
        #         nex2 = dfs(xx+1, ya, xb, yb)
        #         nex = [nex1[0]+nex2[0], ac.min(nex1[1], nex2[1])]
        #         if nex > res:
        #             res = nex[:]
        #
        #     for yy in range(ya, yb):
        #         nex1 = dfs(xa, ya, xb, yy)
        #         nex2 = dfs(xa, yy+1, xb, yb)
        #         nex = [nex1[0]+nex2[0], ac.min(nex1[1], nex2[1])]
        #         if nex > res:
        #             res = nex[:]
        #     return res
        # ans = dfs(0, 0, m-1, n-1)
        # ac.lst(ans)

        dp = [[[[[-inf, -inf] for _ in range(n)] for _ in range(m)] for _ in range(n)] for _ in range(m)]
        for xa in range(m-1, -1, -1):
            for ya in range(n-1, -1, -1):
                for xb in range(xa, m):
                    for yb in range(ya, n):
                        w = pre[xb + 1][yb + 1] - pre[xb + 1][ya] - pre[xa][yb + 1] + pre[xa][ya]
                        if w < s - u:
                            continue
                        res = [1, u - (s - w)]

                        for xx in range(xa, xb):
                            nex1 = dp[xa][ya][xx][yb]
                            nex2 = dp[xx+1][ya][xb][yb]
                            nex = [nex1[0] + nex2[0], ac.min(nex1[1], nex2[1])]
                            if nex > res:
                                res = nex[:]
                        for yy in range(ya, yb):
                            nex1 = dp[xa][ya][xb][yy]
                            nex2 = dp[xa][yy+1][xb][yb]
                            nex = [nex1[0] + nex2[0], ac.min(nex1[1], nex2[1])]
                            if nex > res:
                                res = nex[:]
                        dp[xa][ya][xb][yb] = res
        ac.lst(dp[0][0][m-1][n-1])
        return

    @staticmethod
    def lg_p2217(ac=FastIO()):

        # 模板：矩阵四维DP，可以使用记忆化与迭代计算
        m, n, k = ac.read_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        avg = sum(sum(g) for g in grid)/k
        m, n = len(grid), len(grid[0])
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                pre[i + 1][j + 1] = pre[i][j + 1] + pre[i + 1][j] - pre[i][j] + grid[i][j]

        # @lru_cache(None)
        # def dfs(i, j, x, y, w):
        #     if w == 1:
        #         return (pre[x + 1][y + 1] - pre[x + 1][j] - pre[i][y + 1] + pre[i][j]-avg)**2
        #     res = inf
        #     for a in range(i, x):
        #         for up in range(1, w):
        #             res = ac.min(res, dfs(i, j, a, y, up)+dfs(a+1, j, x, y, w-up))
        #
        #     for b in range(j, y):
        #         for up in range(1, w):
        #             res = ac.min(res, dfs(i, j, x, b, up) + dfs(i, b+1, x, y, w - up))
        #     return res
        # ans = (dfs(0, 0, m-1, n-1, k)/k)**0.5
        # ac.st("%.2f" % ans)

        dp = [[[[[inf] * (k + 1) for _ in range(n)] for _ in range(m)] for _ in range(n)] for _ in range(m)]
        for i in range(m - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                for x in range(i, m):
                    for y in range(j, n):
                        for w in range(k + 1):
                            if w == 1:
                                res = (pre[x + 1][y + 1] - pre[x + 1][j] - pre[i][y + 1] + pre[i][j] - avg) ** 2
                                dp[i][j][x][y][w] = res
                                continue
                            res = inf
                            for a in range(i, x):
                                for up in range(1, w):
                                    res = ac.min(res, dp[i][j][a][y][up] + dp[a + 1][j][x][y][w - up])
                            for b in range(j, y):
                                for up in range(1, w):
                                    res = ac.min(res, dp[i][j][x][b][up] + dp[i][b + 1][x][y][w - up])
                            dp[i][j][x][y][w] = res
        ans = (dp[0][0][m - 1][n - 1][k] / k) ** 0.5
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p1436(ac=FastIO()):

        # 模板：矩阵四维DP，可以使用记忆化与迭代计算
        k = ac.read_int()
        m = n = 8
        grid = [ac.read_list_ints() for _ in range(m)]
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                pre[i + 1][j + 1] = pre[i][j + 1] + pre[i + 1][j] - pre[i][j] + grid[i][j]

        # @lru_cache(None)
        # def dfs(i, j, x, y, w):
        #     if w == 1:
        #         return (pre[x + 1][y + 1] - pre[x + 1][j] - pre[i][y + 1] + pre[i][j])**2
        #     res = inf
        #     for a in range(i, x):
        #         res = ac.min(res, dfs(i, j, a, y, 1)+dfs(a+1, j, x, y, w-1))
        #         res = ac.min(res, dfs(i, j, a, y, w-1) + dfs(a + 1, j, x, y, 1))
        #     for b in range(j, y):
        #         res = ac.min(res, dfs(i, j, x, b, 1) + dfs(i, b+1, x, y, w - 1))
        #         res = ac.min(res, dfs(i, j, x, b, w-1) + dfs(i, b + 1, x, y, 1))
        #     return res
        # ans = dfs(0, 0, m-1, n-1, k)
        # ac.st(ans)

        dp = [[[[[inf] * (k + 1) for _ in range(n)] for _ in range(m)] for _ in range(n)] for _ in range(m)]
        for i in range(m - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                for x in range(i, m):
                    for y in range(j, n):
                        for w in range(k + 1):
                            if w == 1:
                                res = (pre[x + 1][y + 1] - pre[x + 1][j] - pre[i][y + 1] + pre[i][j])**2
                                dp[i][j][x][y][w] = res
                                continue
                            res = inf
                            for a in range(i, x):
                                res = ac.min(res, dp[i][j][a][y][1] + dp[a + 1][j][x][y][w - 1])
                                res = ac.min(res, dp[i][j][a][y][w - 1] + dp[a + 1][j][x][y][1])
                            for b in range(j, y):
                                res = ac.min(res, dp[i][j][x][b][1] + dp[i][b + 1][x][y][w - 1])
                                res = ac.min(res, dp[i][j][x][b][w - 1] + dp[i][b + 1][x][y][1])
                            dp[i][j][x][y][w] = res
        ans = dp[0][0][m - 1][n - 1][k]
        ac.st(ans)
        return

    @staticmethod
    def lg_p5752(ac=FastIO()):

        # 模板：矩阵四维DP，可以使用记忆化与迭代计算
        k = ac.read_int()
        m = n = 8
        grid = [ac.read_list_ints() for _ in range(m)]
        avg = sum(sum(g) for g in grid) / k
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                pre[i + 1][j + 1] = pre[i][j + 1] + pre[i + 1][j] - pre[i][j] + grid[i][j]

        # @lru_cache(None)
        # def dfs(i, j, x, y, w):
        #     if w == 1:
        #         return (pre[x + 1][y + 1] - pre[x + 1][j] - pre[i][y + 1] + pre[i][j]-avg)**2
        #     res = inf
        #     for a in range(i, x):
        #         res = ac.min(res, dfs(i, j, a, y, 1)+dfs(a+1, j, x, y, w-1))
        #         res = ac.min(res, dfs(i, j, a, y, w-1) + dfs(a + 1, j, x, y, 1))
        #     for b in range(j, y):
        #         res = ac.min(res, dfs(i, j, x, b, 1) + dfs(i, b+1, x, y, w - 1))
        #         res = ac.min(res, dfs(i, j, x, b, w-1) + dfs(i, b + 1, x, y, 1))
        #     return res
        # ans = (dfs(0, 0, m-1, n-1, k)/k)**0.5
        # ac.st("%.3f" % ans)

        dp = [[[[[inf] * (k + 1) for _ in range(n)] for _ in range(m)] for _ in range(n)] for _ in range(m)]
        for i in range(m - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                for x in range(i, m):
                    for y in range(j, n):
                        for w in range(k + 1):
                            if w == 1:
                                res = (pre[x + 1][y + 1] - pre[x + 1][j] - pre[i][y + 1] + pre[i][j]-avg)**2
                                dp[i][j][x][y][w] = res
                                continue
                            res = inf
                            for a in range(i, x):
                                res = ac.min(res, dp[i][j][a][y][1] + dp[a + 1][j][x][y][w - 1])
                                res = ac.min(res, dp[i][j][a][y][w - 1] + dp[a + 1][j][x][y][1])
                            for b in range(j, y):
                                res = ac.min(res, dp[i][j][x][b][1] + dp[i][b + 1][x][y][w - 1])
                                res = ac.min(res, dp[i][j][x][b][w - 1] + dp[i][b + 1][x][y][1])
                            dp[i][j][x][y][w] = res
        ans = (dp[0][0][m - 1][n - 1][k]/k)**0.5
        ac.st("%.3f" % ans)
        return

    @staticmethod
    def lg_p2380(ac=FastIO()):
        # 模板：前缀和计算与矩阵DP
        while True:
            m, n = ac.read_ints()
            if m == n == 0:
                break

            grid_west = []
            for _ in range(m):
                lst = ac.read_list_ints()
                grid_west.append(ac.accumulate(lst))

            grid_north = [[0]*(n+1)]
            for _ in range(m):
                lst = ac.read_list_ints()
                grid_north.append([grid_north[-1][i]+lst[i] for i in range(n)])

            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m):
                for j in range(n):
                    # 只能往左或者往上挖
                    dp[i+1][j+1] = ac.max(dp[i][j+1] + grid_west[i][j+1], dp[i+1][j] + grid_north[i+1][j])
            ac.st(dp[-1][-1])
        return

    @staticmethod
    def lg_p2401(ac=FastIO()):
        # 模板：二维DP
        n, k = ac.read_ints()
        dp = [[0] * (k + 1) for _ in range(2)]
        pre = 0
        dp[pre][0] = 1
        mod = 2015
        for i in range(n):
            cur = 1 - pre
            dp[cur][0] = 1
            for j in range(1, ac.min(i + 1, k + 1)):
                dp[cur][j] = (dp[pre][j] * (j + 1) + dp[pre][j - 1] * (i - j + 1)) % mod
            pre = cur
        ac.st(dp[pre][-1])
        return

    @staticmethod
    def lg_p2528(ac=FastIO()):

        # 模板：逆序对矩阵 DP 与模拟构造
        n, t = ac.read_ints()
        dp = [[0] * (t + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(n):
            dp[i + 1][0] = 1
            for j in range(1, t + 1):
                dp[i + 1][j] = sum(dp[i][j - k] for k in range(min(i, j) + 1))
        ac.st(dp[-1][-1])

        lst = list(range(1, n + 1))
        ans = []
        for _ in range(n):
            m = len(lst)
            for i in range(m):
                rest = (m - 1) * (m - 2) // 2 + i
                if rest >= t:
                    ans.append(lst.pop(i))
                    t -= i
                    break
        ac.lst(ans)
        return

    @staticmethod
    def lg_p2733(ac=FastIO()):
        # 模板：经典DP通过边长与差分数组计算正方形子矩阵的个数
        n = ac.read_int()
        grid = [ac.read_str() for _ in range(n)]
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        diff = [0] * (n + 1)
        for i in range(n):
            for j in range(n):
                if grid[i][j] == "1":
                    # 转移公式
                    dp[i + 1][j + 1] = min(dp[i][j], dp[i + 1][j], dp[i][j + 1]) + 1
                    x = dp[i + 1][j + 1]
                    if x >= 2:
                        diff[2] += 1
                        if x + 1 <= n:
                            diff[x + 1] -= 1
        for i in range(2, n + 1):
            diff[i] += diff[i - 1]
            if diff[i] > 0:
                ac.lst([i, diff[i]])
        return

    @staticmethod
    def lg_p2736(ac=FastIO()):

        # 模板：矩阵 DP
        n, t, m = ac.read_ints()
        nums = ac.read_list_ints()

        # @lru_cache(None)
        # def dfs(i, j, pre):
        #     if i == n:
        #         return 0
        #     if j == m:
        #         return 0
        #     res = dfs(i + 1, j, pre)
        #     if pre + nums[i] <= t:
        #         res = ac.max(res, dfs(i + 1, j, pre + nums[i]) + 1)
        #     if nums[i] <= t and j + 1 < m:
        #         res = ac.max(res, dfs(i + 1, j + 1, nums[i]) + 1)
        #     return res
        #
        # ans = dfs(0, 0, 0)
        # ac.st(ans)

        dp = [[[0] * (t + 1) for _ in range(m + 1)] for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                for k in range(t, -1, -1):
                    res = dp[i + 1][j][k]
                    if k + nums[i] <= t:
                        res = ac.max(res, dp[i + 1][j][k + nums[i]] + 1)
                    if nums[i] <= t and j + 1 < m:
                        res = ac.max(res, dp[i + 1][j + 1][nums[i]] + 1)
                    dp[i][j][k] = res
        ac.st(dp[0][0][0])
        return

    @staticmethod
    def lg_p2769(ac=FastIO()):

        # 模板：矩阵 DP 注意初始化条件
        n = ac.read_int()
        a = ac.read_list_ints()
        a.sort()
        m = ac.read_int()
        b = ac.read_list_ints()
        b.sort()

        # @lru_cache(None)
        # def dfs(i, j, state):
        #     if i == n:
        #         return 0 if j == m-1 and state else inf
        #     if not state:
        #         return abs(a[i]-b[j])+dfs(i+1, j, 1)
        #
        #     res = dfs(i+1, j, 1)+abs(a[i]-b[j])
        #     if state and j+1<m:
        #         res = min(res, dfs(i+1, j+1, 1)+abs(a[i]-b[j+1]))
        #     return res
        # ac.st(dfs(0, 0, 0))

        dp = [[inf for _ in range(m + 1)] for _ in range(2)]
        pre = 0
        dp[pre][0] = 0
        for i in range(n):
            cur = 1 - pre
            dp[cur][0] = inf
            for j in range(m):
                dp[cur][j + 1] = ac.min(dp[pre][j] + abs(a[i] - b[j]), dp[pre][j + 1] + abs(a[i] - b[j]))
            pre = cur
        ac.st(dp[pre][-1])
        return


class TestGeneral(unittest.TestCase):

    def test_matrix_dp(self):
        md = MatrixDP()
        matrix = [["1", "0", "1", "0", "0"], ["1", "0", "1", "1", "1"], [
            "1", "1", "1", "1", "1"], ["1", "0", "0", "1", "0"]]
        assert md.maximal_square(matrix) == 4
        return


if __name__ == '__main__':
    unittest.main()
