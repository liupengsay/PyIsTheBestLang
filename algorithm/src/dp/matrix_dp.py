import unittest
from bisect import bisect_left
from collections import defaultdict

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

================================CodeForces================================
https://codeforces.com/problemset/problem/1446/B（最长公共子序列LCS变形问题，理解贡献）
https://codeforces.com/problemset/problem/429/B（四个方向的矩阵DP）
D. Colored Rectangles（https://codeforces.com/problemset/problem/1398/D）三维DP，选取两个不同数组的数乘积，计算最大总和
B. The least round way（https://codeforces.com/problemset/problem/2/B）矩阵DP，计算路径上乘积最少的后缀0个数，经典题目


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


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1305(self, nums1: List[int], nums2: List[int]) -> int:
        # 模板：使用LIS的办法求LCS
        return LcsLis().longest_common_subsequence(nums1, nums2)

    @staticmethod
    def lc_1143(self, s1: str, s2: str) -> int:
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
                    dp[i + 1][j + 1] = min(dp[i][j],
                                           dp[i + 1][j], dp[i][j + 1]) + 1
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


class TestGeneral(unittest.TestCase):

    def test_matrix_dp(self):
        md = MatrixDP()
        matrix = [["1", "0", "1", "0", "0"], ["1", "0", "1", "1", "1"], [
            "1", "1", "1", "1", "1"], ["1", "0", "0", "1", "0"]]
        assert md.maximal_square(matrix) == 4
        return


if __name__ == '__main__':
    unittest.main()
