import unittest

from typing import List
from types import GeneratorType

"""
算法：矩阵DP、二维DP
功能：在二维矩阵上进行转移的DP，经典的有矩阵前缀和，矩阵区间和，正方形最大边长或面积，编辑距离，公共子序列，最长回文子串
题目：

===================================力扣===================================
2435. 矩阵中和能被 K 整除的路径（https://leetcode.cn/problems/paths-in-matrix-whose-sum-is-divisible-by-k/）利用模 K 的特点进行路径计算
2088. 统计农场中肥沃金字塔的数目（https://leetcode.cn/problems/count-fertile-pyramids-in-a-land/）类似求正方形的边长和面积进行矩阵DP
221. 最大正方形（https://leetcode.cn/problems/maximal-square/）求全为 1 的最大正方形面积
72. 编辑距离（https://leetcode.cn/problems/edit-distance/）矩阵DP
329. 矩阵中的最长递增路径（https://leetcode.cn/problems/longest-increasing-path-in-a-matrix/）二维矩阵DP
1478. 安排邮筒（https://leetcode.cn/problems/allocate-mailboxes/）二维DP与一个计算不带权中位数距离的区间DP

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

参考：OI WiKi（xx）
"""


class Solution:
    def __init__(self):
        return

    @staticmethod
    def bootstrap(f, queue=[]):
        def wrappedfunc(*args, **kwargs):
            if queue:
                return f(*args, **kwargs)
            else:
                to = f(*args, **kwargs)
                while True:
                    if isinstance(to, GeneratorType):
                        queue.append(to)
                        to = next(to)
                    else:
                        queue.pop()
                        if not queue:
                            break
                        to = queue[-1].send(to)
                return to
        return wrappedfunc

    def cf_1398d(self, r, g, b, lst):
        # 模板：三维DP，选取两个不同数组的数乘积，计算最大总和
        @self.bootstrap
        def dfs(i, j, k):
            if dp[i][j][k] != -1:
                yield
            res = 0
            if i < r and j < g:
                yield dfs(i + 1, j + 1, k)
                res = max(res, dp[i + 1][j + 1][k] + lst[0][i] * lst[1][j])
            if i < r and k < b:
                yield dfs(i + 1, j, k + 1)
                res = max(res, dp[i + 1][j][k + 1] + lst[0][i] * lst[2][k])
            if j < g and k < b:
                yield dfs(i, j + 1, k + 1)
                res = max(res, dp[i][j + 1][k + 1] + lst[2][k] * lst[1][j])
            dp[i][j][k] = res
            yield

        dp = [[[-1] * (b + 1) for _ in range(g + 1)] for _ in range(r + 1)]
        dfs(0, 0, 0)
        return dp[0][0][0]


class MatrixDP:
    def __init__(self):
        return

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
                dp[i][j] = min(dp[i + 1][j] + 1, dp[i][j + 1] + 1, dp[i + 1][j + 1] + int(word1[i] != word2[j]))
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
                            if dp[i + 1][j + 1][p + 1] < dp[i + a][j + b][p + c]:
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
