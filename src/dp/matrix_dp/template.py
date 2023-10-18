from math import inf

from typing import List


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
        dp = [[inf] * (n + 1) for _ in range(m + 1)]
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
