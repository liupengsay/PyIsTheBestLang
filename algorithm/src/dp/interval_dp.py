
import unittest
from typing import List
from algorithm.src.fast_io import FastIO, inf


MOD = 10 ** 9 + 7


"""
算法：区间DP
功能：前缀和优化区间DP（需要在状态转移的时候更新代价距离）、预处理区间DP（需要预处理一个DP再计算最终DP）
题目：

===================================力扣===================================
2472. 不重叠回文子字符串的最大数目（https://leetcode.cn/problems/maximum-number-of-non-overlapping-palindrome-substrings/）回文子串判定DP加线性DP
2430. 对字母串可执行的最大删除数（https://leetcode.cn/problems/maximum-deletions-on-a-string/）最长公共前缀DP加线性DP
1547. 切棍子的最小成本（https://leetcode.cn/problems/minimum-cost-to-cut-a-stick/）区间DP模拟

===================================洛谷===================================
P1521 求逆序对（https://www.luogu.com.cn/problem/P1521）使用归并排序计算移动次数，也可以使用倍增的树状数组
P1775 石子合并（弱化版）（https://www.luogu.com.cn/problem/P1775）典型区间DP和前缀和预处理
P2426 删数（https://www.luogu.com.cn/problem/P2426）典型区间DP
P2690 [USACO04NOV]Apple Catching G（https://www.luogu.com.cn/problem/P2690）区间DP记忆化搜索模拟
P1435 [IOI2000] 回文字串（https://www.luogu.com.cn/problem/P1435）典型区间DP求最长不连续回文子序列
P1388 算式（https://www.luogu.com.cn/problem/P1388）回溯枚举符号组合，再使用区间DP进行最大值求解
P1103 书本整理（https://www.luogu.com.cn/problem/P1103）三维DP
P2858 [USACO06FEB]Treats for the Cows G/S（https://www.luogu.com.cn/problem/P2858）典型区间DP
P1880 石子合并（https://www.luogu.com.cn/problem/P1880）将数组复制成两遍进行区间DP

================================CodeForces================================
C. The Sports Festival（https://codeforces.com/problemset/problem/1509/C）转换为区间DP进行求解
B. Zuma（https://codeforces.com/problemset/problem/607/B）区间DP，经典通过消除回文子序列删除整个数组的最少次数

参考：OI WiKi（xx）
"""


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_307b(ac=FastIO()):
        #
        n = ac.read_int()
        nums = ac.read_list_ints()

        # 初始化
        dp = [[inf] * n for _ in range(n + 1)]
        for i in range(n):
            for j in range(i):
                dp[i][j] = 0
        dp[n] = [0] * n

        # 状态转移
        for i in range(n - 1, -1, -1):
            dp[i][i] = 1
            if i + 1 < n:
                dp[i][i + 1] = 2 if nums[i] != nums[i + 1] else 1
            for j in range(i + 2, n):

                dp[i][j] = ac.min(dp[i + 1][j], dp[i][j - 1]) + 1
                if nums[i] == nums[i + 1]:
                    dp[i][j] = ac.min(dp[i][j], 1 + dp[i + 2][j])

                for k in range(i + 2, j + 1):
                    dp[i][j] = ac.min(dp[i][j], dp[i][k] + dp[k + 1][j])
                    if nums[k] == nums[i]:
                        dp[i][j] = ac.min(dp[i][j], dp[i + 1][k - 1] + dp[k + 1][j])

        ac.st(dp[0][n - 1])
        return

    @staticmethod
    def cf_1509c(n, nums):
        # 模板：使用数组进行区间DP转移求解
        dp = [[float("inf")] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            dp[i][i] = 0
            for j in range(i + 1, n):
                dp[i][j] = nums[j] - nums[i] + min(dp[i + 1][j], dp[i][j - 1])
        return dp[0][n - 1]

    @staticmethod
    def lc_2472(s: str, k: int) -> int:
        n = len(s)
        res = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            res[i][i] = 1
            if i + 1 < n:
                res[i][i + 1] = 1 if s[i] == s[i + 1] else 0
            for j in range(i + 2, n):
                if s[i] == s[j] and res[i + 1][j - 1]:
                    res[i][j] = 1

        dp = [0] * (n + 1)
        for i in range(n):
            dp[i + 1] = dp[i]
            for j in range(0, i - k + 2):
                if i - j + 1 >= k and res[j][i]:
                    dp[i + 1] = max(dp[i + 1], dp[j] + 1)
        return dp[-1]


class TestGeneral(unittest.TestCase):

    def test_interval_dp(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
