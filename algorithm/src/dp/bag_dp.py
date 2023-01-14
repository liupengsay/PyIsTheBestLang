import math
import random
import unittest
from collections import defaultdict
from itertools import combinations
from itertools import permutations
from decimal import Decimal, getcontext, MAX_PREC
from types import GeneratorType
from functools import cmp_to_key


"""
算法：背包DP、分组背包、一维（无限有限）背包、二位背包、多重背包、分组背包、限制背包
功能：一重背包DP，数量有限从后往前遍历，数量无限则从前往后遍历；多重背包DP，可使用二进制拆分进行优化。

题目：
L0214 最短回文串（https://leetcode.cn/problems/shortest-palindrome/）计算字符串前缀最长回文子串
L2218 从栈中取出 K 个硬币的最大面值和（https://leetcode.cn/problems/maximum-value-of-k-coins-from-piles/）背包DP
P1048 采药（https://www.luogu.com.cn/problem/P1048）一维背包DP，数量有限，从后往前遍历
P1049 [NOIP2001 普及组] 装箱问题（https://www.luogu.com.cn/problem/P1049）一维背包DP
P1776 宝物筛选（https://www.luogu.com.cn/problem/P1776）多重背包，使用二进制拆分进行优化
P1509 找啊找啊找GF（https://www.luogu.com.cn/problem/P1509）四重背包
P1060 [NOIP2006 普及组] 开心的金明（https://www.luogu.com.cn/problem/P1509）一维背包DP
P1566 加等式（https://www.luogu.com.cn/problem/P1566#submit）限制计数背包
P1759 通天之潜水（https://www.luogu.com.cn/problem/P1759）二重背包
P1794 装备运输（https://www.luogu.com.cn/problem/P1794）二重背包
P1806 跑步（https://www.luogu.com.cn/problem/P1806）连续值一维有限背包计数
P1853 投资的最大效益（https://www.luogu.com.cn/problem/P1853）一维无限背包有技巧成倍缩小背包范围

P1874 快速求和（https://www.luogu.com.cn/problem/P1874）类似区间与背包的结合枚举前一个字符串加号分割点求和
P1977 出租车拼车（https://www.luogu.com.cn/problem/P1977）分组有限背包
P1586 四方定理（https://www.luogu.com.cn/problem/P1586）分组无限背包
P1566 加等式（https://www.luogu.com.cn/problem/P1566）一维有限背包计数
P1509 找啊找啊找GF（https://www.luogu.com.cn/problem/P1509）二重背包，转移的时候比较优先级有两个

P1504 积木城堡（https://www.luogu.com.cn/problem/P1504）一维有限背包DP
P2066 机器分配（https://www.luogu.com.cn/problem/P2066）分组有限背包，转移的时候比较优先级有两个

P2340 [USACO03FALL]Cow Exhibition G（https://www.luogu.com.cn/problem/P2340）经典01背包变种问题还带负数加和

P2370 yyy2015c01 的 U 盘（https://www.luogu.com.cn/problem/P2370）使用最小生成树的思想排序后贪心进行背包放入，达成条件后即中止
P2386 放苹果（https://www.luogu.com.cn/problem/P2386）背包DP进行去重组合加和计数

P2623 物品选取（https://www.luogu.com.cn/problem/P2623）综合经典背包，函数取最大值进行一维有限背包，连续个数使用二进制优化背包，无限个数背包

参考：OI WiKi（xx）
"""


class BagDP:
    def __init__(self):
        return

    @staticmethod
    def bin_split(num):
        # 二进制优化是指 1.2.4.x这样连续的而不是二进制10101对应的1
        assert num > 0
        lst = []
        x = 1
        while x <= num:
            lst.append(x)
            num -= x
            x *= 2
        if num:
            lst.append(num)
        return lst

    @staticmethod
    def one_dimension_limited(n, nums):
        # 一维有限背包
        dp = [0] * (n + 1)
        dp[0] = 1
        for num in nums:
            for i in range(n, num - 1, -1):
                dp[i] += dp[i - num]
        return dp[n]

    @staticmethod
    def one_dimension_unlimited(n, nums):
        # 一维无限背包
        dp = [0] * (n + 1)
        dp[0] = 1
        for num in nums:
            for i in range(num, n + 1):
                dp[i] += dp[i - num]
        return dp[n]

    @staticmethod
    def two_dimension_limited(m, n, nums):
        # 二维有限背包（多维背包类似）
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 1
        for a, b in nums:
            for i in range(m, a - 1, -1):
                for j in range(n, b - 1, -1):
                    dp[i][j] += dp[i - a][j - b]
        return dp[m][n]

    @staticmethod
    def two_dimension_unlimited(m, n, nums):
        # 二维无限背包（多维背包类似）
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 1
        for a, b in nums:
            for i in range(a, m + 1):
                for j in range(b, n + 1):
                    dp[i][j] += dp[i - a][j - b]
        return dp[m][n]

    def continuous_bag_with_bin_split(self, n, nums):
        # 使用二进制优化的连续背包（以一维有限背包为例）
        dp = [0] * (n + 1)
        dp[0] = 1
        for num in nums:
            for x in self.bin_split(num):
                for i in range(n, x - 1, -1):
                    dp[i] += dp[i - x]
        return dp[n]

    @staticmethod
    def group_bag_limited(n, d, nums):
        # 分组背包（以一维有限背包为例）计算出租车的最小花费
        pre = [float("inf")] * (n + 1)
        pre[0] = 0
        for r, z in nums:
            cur = pre[:]  # 关键在于这里需要分组背包
            for x in range(1, z + 1):
                cost = d + x * r
                for i in range(n, x - 1, -1):
                    if pre[i - x] + cost < cur[i]:
                        cur[i] = pre[i - x] + cost
            pre = cur[:]
        if pre[n] < float("inf"):
            return pre[n]
        return -1

    @staticmethod
    def group_bag_unlimited(nums):
        # 分组背包（以一维无限背包为例）计算 n 分解成四个数的平方和的方案数
        n = max(nums)
        dp = [[0] * 5 for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, int(math.sqrt(n)) + 1):
            x = i * i
            for j in range(x, n + 1):
                for k in range(1, 5):
                    if dp[j - x][k - 1]:
                        dp[j][k] += dp[j - x][k - 1]
        return [sum(dp[num]) for num in nums]

    @staticmethod
    def one_dimension_limited_use_dct(nums):
        # 一维有限背包（带负数的情况下使用字典做转移记录）
        inf = float("inf")
        pre = defaultdict(lambda: -inf)
        pre[0] = 0
        for s, f in nums:
            cur = pre.copy()
            for p in pre:
                cur[p + s] = max(cur[p + s], pre[p] + f)
            pre = cur
        ans = 0
        for p in pre:
            if p >= 0 and pre[p] >= 0:
                ans = ans if ans > p + pre[p] else p + pre[p]
        return ans


class TestGeneral(unittest.TestCase):

    def test_bag_dp(self):
        bd = BagDP()
        for _ in range(1000):
            num = random.randint(1, 100000000)
            assert sum(bd.bin_split(num)) == num
        return


if __name__ == '__main__':
    unittest.main()
