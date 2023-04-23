import math
import random
import unittest
from collections import defaultdict
from typing import List
from algorithm.src.mathmatics.number_theory import NumberTheory
from algorithm.src.fast_io import FastIO, inf

"""
算法：背包DP、分组背包、一维（无限有限）背包、二位背包、多重背包、分组背包、限制背包
功能：一重背包DP，数量有限从后往前遍历，数量无限则从前往后遍历；多重背包DP，可使用二进制优化进行拆分
题目：

===================================力扣===================================
2218. 从栈中取出 K 个硬币的最大面值和（https://leetcode.cn/problems/maximum-value-of-k-coins-from-piles/）分组背包DP
6310. 获得分数的方法数（https://leetcode.cn/contest/weekly-contest-335/problems/number-of-ways-to-earn-points/）看似二进制优化背包，实则数量转移
2189. 建造纸牌屋的方法数（https://leetcode.cn/problems/number-of-ways-to-build-house-of-cards/）转换为01背包求解
254. 因子的组合（https://leetcode.cn/problems/factor-combinations/）乘法结合背包DP


===================================洛谷===================================
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
P1474 [USACO2.3]Money System / [USACO07OCT]Cow Cash G（https://www.luogu.com.cn/problem/P1474）一维无限背包计数
P1466 [USACO2.2]集合 Subset Sums（https://www.luogu.com.cn/problem/P1466）一维有限背包加和计数
P1455 搭配购买（https://www.luogu.com.cn/problem/P1455）并查集进行搭配购买组合加一维有限背包
P1230 智力大冲浪（https://www.luogu.com.cn/problem/P1230）排序后根据时间限制进行动态更新一维有限背包
P1077 [NOIP2012 普及组] 摆花（https://www.luogu.com.cn/problem/P1077）一维有限背包计数
P2725 [USACO3.1]邮票 Stamps（https://www.luogu.com.cn/problem/P2725）01无限背包计数
P2918 [USACO08NOV]Buying Hay S（https://www.luogu.com.cn/problem/P2918）一维无限背包，需要根据题意增加背包容量上限计算
P3027 [USACO10OCT]Making Money G（https://www.luogu.com.cn/problem/P3027）一维无限背包，需要根据题意进行利润计算
P3030 [USACO11NOV]Tile Exchanging S（https://www.luogu.com.cn/problem/P3030）分组枚举有限背包
P3040 [USACO12JAN]Bale Share S（https://www.luogu.com.cn/problem/P3040）二维变种背包
P4817 [USACO15DEC]Fruit Feast G（https://www.luogu.com.cn/problem/P4817）一维有限背包DP变种
P5087 数学（https://www.luogu.com.cn/problem/P5087）二维有限背包变种问题
P6205 [USACO06JAN]Dollar Dayz S（https://www.luogu.com.cn/problem/P6205）一维无限背包
P6389 [COCI2007-2008#4] MUZICARI（https://www.luogu.com.cn/problem/P6389）一维有限背包变种问题，寻找和尽可能接近的两个分组
P6567 [NOI Online #3 入门组] 买表（https://www.luogu.com.cn/problem/P6567）一维二进制优化有限背包，即物品数为连续值时需要使用二进制优化
P6771 [USACO05MAR]Space Elevator 太空电梯（https://www.luogu.com.cn/problem/P6771）排序后，一维有限变种背包，使用二进制优化


================================CodeForces================================
B. Modulo Sum（https://codeforces.com/problemset/problem/577/B）取模计数二进制优化与背包DP，寻找非空子序列的和整除给定的数
A. Writing Code（https://codeforces.com/problemset/problem/543/A）二维有限背包DP，当作无限进行处理
E. Porcelain（https://codeforces.com/problemset/problem/148/E）01背包枚举，两层动态规划
F. Zero Remainder Sum（https://codeforces.com/problemset/problem/1433/F）01背包枚举，两层动态规划

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


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1433f(ac=FastIO()):
        # 模板：两层背包DP，矩阵动态规划转移
        m, n, k = ac.read_ints()
        pre = [-inf] * k
        pre[0] = 0
        x = n // 2
        for _ in range(m):
            nums = ac.read_list_ints()
            dp = [[-inf] * k for _ in range(x + 1)]
            dp[0][0] = 0
            for num in nums:
                nex = [ls[:] for ls in dp]
                for i in range(x):
                    for j in range(k):
                        d = (j + num) % k
                        nex[i + 1][d] = ac.max(dp[i][j] + num, nex[i + 1][d])
                dp = [ls[:] for ls in nex]
            tmp = [max(dp[i][j] for i in range(x + 1)) for j in range(k)]

            cur = pre[:]
            for i in range(k):
                for j in range(k):
                    cur[(i + j) % k] = ac.max(cur[(i + j) % k], pre[i] + tmp[j])
            pre = cur[:]

        ac.st(pre[0])
        return

    @staticmethod
    def cf_543a(ac=FastIO()):
        # 模板：分组背包 DP 有限作为无限
        n, m, b, mod = ac.read_ints()
        nums = ac.read_list_ints()
        pre = [[0] * (b + 1) for _ in range(m + 1)]
        pre[0][0] = 1
        for num in nums:
            for i in range(1, m + 1):
                # 由于每个用户的天数都可以取到 m 所以当作类似无限背包进行转移
                for j in range(num, b + 1):
                    pre[i][j] = (pre[i][j] + pre[i - 1][j - num]) % mod
        ac.st(sum(pre[m]) % mod)
        return

    @staticmethod
    def cf_577b(m, nums):
        # 模板：取模计数二进制优化与背包DP，寻找非空子序列的和整除给定的数
        cnt = [0] * m
        for num in nums:
            cnt[num % m] += 1
        if cnt[0] or max(cnt) >= m:
            return "YES"
        pre = [0] * m
        for i in range(1, m):
            if cnt[i]:
                for x in BagDP().bin_split(cnt[i]):
                    cur = pre[:]
                    y = (x * i) % m
                    cur[y] = 1
                    for j in range(m):
                        if pre[j]:
                            cur[(j + y) % m] = 1
                    pre = cur[:]
                if pre[0]:
                    return "YES"
        return "NO"

    @staticmethod
    def lc_2218(piles: List[List[int]], k: int) -> int:

        # 模板：线性有限分组背包 DP 注意转移
        cur = [0] * (k + 1)
        for lst in piles:

            n = len(lst)
            pre = [0] * (n + 1)
            for i in range(n):
                pre[i + 1] = pre[i] + lst[i]
            # 注意这里需要进行拷贝
            nex = cur[:]
            for j in range(1, k + 1):
                for x in range(min(n+1, j+1)):
                    nex[j] = max(nex[j], cur[j - x] + pre[x])
            cur = nex[:]
        return cur[-1]

    @staticmethod
    def lg_p6567(ac=FastIO()):
        # 模板：一维有限二进制优化背包
        n, m = ac.read_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        target = ac.read_list_ints()
        ceil = max(target)
        dp = [0] * (ceil + 1)
        dp[0] = 1
        for k, a in nums:
            for b in BagDP().bin_split(a):
                x = b * k
                for i in range(ceil, x - 1, -1):
                    if dp[i - x]:
                        dp[i] = 1
        for t in target:
            if dp[t]:
                ac.st("Yes")
            else:
                ac.st("No")
        return

    @staticmethod
    def lc_6310(target: int, types: List[List[int]]) -> int:
        # 模板：看似二进制优化 DP 实则矩阵 DP 转移
        mod = 10 ** 9 + 7
        n = len(types)
        pre = [0] * (target + 1)
        pre[0] = 1
        for i in range(n):
            c, m = types[i]
            cur = pre[:]
            for x in range(1, c + 1):
                for j in range(target - x * m + 1):
                    if x * m + j <= target:
                        cur[x * m + j] += pre[j]
            pre = [num % mod for num in cur]
        return pre[-1]

    @staticmethod
    def lc_254(n: int) -> List[List[int]]:
        # 模板：使用因子分解与背包dp进行分解计算
        lst = NumberTheory().get_all_factor(n)
        m = len(lst)
        dp = defaultdict(list)
        dp[1] = [[]]
        for i in range(1, m-1):
            for j in range(i, m):
                if lst[j] % lst[i] == 0:
                    x = lst[j] // lst[i]
                    for p in dp[x]:
                        dp[lst[j]].append(p+[lst[i]])
        return [ls for ls in dp[n] if ls]


class TestGeneral(unittest.TestCase):

    def test_bag_dp(self):
        bd = BagDP()
        for _ in range(1000):
            num = random.randint(1, 100000000)
            assert sum(bd.bin_split(num)) == num
        return


if __name__ == '__main__':
    unittest.main()
