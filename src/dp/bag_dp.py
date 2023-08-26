import math
import random
import unittest
from collections import defaultdict, deque, Counter
from functools import lru_cache
from itertools import combinations
from typing import List
from math import inf
from src.graph.union_find import UnionFind
from src.mathmatics.number_theory import NumberTheory
from src.fast_io import FastIO


"""
算法：背包DP、分组背包、一维（无限有限）背包、二位背包、多重背包、分组背包、限制背包、填表法（过去状态预测未来状态）、刷表法（当前状态预测未来状态）
功能：一重背包DP，数量有限从后往前遍历，数量无限则从前往后遍历；多重背包DP，可使用二进制优化进行拆分
题目：

===================================力扣===================================
140. 单词拆分 II（https://leetcode.cn/problems/word-break-ii/）经典 01 背包生成具体方案
2218. 从栈中取出 K 个硬币的最大面值和（https://leetcode.cn/problems/maximum-value-of-k-coins-from-piles/）分组背包DP
2585. 获得分数的方法数（https://leetcode.cn/contest/weekly-contest-335/problems/number-of-ways-to-earn-points/）看似二进制优化背包，实则数量转移
2189. 建造纸牌屋的方法数（https://leetcode.cn/problems/number-of-ways-to-build-house-of-cards/）转换为01背包求解
254. 因子的组合（https://leetcode.cn/problems/factor-combinations/）乘法结合背包DP
1449. 数位成本和为目标值的最大数字（https://leetcode.cn/problems/form-largest-integer-with-digits-that-add-up-to-target/）代价一定情况下的最大数值
1049. 最后一块石头的重量 II（https://leetcode.cn/problems/last-stone-weight-ii/）经典问题，转化为01背包求解
2742. 给墙壁刷油漆（https://leetcode.cn/problems/painting-the-walls/description/）经典剪枝DP，可以转换为01背包求解

===================================洛谷===================================
P1048 采药（https://www.luogu.com.cn/problem/P1048）一维背包DP，数量有限，从后往前遍历
P1049 [NOIP2001 普及组] 装箱问题（https://www.luogu.com.cn/problem/P1049）一维背包DP
P1776 宝物筛选（https://www.luogu.com.cn/problem/P1776）多重背包，使用二进制拆分进行优化，进一步使用单调队列优化
P1509 找啊找啊找GF（https://www.luogu.com.cn/problem/P1509）四重背包
P1060 [NOIP2006 普及组] 开心的金明（https://www.luogu.com.cn/problem/P1509）一维背包DP
P1566 加等式（https://www.luogu.com.cn/problem/P1566#submit）限制计数背包
P1759 通天之潜水（https://www.luogu.com.cn/problem/P1759）二重背包并输出方案
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
P2842 纸币问题 1（https://www.luogu.com.cn/problem/P2842）一维无限背包DP不区分顺序
P2840 纸币问题 2（https://www.luogu.com.cn/problem/P2840）一维无限背包DP区分顺序
P2834 纸币问题 3（https://www.luogu.com.cn/problem/P2834）一维无限背包DP不区分顺序
P1064 [NOIP2006 提高组] 金明的预算方案（https://www.luogu.com.cn/problem/P1064）有依赖的01背包，枚举状态进行分组讨论，分组背包
P1156 垃圾陷阱（https://www.luogu.com.cn/problem/P1156）转换为背包01DP求解
P1273 有线电视网（https://www.luogu.com.cn/problem/P1273）树上分组背包
P1284 三角形牧场（https://www.luogu.com.cn/problem/P1284）枚举三角形两边作为二维bool背包，并使用三角形面积计算公式
P1441 砝码称重（https://www.luogu.com.cn/problem/P1441）枚举加背包DP
P1537 弹珠（https://www.luogu.com.cn/problem/P1537）经典问题二进制背包优化bool背包，划分成和相等的两部分
P1541 [NOIP2010 提高组] 乌龟棋（https://www.luogu.com.cn/problem/P1541）四维背包枚举，填表法
P1759 通天之潜水（https://www.luogu.com.cn/problem/P1759）二维背包并输出字典序最小的方案
P1833 樱花（https://www.luogu.com.cn/problem/P1833）完全背包与单点队列优化多重背包组合
P2014 [CTSC1997] 选课（https://www.luogu.com.cn/problem/P2014）增加一个虚拟源点将DAG转换为树上背包
P2079 烛光晚餐（https://www.luogu.com.cn/problem/P2079）滚动哈希背包DP，使用两层哈希节省空间
P2170 选学霸（https://www.luogu.com.cn/problem/P2170）连通块加二进制01背包优化
P2214 [USACO14MAR]Mooo Moo S（https://www.luogu.com.cn/problem/P2214）变种背包DP贪心
P2306 被 yyh 虐的 mzc（https://www.luogu.com.cn/problem/P2306）根据数据范围计数后进行二进制优化的01背包计算
P2320 [HNOI2006] 鬼谷子的钱袋（https://www.luogu.com.cn/problem/P2320）二进制分解贪心反向计算
P2737 [USACO4.1]麦香牛块Beef McNuggets（https://www.luogu.com.cn/problem/P2737）模板：完全背包变种问题
P2760 科技庄园（https://www.luogu.com.cn/problem/P2760）单调队列优化的多重背包
P2854 [USACO06DEC]Cow Roller Coaster S（https://www.luogu.com.cn/problem/P2854）分组01背包
P2938 [USACO09FEB]Stock Market G（https://www.luogu.com.cn/problem/P2938）分组完全背包
P2979 [USACO10JAN]Cheese Towers S（https://www.luogu.com.cn/problem/P2979）分组01背包
P3010 [USACO11JAN]Dividing the Gold S（https://www.luogu.com.cn/problem/P3010）经典变形01背包，计算两堆差值最小的分配方案数
P3423 [POI2005]BAN-Bank Notes（https://www.luogu.com.cn/problem/P3423）二进制优化多重背包与方案输出
P3983 赛斯石（赛后强化版）（https://www.luogu.com.cn/problem/P3983）两个分组完全背包计算
P5322 [BJOI2019] 排兵布阵（https://www.luogu.com.cn/problem/P5322）典型二维 DP 转换为分组背包
P5365 [SNOI2017] 英雄联盟（https://www.luogu.com.cn/problem/P5365）01背包 DP 枚举数量
P5662 [CSP-J2019] 纪念品（https://www.luogu.com.cn/problem/P5662）完全背包变形贪心题目
P1417 烹调方案（https://www.luogu.com.cn/problem/P1417）经典贪心排序后计算 01 背包最大值

================================CodeForces================================
B. Modulo Sum（https://codeforces.com/problemset/problem/577/B）取模计数二进制优化与背包DP，寻找非空子序列的和整除给定的数
A. Writing Code（https://codeforces.com/problemset/problem/543/A）二维有限背包DP，当作无限进行处理
E. Porcelain（https://codeforces.com/problemset/problem/148/E）01背包枚举，两层动态规划
F. Zero Remainder Sum（https://codeforces.com/problemset/problem/1433/F）01背包枚举，两层动态规划

================================AcWing=====================================
4. 多重背包问题 I（https://www.acwing.com/problem/content/4/）二进制优化多重背包
6. 多重背包问题 III（https://www.acwing.com/problem/content/description/6/）单调队列优化多重背包
7. 混合背包问题（https://www.acwing.com/problem/content/7/）01背包、完全背包与多重背包混合使用
8. 二维费用的背包问题（https://www.acwing.com/problem/content/8/）二维01背包
9. 分组背包问题（https://www.acwing.com/problem/content/9/）分组01背包问题
10. 有依赖的背包问题（https://www.acwing.com/problem/content/10/）树上背包
11. 背包问题求方案数（https://www.acwing.com/problem/content/description/11/）背包问题求方案数
12. 背包问题求具体方案（https://www.acwing.com/problem/content/12/）背包问题求具体方案，有两种写法
4081. 选数（https://www.acwing.com/problem/content/4084/）转换为二维背包问题求解

参考：OI WiKi（xx）
"""


class BagDP:
    def __init__(self):
        return

    @staticmethod
    def bin_split(num):
        # 二进制优化是指 1.2.4.x这样连续的而不是二进制10101对应的1
        if not num:
            return []
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
        pre = [inf] * (n + 1)
        pre[0] = 0
        for r, z in nums:
            cur = pre[:]  # 关键在于这里需要分组背包
            for x in range(1, z + 1):
                cost = d + x * r
                for i in range(n, x - 1, -1):
                    if pre[i - x] + cost < cur[i]:
                        cur[i] = pre[i - x] + cost
            pre = cur[:]
        if pre[n] < inf:
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
    def lc_2742_1(cost: List[int], time: List[int]) -> int:

        # 模板：经典剪枝DP，可以转换为01背包求解
        @lru_cache(None)
        def dfs(i, pre):
            if pre >= n - i:  # 剪枝
                return 0
            if i == n:
                return inf
            res = dfs(i + 1, pre - 1)
            cur = dfs(i + 1, pre + time[i]) + cost[i]
            if cur < res:
                res = cur
            return res

        n = len(cost)
        return dfs(0, 0)

    @staticmethod
    def lc_2742_2(cost: List[int], time: List[int]) -> int:

        # 模板：经典剪枝DP，可以转换为01背包求解
        n = len(cost)
        dp = [sum(time)] * (n + 1)
        dp[0] = 0
        for i in range(n):
            c, t = cost[i], time[i]
            for j in range(n, -1, -1):
                s = j - t - 1 if j - t - 1 >= 0 else 0  # 此时表示付费油漆匠刷的时候免费的油漆匠不一定要刷满t
                if dp[s] + c < dp[j]:
                    dp[j] = dp[s] + c
        return dp[-1]

    @staticmethod
    def lc_2585(target: int, types: List[List[int]]) -> int:
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

    @staticmethod
    def ac_6(ac=FastIO()):
        # 模板：单调队列优化的多重背包问题，即限定个数和体积价值求最大值
        n, m = ac.read_ints()
        dp = [0]*(m+1)
        for _ in range(n):
            # 体积 价值 数量
            v, w, s = ac.read_ints()
            for r in range(v):
                stack = deque()
                for i in range(r, m+1, v):
                    while stack and stack[0][0] < i-s*v:
                        stack.popleft()
                    while stack and stack[-1][1] + (i - stack[-1][0]) // v * w <= dp[i]:
                        stack.pop()
                    stack.append([i, dp[i]])
                    dp[i] = stack[0][1] + (i-stack[0][0])//v*w
        ac.st(dp[-1])
        return

    @staticmethod
    def ac_10(ac=FastIO()):

        # 模板：树上背包
        n, m = ac.read_ints()
        vol = []
        weight = []
        parent = [-1] * n
        dct = [[] for _ in range(n)]
        root = 0
        for i in range(n):
            v, w, p = ac.read_ints()
            p -= 1
            parent[i] = p
            if p != -2:
                dct[p].append(i)
            else:
                root = i
            vol.append(v)
            weight.append(w)

        # 树上背包
        stack = [root]
        sub = [[0] * (m + 1) for _ in range(n)]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in dct[i]:
                    stack.append(j)
            else:
                i = ~i
                sub[i][vol[i]] = weight[i]
                for j in dct[i]:
                    cur = sub[i][:]
                    for x in range(vol[i], m + 1):  # 必须选择父节点的物品
                        for y in range(m + 1 - x):
                            cur[x + y] = max(cur[x + y], sub[i][x] + sub[j][y])
                    sub[i] = cur[:]
        ac.st(max(sub[root]))
        return

    @staticmethod
    def ac_11(ac=FastIO()):
        # 模板：01背包求方案数
        n, m = ac.read_ints()
        dp = [0]*(m+1)
        cnt = [1]*(m+1)  # 注意方案数都初始化为1
        mod = 10**9 + 7
        for _ in range(n):
            v, w = ac.read_ints()
            for i in range(m, v-1, -1):
                if dp[i-v] + w > dp[i]:
                    dp[i] = dp[i-v] + w
                    cnt[i] = cnt[i-v]
                elif dp[i-v] + w == dp[i]:
                    cnt[i] += cnt[i-v]
                    cnt[i] %= mod
        ac.st(cnt[-1])
        return

    @staticmethod
    def ac_12_1(ac=FastIO()):
        # 模板：01背包求具体方案
        n, m = ac.read_ints()
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        nums = [ac.read_list_ints() for _ in range(n)]

        # 要求字典序最小所以倒着来
        for i in range(n - 1, -1, -1):
            v, w = nums[i]
            for j in range(m, -1, -1):
                dp[i][j] = dp[i + 1][j]
                if j >= v and dp[i + 1][j - v] + w > dp[i][j]:
                    dp[i][j] = dp[i + 1][j - v] + w

        # 再正着求最小的字典序
        j = m
        path = []
        for i in range(n):
            v, w = nums[i]
            if j >= v and dp[i][j] == dp[i + 1][j - v] + w:
                j -= v
                path.append(i + 1)
        ac.lst(path)
        return

    @staticmethod
    def ac_12_2(ac=FastIO()):
        # 模板：01背包求具体方案
        n, m = ac.read_ints()
        dp = [[0, [-1]] for _ in range(m+1)]
        for ind in range(n):
            v, w = ac.read_ints()
            for i in range(m, v-1, -1):
                if dp[i-v][0] + w > dp[i][0] or (dp[i-v][0] + w == dp[i][0] and dp[i-v][1]+[ind+1] < dp[i][1]):
                    dp[i] = [dp[i-v][0] + w, dp[i-v][1]+[ind+1]]
        ac.lst(dp[-1][1][1:])
        return

    @staticmethod
    def lg_p1064(ac=FastIO()):
        # 模板：有依赖的分组背包
        n, m = ac.read_ints()
        dct = [[] for _ in range(m)]
        sub = [[] for _ in range(m)]
        for i in range(m):
            v, p, q = ac.read_ints()
            if q == 0:
                dct[i].append([v, p])
            else:
                sub[q - 1].append([v, p])
        dp = [[0] * (n + 1) for _ in range(2)]
        pre = 0
        for i in range(m):
            if dct[i]:
                cur = 1 - pre
                dp[cur] = dp[pre][:]
                x = len(sub[i])
                for j in range(1 << x):
                    lst = dct[i] + [sub[i][k] for k in range(x) if j & (1 << k)]
                    gain = sum(v * p for v, p in lst)
                    cost = sum(v for v, _ in lst)
                    for xx in range(n, cost - 1, -1):
                        dp[cur][xx] = ac.max(
                            dp[cur][xx], dp[pre][xx - cost] + gain)
                pre = cur
        ac.st(dp[pre][-1])
        return

    @staticmethod
    def lg_p1156(ac=FastIO()):
        # 模板：变形背包
        n, m = ac.read_ints()

        dct = [ac.read_list_ints() for _ in range(m)]
        dct.sort(key=lambda it: it[0])

        dp = [-inf]*(n+1)   # dp[height]=life 到达该高度后剩余的生命值
        dp[0] = 10
        for t, f, h in dct:
            if dp[0] < t:
                ac.st(dp[0])
                return
            for i in range(n, -1, -1):
                if dp[i] >= t:
                    if i+h >= n:
                        ac.st(t)
                        return
                    # 不吃
                    if i+h <= n:
                        dp[i+h] = ac.max(dp[i+h], dp[i])
                    # 吃掉
                    dp[i] += f
        ac.st(dp[0])
        return

    @staticmethod
    def lg_p1273(ac=FastIO()):
        # 模板：树上分组背包
        n, m = ac.read_ints()
        dct = [[] for _ in range(n)]
        for j in range(n-m):
            lst = ac.read_list_ints()
            for i in range(1, len(lst), 2):
                # 边的成本
                dct[j].append([lst[i]-1, lst[i+1]])
        # 节点收益
        nums = [0]*(n-m) + ac.read_list_ints()
        sub = [[] for _ in range(n)]
        stack = [0]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j, _ in dct[i]:
                    stack.append(j)
            else:
                i = ~i
                # sub[i][j]表示人数为j时的最大收益
                sub[i].append(0)
                if i >= n-m:
                    sub[i].append(nums[i])
                    continue

                for j, cost in dct[i]:
                    cur = sub[i][:]
                    for k1 in range(m+1):
                        if k1 >= len(sub[i]):
                            break
                        for k2 in range(m-k1+1):
                            if k2 >= len(sub[j]):
                                break
                            if len(cur) < k1+k2+1:
                                cur.extend([-inf]*(k1+k2+1-len(cur)))
                            # 左边k1右边k2个用户时聚拢的最大收益
                            cur[k1+k2] = ac.max(cur[k1+k2], sub[j][k2]+sub[i][k1]-cost)
                    sub[j] = []
                    sub[i] = cur[:]
        for x in range(m, -1, -1):
            if x < len(sub[0]) and sub[0][x] >= 0:
                ac.st(x)
                return
        ac.st(0)
        return

    @staticmethod
    def lg_p1284(ac=FastIO()):

        # 模板：枚举三角形两边作为二维bool背包
        n = ac.read_int()

        def check():
            # 三角形面积计算公式
            ss = (a + b + c) / 2
            return (ss * (ss - a) * (ss - b) * (ss - c)) ** 0.5

        nums = []
        while len(nums) < n:
            nums.extend(ac.read_list_ints())

        # 二维背包 dp[i][j] 表示能否凑成两条便分别为 i 和 j
        s = sum(nums)
        dp = [[0] * (s // 2 + 1) for _ in range(s // 2 + 1)]
        dp[0][0] = 1
        for num in nums:
            for i in range(s // 2, -1, -1):
                for j in range(s // 2, -1, -1):
                    if j >= num and dp[i][j - num]:
                        dp[i][j] = 1
                    if i >= num and dp[i - num][j]:
                        dp[i][j] = 1
        ans = -1
        for a in range(s // 2 + 1):
            for b in range(s // 2 + 1):
                if dp[a][b]:
                    c = s - a - b
                    # 三角形合法判断公式
                    if b + c > a > 0 and a + c > b > 0 and a + b > c > 0:
                        cur = check()
                        ans = ac.max(ans, cur)
        if ans == -1:
            ac.st(ans)
        else:
            ac.st(int(ans * 100))
        return

    @staticmethod
    def lg_p1441(ac=FastIO()):
        # 模板：枚举加背包DP
        n, m = ac.read_ints()
        a = ac.read_list_ints()
        ans = 0
        s = sum(a)
        for item in combinations(a, n-m):
            dp = [0]*(s+1)
            dp[0] = 1
            for num in item:
                for i in range(s, num-1, -1):
                    if dp[i-num]:
                        dp[i] = 1
            cur = sum(dp)-1
            ans= ac.max(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1537(ac=FastIO()):
        # 模板：经典问题二进制背包优化bool背包，划分成和相等的两部分
        case = 0
        while True:
            lst = ac.read_list_ints()
            if sum(lst) == 0:
                break

            case += 1
            ac.st(f"Collection #{case}:")
            s = sum(lst[i]*(i+1) for i in range(6))
            if s % 2:
                ac.st("Can't be divided.")
                ac.st("")
                continue

            m = s//2
            dp = [0] * (m + 1)
            dp[0] = 1
            for x in range(6):
                w, s = x+1, lst[x]
                if s:
                    for num in BagDP().bin_split(s):
                        for i in range(m, w*num-1, -1):
                            if dp[i-num*w]:
                                dp[i] = 1
            if dp[-1]:
                ac.st("Can be divided.")
            else:
                ac.st("Can't be divided.")
            ac.st("")
        return

    @staticmethod
    def lg_p1541(ac=FastIO()):

        # 模板：四维背包
        n, m = ac.read_ints()
        nums = ac.read_list_ints()
        cnt = Counter(ac.read_list_ints())
        a, b, c, d = cnt[1], cnt[2], cnt[3], cnt[4]
        dp = [[[[0] * (d + 1) for _ in range(c + 1)] for _ in range(b + 1)] for _ in range(a + 1)]
        dp[0][0][0][0] = nums[0]
        ans = 0
        for i in range(a + 1):
            for j in range(b + 1):
                for k in range(c + 1):
                    for p in range(d + 1):
                        if i + 2 * j + 3 * k + 4 * p <= n - 1:
                            pre = 0
                            if i:
                                pre = ac.max(pre, dp[i - 1][j][k][p])
                            if j:
                                pre = ac.max(pre, dp[i][j - 1][k][p])
                            if k:
                                pre = ac.max(pre, dp[i][j][k - 1][p])
                            if p:
                                pre = ac.max(pre, dp[i][j][k][p - 1])
                            dp[i][j][k][p] = ac.max(dp[i][j][k][p], pre + nums[i + 2 * j + 3 * k + 4 * p])
                        if i + 2 * j + 3 * k + 4 * p == n - 1:
                            ans = ac.max(ans, dp[i][j][k][p])
        ac.st(ans)
        return

    @staticmethod
    def lg_p1759(ac=FastIO()):
        # 模板：二维背包输出字典序最小的方案
        m, v, n = ac.read_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        dp = [[[0, []] for _ in range(v + 1)] for _ in range(m + 1)]
        # 同时记录时间与字典序最小的方案
        for i in range(n):
            a, b, c = nums[i]
            for j in range(m, a - 1, -1):
                for k in range(v, b - 1, -1):
                    t, p = dp[j - a][k - b]
                    if dp[j][k][0] < t + c or (dp[j][k][0] == t + c and p + [i + 1] < dp[j][k][1]):
                        dp[j][k] = [t + c, p + [i + 1]]
        ans1, ans2 = dp[m][v]
        ac.st(ans1)
        ac.lst(ans2)
        return

    @staticmethod
    def lg_p1776(ac=FastIO()):
        # 模板：单调队列优化的多重背包问题，即限定个数和体积价值求最大值
        n, m = ac.read_ints()
        dp = [0]*(m+1)
        for _ in range(n):
            a, b, c = ac.read_ints()
            # 体积 价值 数量
            v, w, s = b, a, c
            for r in range(v):
                stack = deque()
                for i in range(r, m+1, v):
                    while stack and stack[0][0] < i-s*v:
                        stack.popleft()
                    while stack and stack[-1][1] + (i - stack[-1][0]) // v * w <= dp[i]:
                        stack.pop()
                    stack.append([i, dp[i]])
                    dp[i] = stack[0][1] + (i-stack[0][0])//v*w
        ac.st(dp[-1])
        return

    @staticmethod
    def lg_p1799(ac=FastIO()):
        # 模板：典型二维矩阵DP
        n = ac.read_int()
        if not n:
            ac.st(0)
            return
        nums = ac.read_list_ints()
        dp = [[-inf] * (n + 1) for _ in range(n)]
        dp[0][0] = 0
        dp[0][1] = 1 if nums[0] == 1 else 0
        for i in range(1, n):
            dp[i][0] = 0
            for j in range(1, i + 2):
                # 前i个数取j个的最大得分
                dp[i][j] = ac.max(dp[i - 1][j], dp[i - 1][j - 1] + int(nums[i] == j))
        ac.st(max(dp[n - 1]))
        return

    @staticmethod
    def lg_p1833(ac=FastIO()):

        def check(st):
            hh, mm = st.split(":")
            return int(hh) * 60 + int(mm)

        # 模板：完全背包与单点队列优化多重背包组合
        s, e, n = ac.read_list_strs()
        t = check(e) - check(s)
        dp = [0] * (t + 1)
        for _ in range(int(n)):
            tt, cc, p = ac.read_ints()
            if not p:
                for i in range(tt, t + 1):
                    dp[i] = ac.max(dp[i], dp[i - tt] + cc)
            else:
                v, w, s = tt, cc, p
                for r in range(v):
                    stack = deque()
                    for i in range(r, t + 1, v):
                        while stack and stack[0][0] < i - s * v:
                            stack.popleft()
                        while stack and stack[-1][1] + (i - stack[-1][0]) // v * w <= dp[i]:
                            stack.pop()
                        stack.append([i, dp[i]])
                        dp[i] = stack[0][1] + (i - stack[0][0]) // v * w
        ac.st(dp[-1])
        return

    @staticmethod
    def lg_p2014(ac=FastIO()):
        # 模板：增加一个虚拟源点将DAG转换为树上背包
        n, m = ac.read_ints()
        dct = [[] for _ in range(n + 1)]
        nums = [0]
        for i in range(n):
            k, s = ac.read_ints()
            nums.append(s)
            dct[k].append(i + 1)
        dp = [[0] * (m + 2) for _ in range(n + 1)]
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                dp[i][1] = nums[i]
                for j in dct[i]:
                    if j != fa:
                        cur = dp[i][:]
                        for x in range(m + 2):
                            for y in range(m + 2 - x):
                                cur[x + y] = ac.max(cur[x + y], dp[i][x] + dp[j][y])
                        dp[i] = cur[:]
        ac.st(dp[0][m + 1])
        return

    @staticmethod
    def lg_p2079(ac=FastIO()):
        # 模板：滚动哈希背包DP，使用两层哈希节省空间
        n, v = ac.read_ints()
        dp = [defaultdict(lambda: defaultdict(lambda: -inf)), defaultdict(lambda: defaultdict(lambda: -inf))]
        pre = 0
        dp[pre][0][0] = 0
        for i in range(n):
            c, x, y = ac.read_ints()
            cur = 1-pre
            for c1 in dp[pre]:
                for x1 in dp[pre][c1]:
                    if c1+c<=v:
                        dp[cur][c1+c][x1+x] = ac.max(dp[cur][c1+c][x1+x], dp[pre][c1][x1]+y)
                    dp[cur][c1][x1] = ac.max(dp[cur][c1][x1], dp[pre][c1][x1])
            pre = cur
        ans = -inf
        for c1 in dp[pre]:
            for x1 in dp[pre][c1]:
                if x1 >= 0:
                    ans = ac.max(ans, dp[pre][c1][x1])
        ac.st(ans)
        return

    @staticmethod
    def lg_p2170(ac=FastIO()):
        # 模板：连通块加二进制01背包优化
        n, m, k = ac.read_ints()
        uf = UnionFind(n)
        for _ in range(k):
            i, j = ac.read_ints_minus_one()
            uf.union(i, j)
        dct = defaultdict(int)
        for i in range(n):
            dct[uf.find(i)] += 1
        lst = list(dct.values())
        del uf

        # 使用二进制优化的01背包
        target = ac.min(2 * m, n)
        dp = [0] * (target + 1)
        dp[0] = 1
        cnt = Counter(lst)
        for num in cnt:
            for x in BagDP().bin_split(cnt[num]):
                for i in range(target, x * num - 1, -1):
                    if dp[i - x * num]:
                        dp[i] = 1
        ans = 0
        for i in range(1, target + 1):
            if dp[i] and abs(i - m) < abs(ans - m):
                ans = i
        ac.st(ans)
        return

    @staticmethod
    def lg_p2214(ac=FastIO()):
        # 模板：变种背包DP贪心
        n, b = ac.read_ints()
        nums = [ac.read_int() for _ in range(b)]
        voice = [ac.read_int() for _ in range(n)]

        # 从后往前计算原始得分
        for i in range(n - 1, 0, -1):
            if voice[i - 1] > 0:
                voice[i] -= voice[i - 1] - 1
        ceil = max(voice)
        if any(v < 0 for v in voice):
            ac.st(-1)
            return

        # 完全背包计算最少数量
        dp = [inf] * (ceil + 1)
        dp[0] = 0
        for num in nums:
            for i in range(num, ceil + 1):
                dp[i] = ac.min(dp[i - num] + 1, dp[i])
        ans = sum(dp[x] for x in voice)
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def lg_p2306(ac=FastIO()):
        # 模板：根据数据范围计数后进行二进制优化的01背包计算
        n, m, k = ac.read_ints()
        cnt = defaultdict(lambda: defaultdict(int))
        for _ in range(n):
            a, b = ac.read_ints()
            cnt[a][b] += 1
        dp = [0] * (m + 1)
        for a in cnt:
            for b in cnt[a]:
                for x in BagDP().bin_split(cnt[a][b]):
                    for i in range(m, x * a - 1, -1):
                        dp[i] = ac.max(dp[i], dp[i - x * a] + x * b)
        ans = max(dp)
        if ans >= k:
            ac.st("yes")
        else:
            ac.st("no")
        ac.st(ans)
        return

    @staticmethod
    def lg_p2320(ac=FastIO()):
        # 模板：二进制分解贪心反向计算
        m = ac.read_int()
        ans = []
        while m:
            ans.append((m + 1) // 2)
            m //= 2
        ac.st(len(ans))
        ac.st(*ans[::-1])
        return

    @staticmethod
    def lg_p2737(ac=FastIO()):
        # 模板：完全背包变种问题
        n = ac.read_int()
        ceil = 256**2 + 1
        nums = [ac.read_int() for _ in range(n)]
        dp = [0] * (ceil + 1)
        dp[0] = 1
        for i in range(1, ceil + 1):
            for num in nums:
                if i >= num and dp[i - num]:
                    dp[i] = 1
        ans = 0
        for i in range(1, ceil + 1):
            if not dp[i]:
                ans = i
        ac.st(ans if ans < ceil else 0)
        return

    @staticmethod
    def lg_p2760(ac=FastIO()):
        # 模板：单调队列优化的多重背包
        m, n, p, t = ac.read_ints()
        rest = ac.min(p, t - 1)
        dp = [0] * (rest + 1)
        grid = [ac.read_list_ints() for _ in range(m)]
        mat = [ac.read_list_ints() for _ in range(m)]
        for a in range(m):
            for b in range(n):
                if grid[a][b]:
                    v, w, s = (a + 1 + b + 1) * 2, grid[a][b], mat[a][b]
                    for r in range(v):
                        stack = deque()
                        for i in range(r, rest + 1, v):
                            while stack and stack[0][0] < i - s * v:
                                stack.popleft()
                            while stack and stack[-1][1] + (i - stack[-1][0]) // v * w <= dp[i]:
                                stack.pop()
                            stack.append([i, dp[i]])
                            dp[i] = stack[0][1] + (i - stack[0][0]) // v * w
        ac.st(dp[-1])
        return

    @staticmethod
    def lg_p2854(ac=FastIO()):
        # 模板：分组01背包
        length, n, b = ac.read_ints()
        dp = [[-inf] * (b + 1) for _ in range(length + 1)]
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort()
        for x, w, f, c in nums:
            if x == 0:
                if c <= b:
                    dp[x + w][c] = ac.max(dp[x + w][c], f)
            else:
                for i in range(b + 1):
                    if i + c <= b and x + w <= length:
                        dp[x + w][i + c] = ac.max(dp[x + w][i + c], dp[x][i] + f)
        ans = max(dp[length])
        ac.st(ans if ans > -inf else -1)
        return

    @staticmethod
    def lg_p2938(ac=FastIO()):
        # 模板：分组完全背包
        s, d, m = ac.read_ints()
        nums = [ac.read_list_ints() for _ in range(s)]
        for i in range(1, d):
            dp = [0] * (m + 1)
            for j in range(s):
                a, b = nums[j][i - 1], nums[j][i]
                if b > a:
                    for p in range(a, m + 1):
                        dp[p] = ac.max(dp[p], dp[p - a] + b)
            m = max(m - i + dp[i] for i in range(m + 1))
        ac.st(m)
        return

    @staticmethod
    def lg_p2979(ac=FastIO()):
        # 模板：分组01背包
        n, t, k = ac.read_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        m = 5*t//4 + 1

        # 先算不缩减高度的
        dp1 = [0] * (m + 1)
        for v, h in nums:
            for i in range(h, m + 1):
                if dp1[i - h] + v > dp1[i]:
                    dp1[i] = dp1[i - h] + v
        ans = dp1[t]
        # 枚举最后一棒高度大于等于 k 的
        for v, h in nums:
            if h >= k:
                for i in range(t, h - 1, -1):
                    ans = ac.max(ans, dp1[(i - h)*5//4] + v)
        ac.st(ans)
        return

    @staticmethod
    def lg_p3010(ac=FastIO()):
        # 模板：变形01背包，计算两堆差值最小的分配方案数
        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]
        s = sum(nums)
        mod = 10 ** 6
        # 临界点作为目标值
        t = s // 2
        dp = [0] * (t + 1)  # 背包
        dp[0] = 1
        cnt = [0] * (t + 1)  # 方案数
        cnt[0] = 1
        for num in nums:
            for i in range(t, num - 1, -1):
                if dp[i - num]:
                    dp[i] = 1
                    cnt[i] += cnt[i - num]
                    cnt[i] %= mod

        # 枚举最小差值
        for i in range(t, -1, -1):
            if dp[i]:
                ac.st(s - 2 * i)
                ac.st(cnt[i])
                break
        return

    @staticmethod
    def lg_p3423(ac=FastIO()):
        # 模板：二进制优化多重背包并计算方案数
        n = ac.read_int()
        b = ac.read_list_ints()
        c = ac.read_list_ints()
        k = ac.read_int()
        dp = [inf] * (k + 1)
        dp[0] = 0
        state = [[] for _ in range(k + 1)]
        for j in range(n):
            bb, cc = b[j], c[j]
            for x in BagDP().bin_split(cc):
                for i in range(k, x * bb - 1, -1):
                    if dp[i - x * bb] + x < dp[i]:
                        dp[i] = dp[i - x * bb] + x
                        state[i] = state[i - x * bb][:] + [[bb, x]]
        cnt = defaultdict(int)
        for bb, xx in state[k]:
            cnt[bb] += xx
        ac.st(dp[k])
        ac.lst([cnt[bb] for bb in b])
        return

    @staticmethod
    def lg_p3983(ac=FastIO()):
        # 模板：两个分组完全背包计算
        n = ac.read_int()
        # 第一个背包计算每个重量可拆分后的最大价格
        m = 10
        a = [0] + ac.read_list_ints()
        for i in range(1, m + 1):
            for j in range(i + 1):
                a[i] = ac.max(a[i], a[j] + a[i - j])
        # 第二个背包计算船运载的最大盈利
        cost = [0] + [1, 3, 5, 7, 9, 10, 11, 14, 15, 17]
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(i, n + 1):
                dp[j] = ac.max(dp[j], dp[j - i] + a[i] - cost[i])
        ac.st(dp[-1])
        return

    @staticmethod
    def lg_p5322(ac=FastIO()):
        # 模板：典型二维 DP 转换为分组背包
        s, n, m = ac.read_ints()
        grid = [ac.read_list_ints() for _ in range(s)]
        dp = [0] * (m + 1)
        for i in range(n):
            # dp[j] 表示派出 j 个士兵到前 i 个城堡的得分
            lst = [grid[x][i] for x in range(s)]
            lst.sort()
            for j in range(m, -1, -1):
                for ind, w in enumerate(lst):
                    if j <= w * 2:
                        break
                    dp[j] = ac.max(dp[j], dp[j - 2 * w - 1] + (ind + 1) * (i + 1))
        ac.st(dp[-1])
        return

    @staticmethod
    def lg_p5365(ac=FastIO()):
        # 模板：01背包 DP 枚举数量
        n, m = ac.read_ints()
        kk = ac.read_list_ints()
        cc = ac.read_list_ints()
        s = sum(kk[i] * cc[i] for i in range(n))
        dp = [0] * (s + 1)
        dp[0] = 1
        for i in range(n):
            k, c = kk[i], cc[i]
            for x in range(s, -1, -1):
                for p in range(1, k + 1):
                    if x < p * c:
                        break
                    dp[x] = ac.max(dp[x], dp[x - p * c] * p)
        for i in range(s + 1):
            if dp[i] >= m:
                ac.st(i)
                break
        return

    @staticmethod
    def lg_p5662(ac=FastIO()):
        # 模板：完全背包变形贪心题目
        t, n, m = ac.read_ints()
        grid = [ac.read_list_ints() for _ in range(t)]
        for i in range(1, t):
            dp = [0] * (m + 1)
            for j in range(n):
                b, a = grid[i][j], grid[i - 1][j]
                if b > a:
                    for x in range(a, m + 1):
                        dp[x] = ac.max(dp[x], dp[x - a] + b)
            # 注意此时的 m 更新值
            m = max(m - i + dp[i] for i in range(m + 1))
        ac.st(m)
        return

    @staticmethod
    def lg_p1417(ac=FastIO()):
        # 模板：经典贪心排序后计算 01 背包最大值
        t, n = ac.read_ints()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        c = ac.read_list_ints()
        dp = [0]*(t+1)
        ind = list(range(n))
        ind.sort(key=lambda it: -b[it]/c[it])
        for i in ind:
            aa, bb, cc = a[i], b[i], c[i]
            for j in range(t, cc-1, -1):
                dp[j] = ac.max(dp[j], dp[j-cc]+aa-j*bb)
        ac.st(max(dp))
        return

    @staticmethod
    def ac_4081(ac=FastIO()):
        # 模板：经典矩阵DP类似背包思想

        n, k = ac.read_ints()
        nums = ac.read_list_ints()

        def check2(x):
            res = 0
            while x % 2 == 0:
                res += 1
                x //= 2
            return res

        def check5(x):
            res = 0
            while x % 5 == 0:
                res += 1
                x //= 5
            return res

        cnt2 = [check2(num) for num in nums]
        cnt5 = [check5(num) for num in nums]

        s5 = sum(cnt5)
        dp = [[-inf] * (s5 + 1) for _ in range(k + 1)]
        dp[0][0] = 0
        for i in range(n):
            a2 = cnt2[i]
            a5 = cnt5[i]
            for j in range(k, 0, -1):
                for p in range(s5, a5 - 1, -1):
                    x, y = dp[j][p], dp[j - 1][p - a5] + a2
                    if y > x:
                        dp[j][p] = y
        ans = 0
        for a5 in range(s5 + 1):
            cur = ac.min(dp[k][a5], a5)
            if cur > ans:
                ans = cur
        ac.st(ans)
        return

    @staticmethod
    def lc_1049(stones: List[int]) -> int:
        # 模板：经典问题，转化为01背包求解
        s = sum(stones)
        dp = [0]*(s//2 + 1)
        dp[0] = 1
        for num in stones:
            for i in range(s//2, num-1, -1):
                if dp[i-num]:
                    dp[i] = 1
        return min(abs(s-2*i) for i in range(s//2 + 1) if dp[i])


class TestGeneral(unittest.TestCase):

    def test_bag_dp(self):
        bd = BagDP()
        for _ in range(1000):
            num = random.randint(1, 100000000)
            assert sum(bd.bin_split(num)) == num
        return


if __name__ == '__main__':
    unittest.main()
