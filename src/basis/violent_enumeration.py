import bisect
import unittest
import math
from collections import defaultdict, deque
from functools import reduce
from itertools import combinations, permutations
from operator import mul
from typing import List

from src.fast_io import FastIO, inf


"""
算法：暴力枚举、旋转矩阵、螺旋矩阵（也叫brute_force）、贡献法
功能：根据题意，在复杂度有限的情况下，进行所有可能情况的枚举
题目：

===================================力扣===================================
670. 最大交换（https://leetcode.cn/problems/maximum-swap/）看似贪心，在复杂度允许的情况下使用枚举暴力保险
395. 至少有 K 个重复字符的最长子串（https://leetcode.cn/problems/longest-substring-with-at-least-k-repeating-characters/）经典枚举分治
1330. 翻转子数组得到最大的数组值（https://leetcode.cn/problems/reverse-subarray-to-maximize-array-value/）经典枚举绝对值正负数
2488. 统计中位数为 K 的子数组（https://leetcode.cn/problems/count-subarrays-with-median-k/）利用中位数的定义枚举前后子序列中满足大于 K 和小于 K 的数个数相等的子数组
2484. 统计回文子序列数目（https://leetcode.cn/problems/count-palindromic-subsequences/）利用前后缀哈希计数枚举当前索引作为回文中心的回文子串的前后缀个数
2322. 从树中删除边的最小分数（https://leetcode.cn/problems/minimum-score-after-removals-on-a-tree/）枚举删除的第一条边后使用树形递归再枚举第二条边计算连通块异或差值最小分数
2321. 拼接数组的最大分数（https://leetcode.cn/problems/maximum-score-of-spliced-array/）借助经典的最大最小子数组和枚举交换的连续子序列
2306. 公司命名（https://leetcode.cn/problems/naming-a-company/）利用字母个数有限的特点枚举首字母交换对
2272. 最大波动的子字符串（https://leetcode.cn/problems/substring-with-largest-variance/）利用字母个数有限的特点进行最大字符与最小字符枚举
2183. 统计可以被 K 整除的下标对数目（https://leetcode.cn/problems/count-array-pairs-divisible-by-k/）按照最大公约数进行分组枚举
2151. 基于陈述统计最多好人数（https://leetcode.cn/problems/maximum-good-people-based-on-statements/）使用状态压缩进行枚举并判定是否合法
2147. 分隔长廊的方案数（https://leetcode.cn/problems/number-of-ways-to-divide-a-long-corridor/）根据题意进行分隔点枚举计数
2122. 还原原数组（https://leetcode.cn/problems/recover-the-original-array/）枚举差值 k 判断是否合法
2468 根据限制分割消息（https://leetcode.cn/problems/split-message-based-on-limit/）根据长度限制进行二分
2417. 最近的公平整数（https://leetcode.cn/problems/closest-fair-integer/）按照位数贪心枚举加和
2681. 英雄的力量（https://leetcode.cn/problems/power-of-heroes/）按照贡献法枚举计数
1625. 执行操作后字典序最小的字符串（https://leetcode.cn/problems/lexicographically-smallest-string-after-applying-operations/）经典枚举计算最小的字典序
1819. 序列中不同最大公约数的数目（https://leetcode.cn/problems/number-of-different-subsequences-gcds/）经典调和级数枚举
1862. 向下取整数对和（https://leetcode.cn/submissions/detail/371754298/）枚举商并利用调和级数的复杂度进行计算
2014. 重复 K 次的最长子序列（https://leetcode.cn/problems/longest-subsequence-repeated-k-times/）经典利用数据范围进行枚举
2077. 殊途同归（https://leetcode.cn/problems/paths-in-maze-that-lead-to-same-room/）经典使用位运算枚举
2081. k 镜像数字的和（https://leetcode.cn/problems/sum-of-k-mirror-numbers/）经典回文串进制数据枚举

===================================洛谷===================================
P1548 棋盘问题（https://www.luogu.com.cn/problem/P1548）枚举正方形与长方形的右小角计算个数
P1632 点的移动（https://www.luogu.com.cn/problem/P1632）枚举横坐标和纵坐标的所有组合移动距离
P2128 赤壁之战（https://www.luogu.com.cn/problem/P2128）枚举完全图的顶点组合，平面图最多四个点
P2191 小Z的情书（https://www.luogu.com.cn/problem/P2191）逆向思维旋转矩阵
P2699 【数学1】小浩的幂次运算（https://www.luogu.com.cn/problem/P2699）分类讨论、暴力枚举模拟
P1371 NOI元丹（https://www.luogu.com.cn/problem/P1371）前后缀枚举计数
P1369 矩形（https://www.luogu.com.cn/problem/P1369）矩阵DP与有贪心枚举在矩形边上的最多点个数
P1158 [NOIP2010 普及组] 导弹拦截（https://www.luogu.com.cn/problem/P1158）按照距离的远近进行预处理排序枚举，使用后缀最大值进行更新
P8928 「TERRA-OI R1」你不是神，但你的灵魂依然是我的盛宴（https://www.luogu.com.cn/problem/P8928）枚举取模计数的组合
P8892 「UOI-R1」磁铁（https://www.luogu.com.cn/problem/P8892）枚举字符串的分割点，经典查看一个字符串是否为另一个字符串的子序列
P8799 [蓝桥杯 2022 国 B] 齿轮（https://www.luogu.com.cn/problem/P8799）暴力枚举是否有组合相除得到查询值
P3142 [USACO16OPEN]Field Reduction S（https://www.luogu.com.cn/problem/P3142）暴力枚举临近左右上下边界的点进行最小面积计算
P3143 [USACO16OPEN] Diamond Collector S（https://www.luogu.com.cn/problem/P3143）枚举前缀与后缀序列的最多个数相加
P3670 [USACO17OPEN]Bovine Genomics S（https://www.luogu.com.cn/problem/P3670）哈希枚举计数
P3799 妖梦拼木棒（https://www.luogu.com.cn/problem/P3799）枚举正三角形的木棒边长
P3910 纪念邮票（https://www.luogu.com.cn/problem/P3910）结合因数分解枚举可行的连续数组和为目标数字
P4086 [USACO17DEC]My Cow Ate My Homework S（https://www.luogu.com.cn/problem/P4086）利用后缀进行倒序枚举
P4596 [COCI2011-2012#5] RAZBIBRIGA（https://www.luogu.com.cn/problem/P4596）枚举可行的正方形单词与方案数
P4759 [CERC2014]Sums（https://www.luogu.com.cn/problem/P4759）结合因数分解枚举可行的连续数组和为目标数字
P6267 [SHOI2002]N的连续数拆分（https://www.luogu.com.cn/problem/P6267）结合因数分解枚举可行的连续数组和为目标数字
P5077 Tweetuzki 爱等差数列（https://www.luogu.com.cn/problem/P5077）结合因数分解枚举可行的连续数组和为目标数字
P4960 血小板与凝血因子（https://www.luogu.com.cn/problem/P4960）按照题意模拟枚举
P4994 终于结束的起点（https://www.luogu.com.cn/problem/P4994）暴力模拟，皮萨诺周期可以证明pi(n)<=6n
P5190 [COCI2009-2010#5] PROGRAM（https://www.luogu.com.cn/problem/P5190）使用埃氏筛的思想进行计数与前缀和计算查询，复杂度为调和级数O(nlogn)
P5614 [MtOI2019]膜Siyuan（https://www.luogu.com.cn/problem/P5614）根据题意枚举其中两个数，计算满足条件的另一个数的个数
P6014 [CSGRound3]斗牛（https://www.luogu.com.cn/problem/P6014）使用哈希计算整体取模与每个单个数字确定互补取模计数
P6067 [USACO05JAN]Moo Volume S（https://www.luogu.com.cn/problem/P6067）排序后使用前后缀和进行枚举计算
P6248 准备战斗，选择你的英雄（https://www.luogu.com.cn/problem/P6248）计算可能性进行暴力枚举
P6355 [COCI2007-2008#3] DEJAVU（https://www.luogu.com.cn/problem/P6355）枚举直角三角形的直角点进行计数
P6365 [传智杯 #2 初赛] 众数出现的次数（https://www.luogu.com.cn/problem/P6365）使用容斥原理进行枚举计数
P6439 [COCI2011-2012#6] ZAGRADE（https://www.luogu.com.cn/problem/P6439）枚举删除的位置组合，使用几集合进行去重
P6686 混凝土数学（https://www.luogu.com.cn/problem/P6686）枚举等腰三角形的边长计数
P2666 [USACO07OCT]Bessie's Secret Pasture S（https://www.luogu.com.cn/problem/P2666）枚举计数，计算将n拆解为4个数的平方和的方案数
P2705 小球（https://www.luogu.com.cn/problem/P2705）枚举红色小球放在蓝色盒子的数量计算
P5690 [CSP-S2019 江西] 日期（https://www.luogu.com.cn/problem/P5690）对于日期，典型地暴力进行枚举确认
P7076 [CSP-S2020] 动物园（https://www.luogu.com.cn/problem/P7076）位运算枚举计数
P7094 [yLOI2020] 金陵谣（https://www.luogu.com.cn/problem/P7094）变换公式根据，数据范围进行枚举
P7273 ix35 的等差数列（https://www.luogu.com.cn/problem/P7273）经典公差枚举，计算使得首项相同的个数，贪心选择最佳
P7286 「EZEC-5」人赢（https://www.luogu.com.cn/problem/P7286）排序后枚举最小值，选择最优结果计数
P7626 [COCI2011-2012#1] MATRIX（https://www.luogu.com.cn/problem/P7626）枚举正方形子矩阵的主对角线与副对角线之差
P7799 [COCI2015-2016#6] PIANINO（https://www.luogu.com.cn/problem/P7799）哈希枚举计数
P1018 [NOIP2000 提高组] 乘积最大（https://www.luogu.com.cn/problem/P1018）枚举乘号位置
P1311 [NOIP2011 提高组] 选择客栈（https://www.luogu.com.cn/problem/P1311）线性枚举计数，每次重置避免重复计数
P2119 [NOIP2016 普及组] 魔法阵（https://www.luogu.com.cn/problem/P2119）枚举差值，并计算前后缀个数
P2652 同花顺（https://www.luogu.com.cn/problem/P2652）枚举花色与双指针计算长度
P2994 [USACO10OCT]Dinner Time S（https://www.luogu.com.cn/problem/P2994）按照座位枚举分配人员
P3985 不开心的金明（https://www.luogu.com.cn/problem/P3985）看似背包实则枚举
P4181 [USACO18JAN]Rental Service S（https://www.luogu.com.cn/problem/P4181）贪心枚举与后缀和
P6149 [USACO20FEB] Triangles S（https://www.luogu.com.cn/problem/P6149）经典枚举三角形的直角点使用前缀和与二分计算距离和
P6393 隔离的日子（https://www.luogu.com.cn/problem/P6393）经典利用值域范围进行枚举计算
P6767 [BalticOI 2020/2012 Day0] Roses（https://www.luogu.com.cn/problem/P6767）
P8270 [USACO22OPEN] Subset Equality S（https://www.luogu.com.cn/problem/P8270）经典脑筋急转弯枚举，转换为两两字母比较
P8587 新的家乡（https://www.luogu.com.cn/problem/P8587）桶计数枚举
P8663 [蓝桥杯 2018 省 A] 倍数问题（https://www.luogu.com.cn/problem/P8663）桶计数枚举
P8672 [蓝桥杯 2018 国 C] 交换次数（https://www.luogu.com.cn/problem/P8672）字符串枚举与经典置换环计数
P8712 [蓝桥杯 2020 省 B1] 整数拼接（https://www.luogu.com.cn/problem/P8712）整数长度枚举
P8749 [蓝桥杯 2021 省 B] 杨辉三角形（https://www.luogu.com.cn/problem/P8749）利用杨辉三角形特点进行枚举
P8808 [蓝桥杯 2022 国 C] 斐波那契数组（https://www.luogu.com.cn/problem/P8808）利用斐波那契数组的特点进行枚举
P8809 [蓝桥杯 2022 国 C] 近似 GCD（https://www.luogu.com.cn/problem/P8809）枚举加贡献计数
P9076 [PA2018] PIN（https://www.luogu.com.cn/problem/P9076）根据数字的因数进行枚举
P9008 [入门赛 #9] 大碗宽面 (Hard Version)（https://www.luogu.com.cn/problem/P9008）经典朋友敌人陌生人容斥枚举计数
P9006 [入门赛 #9] 神树大人挥动魔杖 (Hard Version)（https://www.luogu.com.cn/problem/P9006）经典枚举取模计数
P8948 [YsOI2022]NOIp和省选（https://www.luogu.com.cn/problem/P8948）预处理和枚举所有情况
P8894 「UOI-R1」求和（https://www.luogu.com.cn/problem/P8894）按照区间范围值进行枚举前后缀计数
P8872 [传智杯 #5 初赛] D-莲子的物理热力学（https://www.luogu.com.cn/problem/P8872）排序后前后缀移动次数枚举

================================CodeForces================================
https://codeforces.com/problemset/problem/1426/F（分类枚举中间的b计数两边的?ac，并使用快速幂进行求解）
D. Zigzags（https://codeforces.com/problemset/problem/1400/D）枚举+二分
D. Moscow Gorillas（https://codeforces.com/contest/1793/problem/D）枚举计数
D. Dima and Lisa（https://codeforces.com/problemset/problem/584/D）确定一个质数3，枚举第二三个质数，小于 10**9 的任何数都可以分解为最多三个质数的和
D. Three Integers（https://codeforces.com/problemset/problem/1311/D）根据题意，确定一个上限值，贪心枚举
C. Flag（https://codeforces.com/problemset/problem/1181/C）按列进行枚举
B. Maximum Value（https://codeforces.com/problemset/problem/484/B）排序后进行枚举，并使用二分查找进行确认
C. Arithmetic Progression（https://codeforces.com/problemset/problem/382/C）分类讨论


================================Acwing===================================
95. 费解的开关（https://www.acwing.com/problem/content/description/97/）枚举第一行的开关按钮使用状态

参考：OI WiKi（xx）
"""




class ViolentEnumeration:
    def __init__(self):
        return

    @staticmethod
    def matrix_rotate(matrix):  # 旋转矩阵

        # 将矩阵顺时针旋转 90 度
        n = len(matrix)
        for i in range(n // 2):
            for j in range((n + 1) // 2):
                a, b, c, d = matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1], matrix[i][j]
                matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1] = a, b, c, d

        # 将矩阵逆时针旋转 90 度
        n = len(matrix)
        for i in range(n // 2):
            for j in range((n + 1) // 2):
                a, b, c, d = matrix[j][n - i - 1], matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1]
                matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1] = a, b ,c ,d

        return matrix


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1311d(ac=FastIO()):
        # 模板：根据贪心策略 a=b=1 时显然满足条件，因此枚举不会超过这个代价的范围就行
        for _ in range(ac.read_int()):
            a, b, c = ac.read_list_ints()
            ans = inf
            res = []
            for x in range(1, 2 * a + 1):
                for y in range(x, 2 * b + 1, x):
                    if y % x == 0:
                        for z in [(c // y) * y, (c // y) * y + y]:
                            cost = abs(a - x) + abs(b - y) + abs(c - z)
                            if cost < ans:
                                ans = cost
                                res = [x, y, z]
            ac.st(ans)
            ac.lst(res)
        return

    @staticmethod
    def cf_584d(ac=FastIO()):
        # 模板：将 n 分解为最多三个质数的和
        def is_prime4(x):
            if x == 1:
                return False
            if (x == 2) or (x == 3):
                return True
            if (x % 6 != 1) and (x % 6 != 5):
                return False
            for i in range(5, int(math.sqrt(x)) + 1, 6):
                if (x % i == 0) or (x % (i + 2) == 0):
                    return False
            return True

        # 模板：将正整数分解为最多三个质数的和
        n = ac.read_int()
        assert 3 <= n < 10**9

        if is_prime4(n):
            ac.st(1)
            ac.st(n)
            return

        for i in range(2, 10 ** 5):
            j = n - 3 - i
            if is_prime4(i) and is_prime4(j):
                ac.st(3)
                ac.lst([3, i, j])
                return
        return

    @staticmethod
    def lc_670(num: int) -> int:
        # 模板：在复杂度有限的情况下有限采用枚举的方式计算而不是贪心

        def check():  # 贪心
            lst = list(str(num))
            n = len(lst)
            post = list(range(n))
            # 从后往前遍历，对每个数位，记录其往后最大且最靠后的比它大的数位位置，再从前往后交换第一个有更大的靠后值得数位
            j = n - 1
            for i in range(n - 2, -1, -1):
                if lst[i] > lst[j]:
                    j = i
                if lst[j] > lst[i]:
                    post[i] = j

            for i in range(n):
                if post[i] != i:
                    lst[i], lst[post[i]] = lst[post[i]], lst[i]
                    return int("".join(lst))
            return int("".join(lst))

        def check2():  # 枚举
            lst = list(str(num))
            n = len(lst)
            ans = num
            for item in combinations(list(range(n)), 2):
                cur = lst[:]
                i, j = item
                cur[i], cur[j] = cur[j], cur[i]
                x = int("".join(cur))
                ans = ans if ans > x else x
            return ans
        return check2()

    @staticmethod
    def cf_484b(ac=FastIO()):
        # 模板：查询数组中两两取模运算的最大值（要求较小值作为取模数）
        ac.read_int()
        nums = sorted(list(set(ac.read_list_ints())))
        n = len(nums)
        ceil = nums[-1]

        dp = [0] * (ceil + 1)
        i = 0
        for x in range(1, ceil + 1):
            dp[x] = dp[x - 1]
            while i < n and nums[i] <= x:
                dp[x] = nums[i]
                i += 1

        ans = 0
        for num in nums:
            x = 1
            while x * num <= ceil:
                x += 1
                for a in [x * num - 1]:
                    ans = ac.max(ans, dp[ac.min(a, ceil)] % num)
        ac.st(ans)
        return

    @staticmethod
    def cf_382c(ac=FastIO()):

        # 2023年3月29日·灵茶试炼·分类讨论
        n = ac.read_int()
        nums = sorted(ac.read_list_ints())

        # 只有一种情况有无穷多个
        if n == 1:
            ac.st(-1)
            return

        # 计算排序后相邻项差值最大值与最小值以及不同差值
        diff = [nums[i] - nums[i - 1] for i in range(1, n)]
        high = max(diff)
        low = min(diff)
        cnt = len(set(diff))

        # 1. 大于等于3个不同差值显然没有
        if cnt >= 3:
            ac.st(0)
            return
        elif cnt == 2:
            # 2. 有2个不同差值存在合理情况当且仅当 high=2*low 且 count(high)==1
            if high != 2 * low or diff.count(high) != 1:
                ac.st(0)
                return

            for i in range(1, n):
                if nums[i] - nums[i - 1] == high:
                    ac.st(1)
                    ac.st(nums[i - 1] + low)
                    return
        else:
            # 3.有1个差值时分为0与不为0，不为0分 n大于2 与等于2
            if low == high == 0:
                ac.st(1)
                ac.st(nums[0])
                return
            if n == 2:
                if low % 2 == 0:
                    ac.st(3)
                    ac.lst([nums[0] - low, nums[0] + low // 2, nums[1] + low])
                else:
                    ac.st(2)
                    ac.lst([nums[0] - low, nums[1] + low])
            else:
                ac.st(2)
                ac.lst([nums[0] - low, nums[-1] + low])
        return

    @staticmethod
    def ac_95(ac=FastIO()):
        # 模板：枚举第一行状态
        n = ac.read_int()

        for _ in range(n):
            grid = [[int(w) for w in ac.read_str()] for _ in range(5)]
            ac.read_str()

            ans = -1
            for state in range(1 << 5):
                lst = [x for x in range(5) if state & (1 << x)]
                temp = [g[:] for g in grid]
                cur = len(lst)
                for x in lst:
                    i, j = 0, x
                    temp[i][j] = 1 - temp[i][j]
                    for a, b in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                        if 0 <= a < 5 and 0 <= b < 5:
                            temp[a][b] = 1 - temp[a][b]

                for r in range(1, 5):
                    for j in range(5):
                        if temp[r - 1][j] == 0:
                            i, j = r, j
                            temp[i][j] = 1 - temp[i][j]
                            cur += 1
                            for a, b in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                                if 0 <= a < 5 and 0 <= b < 5:
                                    temp[a][b] = 1 - temp[a][b]
                if all(all(x == 1 for x in g) for g in temp):
                    ans = ans if ans < cur and ans != -1 else cur
            ac.st(ans if ans <= 6 else -1)
        return

    @staticmethod
    def lg_p1018(ac=FastIO()):
        # 模板：枚举乘号的位置
        n, k = ac.read_ints()
        nums = ac.read_list_str()

        ans = 0
        for item in combinations(list(range(1, n)), k):
            cur = nums[:]
            for i in item:
                cur[i] = "*"+cur[i]
            res = [int(w) for w in ("".join(cur)).split("*")]
            cur = reduce(mul, res)
            ans = ac.max(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1311(ac=FastIO()):
        # 模板：线性枚举计数，每次重置避免重复计数
        n, k, p = ac.read_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        cnt = [0]*k
        for i in range(n):
            cnt[nums[i][0]] += 1
        pre = [0]*k
        ans = 0
        for i in range(n):
            c = nums[i][0]
            pre[c] += 1
            if nums[i][1] <= p:
                for j in range(k):
                    if j != c:
                        ans += pre[j]*(cnt[j]-pre[j])
                    else:
                        ans += pre[j]-1
                        ans += cnt[j]-pre[j]
                        ans += (pre[j]-1)*(cnt[j]-pre[j])
                    cnt[j] -= pre[j]
                pre = [0]*k
        ac.st(ans)
        return

    @staticmethod
    def lg_p2119(ac=FastIO()):

        # 模板：枚举差值，并计算前后缀个数
        n, m = ac.read_ints()
        nums = [ac.read_int() for _ in range(m)]

        cnt = [0] * (n + 1)
        for num in nums:
            cnt[num] += 1

        aa = [0] * (n + 1)
        bb = [0] * (n + 1)
        cc = [0] * (n + 1)
        dd = [0] * (n + 1)

        # 枚举b-a=x
        for x in range(1, n // 9 + 1):
            if 1 + 9 * x + 1 > n:
                break

            # 前缀ab计数
            pre_ab = [0] * (n + 1)
            for b in range(2 * x + 1, n + 1):
                pre_ab[b] = pre_ab[b - 1]
                pre_ab[b] += cnt[b] * cnt[b - 2 * x]

            # 作为cd
            for c in range(n - x, -1, -1):
                if c - 6 * x - 1 >= 1:
                    cc[c] += pre_ab[c - 6 * x - 1] * cnt[c + x]
                    dd[c + x] += pre_ab[c - 6 * x - 1] * cnt[c]
                else:
                    break

            # 后缀cd
            post_cd = [0] * (n + 2)
            for c in range(n - x, -1, -1):
                post_cd[c] = post_cd[c + 1]
                post_cd[c] += cnt[c] * cnt[c + x]

            # 作为ab计数
            for b in range(2 * x + 1, n + 1):
                if b + 6 * x + 1 <= n:
                    aa[b - 2 * x] += post_cd[b + 6 * x + 1] * cnt[b]
                    bb[b] += post_cd[b + 6 * x + 1] * cnt[b - 2 * x]
                else:
                    break

        for x in nums:
            ac.lst([aa[x], bb[x], cc[x], dd[x]])
        return

    @staticmethod
    def lg_p2652(ac=FastIO()):

        # 模板：枚举花色与双指针计算长度
        n = ac.read_int()
        dct = defaultdict(set)
        for _ in range(n):
            a, b = ac.read_ints()
            dct[a].add(b)
        ans = n
        for a in dct:
            lst = sorted(list(dct[a]))
            m = len(lst)
            j = 0
            for i in range(m):
                while j < m and lst[j] - lst[i] <= n - 1:
                    j += 1
                ans = ac.min(ans, n - (j - i))
        ac.st(ans)
        return

    @staticmethod
    def lg_p2994(ac=FastIO()):

        # 模板：按照座位枚举分配人员
        def dis():
            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        n, m = ac.read_ints()
        cow = [ac.read_list_ints() for _ in range(n)]
        pos = [ac.read_list_ints() for _ in range(n)]
        visit = [0] * n
        for j in range(m):
            ceil = inf
            ind = 0
            x1, y1 = pos[j]
            for i in range(n):
                if visit[i]:
                    continue
                x2, y2 = cow[i]
                cur = dis()
                if cur < ceil:
                    ceil = cur
                    ind = i
            if ceil < inf:
                visit[ind] = 1
        ans = [i + 1 for i in range(n) if not visit[i]]
        if not ans:
            ac.st(0)
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lg_p4181(ac=FastIO()):
        # 模板：贪心枚举与后缀和
        n, m, r = ac.read_ints()
        cow = [ac.read_int() for _ in range(n)]
        cow.sort()
        nums1 = [ac.read_list_ints()[::-1] for _ in range(m)]
        nums1.sort(key=lambda it: -it[0])
        nums2 = [ac.read_int() for _ in range(r)]
        nums2.sort(reverse=True)
        # 预处理后缀和
        ind = 0
        post = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            cur = 0
            while ind < m and cow[i]:
                if nums1[ind][1] == 0:
                    ind += 1
                    continue
                x = ac.min(nums1[ind][1], cow[i])
                cow[i] -= x
                nums1[ind][1] -= x
                cur += nums1[ind][0] * x
            post[i] = post[i + 1] + cur
        # 枚举
        ans = post[0]
        pre = 0
        for i in range(ac.min(r, n)):
            pre += nums2[i]
            ans = ac.max(ans, pre + post[i + 1])
        ac.st(ans)
        return

    @staticmethod
    def lg_p6149(ac=FastIO()):
        # 模板：经典枚举三角形的直角点使用前缀和与二分计算距离和
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        dct_x = defaultdict(list)
        dct_y = defaultdict(list)
        for x, y in nums:
            dct_x[x].append(y)
            dct_y[y].append(x)
        pre_x = defaultdict(list)
        for x in dct_x:
            dct_x[x].sort()
            pre_x[x] = ac.accumulate(dct_x[x])
        pre_y = defaultdict(list)
        for y in dct_y:
            dct_y[y].sort()
            pre_y[y] = ac.accumulate(dct_y[y])

        ans = 0
        mod = 10**9 + 7
        for x, y in nums:
            # 二分找到中间点 xi 计算两侧距离
            xi = bisect.bisect_left(dct_y[y], x)
            left_x = (xi + 1) * x - pre_y[y][xi + 1]
            right_x = pre_y[y][-1] - pre_y[y][xi + 1] - (len(dct_y[y]) - xi - 1) * x

            yi = bisect.bisect_left(dct_x[x], y)
            left_y = (yi + 1) * y - pre_x[x][yi + 1]
            right_y = pre_x[x][-1] - pre_x[x][yi + 1] - (len(dct_x[x]) - yi - 1) * y
            ans += (left_x + right_x) * (left_y + right_y)
            ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def lg_p6393(ac=FastIO()):
        # 模板：经典利用值域范围进行枚举计算
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        dct = dict()
        for i in range(n):
            a, b = nums[i]
            if b not in dct:
                dct[b] = dict()
            if a not in dct[b]:
                dct[b][a] = deque()
            dct[b][a].append(i)
        for i in range(n):
            a, b = nums[i]
            ind = -2
            for bb in dct:
                if (b * b) % bb == 0:
                    # 寻找符合条件的最小值
                    aa = a + b * b // bb + b
                    if aa in dct[bb]:
                        while dct[bb][aa] and dct[bb][aa][0] <= i:
                            dct[bb][aa].popleft()
                        if dct[bb][aa]:
                            j = dct[bb][aa][0]
                            if ind == -2 or j < ind:
                                ind = j
                        else:
                            del dct[bb][aa]
                            if not dct[bb]:
                                del dct[bb]
            ac.st(ind + 1)
        return

    @staticmethod
    def lc_2681(nums: List[int]) -> int:
        # 模板：按照贡献法枚举计数
        mod = 10**9 + 7
        nums.sort()
        ans = pre = 0
        for num in nums:
            ans += num * num * pre
            ans += num * num * num
            pre %= mod
            ans %= mod
            pre *= 2
            pre += num
        return ans

    @staticmethod
    def lg_p6767(ac=FastIO()):
        # 模板：贪心枚举性价比较低的数量
        n, a, b, c, d = ac.read_ints()
        if b * c > a * d:
            a, b, c, d = c, d, a, b

        ans = inf
        for x in range(10**5 + 1):
            cur = x * d + b * ac.max(math.ceil((n - x * c) / a), 0)
            ans = ac.min(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p8270(ac=FastIO()):
        # 模板：经典脑筋急转弯枚举，转换为两两字母比较
        s = ac.read_str()
        t = ac.read_str()
        lst = sorted(list("abcdefghijklmnopqr"))
        m = len(lst)
        pre = set()
        for i in range(m):
            for j in range(i, m):
                cur = {lst[i], lst[j]}
                ss = ""
                tt = ""
                for w in s:
                    if w in cur:
                        ss += w
                for w in t:
                    if w in cur:
                        tt += w
                if ss == tt:
                    pre.add(lst[i]+lst[j])
                    pre.add(lst[j]+lst[i])
        ans = ""
        for _ in range(ac.read_int()):
            st = ac.read_str()
            m = len(st)
            flag = True
            for i in range(m):
                for j in range(i, m):
                    if st[i]+st[j] not in pre:
                        flag = False
                        break
                if not flag:
                    break
            ans += "Y" if flag else "N"
        ac.st(ans)
        return

    @staticmethod
    def lg_p8672(ac=FastIO()):
        # 模板：字符串枚举与经典置换环计数
        s = ac.read_str()
        n = len(s)
        dct = dict()
        dct["B"] = s.count("B")
        dct["A"] = s.count("A")
        dct["T"] = s.count("T")
        ans = inf
        for item in permutations("BAT", 3):
            t = ""
            for w in item:
                t += dct[w]*w
            cnt = defaultdict(int)
            for i in range(n):
                if s[i] != t[i]:
                    cnt[s[i]+t[i]] += 1
            cur = 0
            for w in item:
                for p in item:
                    if w != p:
                        x = ac.min(cnt[w+p], cnt[p+w])
                        cur += x
                        cnt[w+p] -= x
                        cnt[p+w] -= x
            rest = sum(cnt.values())
            cur += rest*2 // 3
            ans = ac.min(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p9076(ac=FastIO()):
        # 模板：根据数字的因数进行枚举
        n = ac.read_int()
        ans = 0
        pre = set()
        for a in range(1, int(n**0.5)+1):
            if n % a == 0:
                for bc in [n//a-1, a-1]:
                    if bc in pre:
                        continue
                    pre.add(bc)
                    for x in range(2, bc+1):
                        if bc % x == 0:
                            y = bc//x-1
                            if y > 1:
                                ans += 1
                        if bc // x <= 2:
                            break
        ac.st(ans)
        return

    @staticmethod
    def lg_p9008(ac=FastIO()):
        # 模板：经典朋友敌人陌生人容斥枚举计数
        n, p, q = ac.read_ints()
        friend = defaultdict(set)
        for _ in range(p):
            u, v = ac.read_ints()
            friend[u].add(v)
            friend[v].add(u)
        ans = n * (n - 1) // 2
        rem = set()
        for _ in range(q):
            u, v = ac.read_ints()
            rem.add((u, v) if u < v else (v, u))
            for x in friend[u]:
                if x not in friend[v]:
                    rem.add((x, v) if x < v else (v, x))
            for y in friend[v]:
                if y not in friend[u]:
                    rem.add((y, u) if y < u else (u, y))
        ac.st(ans - len(rem))
        return

    @staticmethod
    def lg_p9006(ac=FastIO()):
        # 模板：经典枚举取模计数
        mod = 100000007
        n, k = ac.read_ints()
        num = 9 * 10**(n - 1)
        x = num // k
        x %= mod
        ans = [x] * k
        for y in range(10**(n - 1) + x * k, 10**(n - 1) + x * k + num % k):
            ans[y % k] += 1
        ac.lst([x % mod for x in ans])
        return

    @staticmethod
    def lg_p8948(ac=FastIO()):
        # 模板：预处理和枚举所有情况
        dct = dict()
        dct[2000] = [400, 600]
        for i in range(401):
            for j in range(601):
                x = (3 * i + 2 * j) * 10 / 12
                x = int(x) + int(x - int(x) >= 0.5)
                if 10 <= x <= 1990:
                    dct[x] = [i, j]
        for _ in range(ac.read_int()):
            ac.lst(dct[ac.read_int()])
        return

    @staticmethod
    def lg_p8894(ac=FastIO()):
        # 模板：按照区间范围值进行枚举前后缀计数
        n = ac.read_int()
        mod = 998244353
        nums = [ac.read_list_ints() for _ in range(n)]
        ceil = max(q for _, q in nums)
        low = min(p for p, _ in nums)
        ans = 0
        for s in range(low, ceil + 1):
            pre = [0] * (n + 1)
            pre[0] = 1
            for i in range(n):
                p, q = nums[i]
                if p > s:
                    pre[i + 1] = 0
                    break
                else:
                    pre[i + 1] = pre[i] * (ac.min(s, q) - p + 1) % mod

            post = [0] * (n + 1)
            post[n] = 1
            for i in range(n - 1, -1, -1):
                p, q = nums[i]
                if p >= s:
                    post[i] = 0
                    break
                else:
                    post[i] = post[i + 1] * (ac.min(q, s - 1) - p + 1) % mod
            for i in range(n):
                p, q = nums[i]
                if p <= s <= q:
                    ans += pre[i] * post[i + 1] * s
                    ans %= mod
                if pre[i+1] == 0:
                    break
        ac.st(ans)
        return

    @staticmethod
    def lg_p8872(ac=FastIO()):
        # 模板：排序后前后缀移动次数枚举
        n, m = ac.read_ints()
        nums = sorted(ac.read_list_ints())
        ans = inf
        for i in range(n):
            if i > m:
                break
            right = (m - i) // 2
            if right >= n - i - 1:
                ac.st(0)
                return
            cur = nums[-right - 1] - nums[i]
            ans = ac.min(ans, cur)

        for i in range(n - 1, -1, -1):
            if n - i - 1 > m:
                break
            left = (m - n + i + 1) // 2
            if left >= i:
                ac.st(0)
                return
            cur = nums[i] - nums[left]
            ans = ac.min(ans, cur)
        ac.st(ans)
        return


class TestGeneral(unittest.TestCase):

    def test_violent_enumeration(self):
        ve = ViolentEnumeration()
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert ve.matrix_rotate(matrix) == matrix
        return


if __name__ == '__main__':
    unittest.main()