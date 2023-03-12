import unittest
import math
from itertools import combinations

from algorithm.src.fast_io import FastIO, inf


"""
算法：暴力枚举、旋转矩阵、螺旋矩阵（也叫brute_force）
功能：根据题意，在复杂度有限的情况下，进行所有可能情况的枚举
题目：

===================================力扣===================================
670. 最大交换（https://leetcode.cn/problems/maximum-swap/）看似贪心，在复杂度允许的情况下使用枚举暴力保险
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

================================CodeForces================================
https://codeforces.com/problemset/problem/1426/F（分类枚举中间的b计数两边的?ac，并使用快速幂进行求解）
D. Zigzags（https://codeforces.com/problemset/problem/1400/D）枚举+二分
D. Moscow Gorillas（https://codeforces.com/contest/1793/problem/D）枚举计数
D. Dima and Lisa（https://codeforces.com/problemset/problem/584/D）确定一个质数3，枚举第二三个质数，小于 10**9 的任何数都可以分解为最多三个质数的和
D. Three Integers（https://codeforces.com/problemset/problem/1311/D）根据题意，确定一个上限值，贪心枚举

参考：OI WiKi（xx）
"""


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


class TestGeneral(unittest.TestCase):

    def test_violent_enumeration(self):
        ve = ViolentEnumeration()
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert ve.matrix_rotate(matrix) == matrix
        return


if __name__ == '__main__':
    unittest.main()
