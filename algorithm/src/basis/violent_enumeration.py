import unittest

"""
算法：暴力枚举、旋转矩阵、螺旋矩阵
功能：xxx
题目：
P1548 棋盘问题（https://www.luogu.com.cn/problem/P1548）枚举正方形与长方形的右小角计算个数
L2488 统计中位数为 K 的子数组（https://leetcode.cn/problems/count-subarrays-with-median-k/）利用中位数的定义枚举前后子序列中满足大于 K 和小于 K 的数个数相等的子数组
L2484 统计回文子序列数目（https://leetcode.cn/problems/count-palindromic-subsequences/）利用前后缀哈希计数枚举当前索引作为回文中心的回文子串的前后缀个数
L2322 从树中删除边的最小分数（https://leetcode.cn/problems/minimum-score-after-removals-on-a-tree/）枚举删除的第一条边后使用树形递归再枚举第二条边计算连通块异或差值最小分数
L2321 拼接数组的最大分数（https://leetcode.cn/problems/maximum-score-of-spliced-array/）借助经典的最大最小子数组和枚举交换的连续子序列
L2306 公司命名（https://leetcode.cn/problems/naming-a-company/）利用字母个数有限的特点枚举首字母交换对
L2272 最大波动的子字符串（https://leetcode.cn/problems/substring-with-largest-variance/）利用字母个数有限的特点进行最大字符与最小字符枚举
L2183 统计可以被 K 整除的下标对数目（https://leetcode.cn/problems/count-array-pairs-divisible-by-k/）按照最大公约数进行分组枚举
L2151 基于陈述统计最多好人数（https://leetcode.cn/problems/maximum-good-people-based-on-statements/）使用状态压缩进行枚举并判定是否合法
L2147 分隔长廊的方案数（https://leetcode.cn/problems/number-of-ways-to-divide-a-long-corridor/）根据题意进行分隔点枚举计数
L2122 还原原数组（https://leetcode.cn/problems/recover-the-original-array/）枚举差值 k 判断是否合法

P1632 点的移动（https://www.luogu.com.cn/problem/P1632）枚举横坐标和纵坐标的所有组合移动距离
P2128 赤壁之战（https://www.luogu.com.cn/problem/P2128）枚举完全图的顶点组合，平面图最多四个点
P2191 小Z的情书（https://www.luogu.com.cn/problem/P2191）逆向思维旋转矩阵
P2699 【数学1】小浩的幂次运算（https://www.luogu.com.cn/problem/P2699）分类讨论、暴力枚举模拟
P1371 NOI元丹（https://www.luogu.com.cn/problem/P1371）前后缀枚举计数
P1369 矩形（https://www.luogu.com.cn/problem/P1369）矩阵DP与有贪心枚举在矩形边上的最多点个数

P1158 [NOIP2010 普及组] 导弹拦截（https://www.luogu.com.cn/problem/P1158）按照距离的远近进行预处理排序枚举，使用后缀最大值进行更新

P8928 「TERRA-OI R1」你不是神，但你的灵魂依然是我的盛宴（https://www.luogu.com.cn/problem/P8928）枚举取模计数的组合
P8892 「UOI-R1」磁铁（https://www.luogu.com.cn/problem/P8892）枚举字符串的分割点，经典查看一个字符串是否为另一个字符串的子序列
670. 最大交换（https://leetcode.cn/problems/maximum-swap/）看似贪心，在复杂度允许的情况下使用枚举暴力保险
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


class TestGeneral(unittest.TestCase):

    def test_violent_enumeration(self):
        ve = ViolentEnumeration()
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert ve.matrix_rotate(matrix) == matrix
        return


if __name__ == '__main__':
    unittest.main()
