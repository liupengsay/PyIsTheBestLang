
import unittest

"""
算法：线性DP
功能：遍历数组，根据前序或者后序结果进行更新，最大非空连续子序列和
题目：

===================================力扣===================================
2361. 乘坐火车路线的最少费用（https://leetcode.cn/problems/minimum-costs-using-the-train-line/）当前状态只跟前一个状态有关
2318. 不同骰子序列的数目（https://leetcode.cn/problems/number-of-distinct-roll-sequences/）当前状态只跟前一个状态有关使用枚举计数
2263. 数组变为有序的最小操作次数（https://leetcode.cn/problems/make-array-non-decreasing-or-non-increasing/）当前状态只跟前一个状态有关
2209. 用地毯覆盖后的最少白色砖块（https://leetcode.cn/problems/minimum-white-tiles-after-covering-with-carpets/）前缀优化与处理进行转移
2188. 完成比赛的最少时间（https://leetcode.cn/problems/minimum-time-to-finish-the-race/）预处理DP
2167. 移除所有载有违禁货物车厢所需的最少时间（https://leetcode.cn/problems/minimum-time-to-remove-all-cars-containing-illegal-goods/）使用前缀后缀DP预处理后进行枚举

===================================洛谷===================================
P1970 [NOIP2013 提高组] 花匠（https://www.luogu.com.cn/problem/P1970）使用贪心与动态规划计算最长的山脉子数组
P1564 膜拜（https://www.luogu.com.cn/problem/P1564）线性DP
P1481 魔族密码（https://www.luogu.com.cn/problem/P1481）线性DP
P2029 跳舞（https://www.luogu.com.cn/problem/P2029）线性DP
P2031 脑力达人之分割字串（https://www.luogu.com.cn/problem/P2031）线性DP
P2062 分队问题（https://www.luogu.com.cn/problem/P2062）线性DP+前缀最大值DP剪枝优化
P2072 宗教问题（https://www.luogu.com.cn/problem/P2072）两个线性DP

P2096 最佳旅游线路（https://www.luogu.com.cn/problem/P2096）最大连续子序列和变种
P5761 [NOI1997] 最佳游览（https://www.luogu.com.cn/problem/P5761）最大连续子序列和变种
P2285 [HNOI2004]打鼹鼠（https://www.luogu.com.cn/problem/P2285）线性DP+前缀最大值DP剪枝优化

P2642 双子序列最大和（https://www.luogu.com.cn/problem/P2642）枚举前后两个非空的最大子序列和
P1470 [USACO2.3]最长前缀 Longest Prefix（https://www.luogu.com.cn/problem/P1470）线性DP
P1096 [NOIP2007 普及组] Hanoi 双塔问题（https://www.luogu.com.cn/problem/P1096）经典线性DP

P2896 [USACO08FEB]Eating Together S（https://www.luogu.com.cn/problem/P2896）前后缀动态规划
P2904 [USACO08MAR]River Crossing S（https://www.luogu.com.cn/problem/P2904）前缀和预处理加线性DP

P3062 [USACO12DEC]Wifi Setup S（https://www.luogu.com.cn/problem/P3062）线性DP枚举

P3842 [TJOI2007]线段（https://www.luogu.com.cn/problem/P3842）线性DP进行模拟
P3903 导弹拦截III（https://www.luogu.com.cn/problem/P3903）线性DP枚举当前元素作为谷底与山峰的子序列长度
P5414 [YNOI2019] 排序（https://www.luogu.com.cn/problem/P5414）贪心，使用线性DP计算最大不降子序列和
P6191 [USACO09FEB]Bulls And Cows S（https://www.luogu.com.cn/problem/P6191）线性DP枚举计数
P6208 [USACO06OCT] Cow Pie Treasures G（https://www.luogu.com.cn/problem/P6208）线性DP模拟
P7404 [JOI 2021 Final] とてもたのしい家庭菜園 4（https://www.luogu.com.cn/problem/P7404）动态规划枚举，计算变成山脉数组的最少操作次数
P7541 [COCI2009-2010#1] DOBRA（https://www.luogu.com.cn/problem/P7541）线性DP记忆化搜索，类似数位DP
P7767 [COCI 2011/2012 #5] DNA（https://www.luogu.com.cn/problem/P7767）线性DP，计算前缀变成全部相同字符的最少操作次数

================================CodeForces================================
https://codeforces.com/problemset/problem/75/D（经典压缩数组，最大子段和升级）
https://codeforces.com/problemset/problem/1084/C（线性DP加前缀和优化）
https://codeforces.com/problemset/problem/166/E（线性DP计数）
https://codeforces.com/problemset/problem/1221/D（线性DP模拟）
C. Chef Monocarp（https://codeforces.com/problemset/problem/1437/C）二维线性DP，两个数组线性移动进行匹配计算最大或者最小值

参考：OI WiKi（xx）
"""


class LinearDP:
    def __init__(self):
        return

    @staticmethod
    def cf_1437c(n, nums):
        # 模板：两个数组线性移动进行匹配计算最大或者最小值
        nums.sort()
        m = 2 * n
        dp = [[float("inf")] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        for i in range(m):
            dp[i + 1][0] = 0
            for j in range(n):
                dp[i + 1][j + 1] = min(dp[i][j + 1], dp[i][j] + abs(nums[j] - i - 1))
        return dp[m][n]

    @staticmethod
    def liner_dp_template(nums):
        # 线性 DP 递推模板（以最长上升子序列长度为例）
        n = len(nums)
        dp = [0] * (n + 1)
        for i in range(n):
            dp[i + 1] = 1
            for j in range(i):
                if nums[i] > nums[j] and dp[j] + 1 > dp[i + 1]:
                    dp[i + 1] = dp[j] + 1
        return max(dp)


class TestGeneral(unittest.TestCase):

    def test_linear_dp(self):
        ld = LinearDP()
        nums = [6, 3, 5, 2, 1, 6, 8, 9]
        assert ld.liner_dp_template(nums) == 4
        return


if __name__ == '__main__':
    unittest.main()
