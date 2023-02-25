import bisect
import unittest
from typing import List

"""

算法：最长上升（或不降）子序列 Longest Increasing Subsequence（LIS）
最长单调递增子序列（严格上升）：<
最长单调不减子序列（不降）：<=
最长单调递减子序列（严格下降）：>
最长单调不增子序列（不升）：>=
对于数组来说，正数反可以将后两个问题3和4转换为前两个问题1和2进行解决，可以算全局的最长单调子序列，也可以计算前后缀的最长单调子序列
dilworth定理：不下降子序列最小个数等于最大上升子序列的长度，不上升子序列最小个数等于最大下降子序列的长度。
参考题目：
===================================力扣===================================
2111. 使数组 K 递增的最少操作次数（https://leetcode.cn/problems/minimum-operations-to-make-the-array-k-increasing/）分成 K 组计算每组的最长递增子序列

===================================洛谷===================================
P1020 导弹拦截（https://www.luogu.com.cn/problem/P1020）使用贪心加二分计算最长单调不减和单调不增子序列的长度
P1439 最长公共子序列（https://www.luogu.com.cn/problem/P1439）使用贪心加二分计算最长单调递增子序列的长度
P1091 合唱队形（https://www.luogu.com.cn/problem/P1091）可以往前以及往后计算最长单调子序列
P1233 木棍加工（https://www.luogu.com.cn/problem/P1233）按照一个维度排序后计算另一个维度的，最长严格递增子序列的长度
P2782 友好城市（https://www.luogu.com.cn/problem/P2782）按照一个维度排序后计算另一个维度的，最长严格递增子序列的长度（也可以考虑使用线段树求区间最大值）
P3902 递增（https://www.luogu.com.cn/problem/P3902）最长严格上升子序列
P6403 [COCI2014-2015#2] STUDENTSKO（https://www.luogu.com.cn/problem/P6403）问题转化为最长不降子序列

"""



class LongestIncreasingSubsequence:
    def __init__(self):
        return

    @staticmethod
    def definitely_increase(nums):
        # 最长单调递增子序列（严格上升）
        dp = []
        for num in nums:
            i = bisect.bisect_left(dp, num)
            if 0 <= i < len(dp):
                dp[i] = num
            else:
                dp.append(num)
        return len(dp)

    @staticmethod
    def definitely_not_reduce(nums):
        # 最长单调不减子序列（不降）
        dp = []
        for num in nums:
            i = bisect.bisect_right(dp, num)
            if 0 <= i < len(dp):
                dp[i] = num
            else:
                dp.append(num)
        return len(dp)

    def definitely_reduce(self, nums):
        # 最长单调递减子序列（严格下降）
        nums = [-num for num in nums]
        return self.definitely_increase(nums)

    def definitely_not_increase(self, nums):
        # 最长单调不增子序列（不升）
        nums = [-num for num in nums]
        return self.definitely_not_reduce(nums)


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_2111(arr: List[int], k: int) -> int:
        # 模板：最长不降子序列
        ans = 0
        for i in range(k):
            lst = arr[i::k]
            ans += len(lst)-LongestIncreasingSubsequence().definitely_not_reduce(lst)
        return ans



class TestGeneral(unittest.TestCase):

    def test_longest_increasing_subsequence(self):
        lis = LongestIncreasingSubsequence()
        nums = [1, 2, 3, 3, 2, 2, 1]
        assert lis.definitely_increase(nums) == 3
        assert lis.definitely_not_reduce(nums) == 4
        assert lis.definitely_reduce(nums) == 3
        assert lis.definitely_not_increase(nums) == 5
        return


if __name__ == '__main__':
    unittest.main()
