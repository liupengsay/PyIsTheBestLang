"""
Algorithm：Minimum Excluded Element
Ability：brain storming like or g greedy
Reference：

====================================LeetCode====================================
330（https://leetcode.com/problems/patching-array/）greedy|sorting|implemention1798. Maximum Number of Consecutive Values You Can Make（https://leetcode.com/problems/maximum-number-of-consecutive-values-you-can-make/）看似背包实则greedy
1798（https://leetcode.com/problems/maximum-number-of-consecutive-values-you-can-make/）greedy|sorting|implemention
2952（https://leetcode.com/problems/minimum-number-of-coins-to-be-added/）greedy|sorting|implemention

===================================Luogu=====================================
9202（https://www.luogu.com.cn/problem/P9202）最少修改次数使得任意非空连续子数组的mex不等于k
9199（https://www.luogu.com.cn/problem/P9199）最少修改次数使得任意非空连续子数组的mex不等于k

===================================CodeForces===================================
xx（xxx）xxxxxxxxxxxxxxxxxxxx

=============================================================================
"""
from typing import List


class XXX:
    def __init__(self):
        return


class Solution:
    def __int__(self):
        return


    @staticmethod
    def lc_330(nums: List[int], n: int) -> int:
        nums.sort()
        m = len(nums)
        i = 0
        mex = 1
        ans = 0
        while mex <= n:
            if i < m and nums[i] <= mex:
                mex += nums[i]
                i += 1
            else:
                ans += 1
                mex *= 2
        return ans

    @staticmethod
    def lc_2952(nums: List[int], n: int) -> int:
        nums.sort()
        m = len(nums)
        i = 0
        mex = 1
        ans = 0
        while mex <= n:
            if i < m and nums[i] <= mex:
                mex += nums[i]
                i += 1
            else:
                ans += 1
                mex *= 2
        return ans

    @staticmethod
    def lc_1798(coins: List[int]) -> int:
        coins.sort()
        mex = 1
        for coin in coins:
            if coin <= mex:
                mex += coin
        return mex