"""
Algorithm：Minimum Excluded Element
Ability：brain storming like or g greedy
Reference：

====================================LeetCode====================================
330（https://leetcode.cn/problems/patching-array/）greedy|sort|implemention|mex
1798（https://leetcode.cn/problems/maximum-number-of-consecutive-values-you-can-make/）greedy|sort|implemention|mex
2952（https://leetcode.cn/problems/minimum-number-of-coins-to-be-added/）greedy|sort|implemention|mex

======================================Luogu=====================================
P9202（https://www.luogu.com.cn/problem/P9202）mex|operation
P9199（https://www.luogu.com.cn/problem/P9199）mex|operation

===================================CodeForces===================================


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
        """
        url: https://leetcode.cn/problems/patching-array/
        tag: greedy|sort|implemention|mex
        """
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
        """
        url: https://leetcode.cn/problems/minimum-number-of-coins-to-be-added/
        tag: greedy|sort|implemention|mex
        """
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
        """
        url: https://leetcode.cn/problems/maximum-number-of-consecutive-values-you-can-make/
        tag: greedy|sort|implemention|mex
        """
        coins.sort()
        mex = 1
        for coin in coins:
            if coin <= mex:
                mex += coin
        return mex
