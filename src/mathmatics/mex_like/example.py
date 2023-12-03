"""
Algorithm：
Ability：
Reference：

===================================LeetCode===================================
330. Patching Array（https://leetcode.com/problems/patching-array/）greedy|sorting|implemention
2952. Minimum Number of Coins to be Added （https://leetcode.com/problems/minimum-number-of-coins-to-be-added/）greedy|sorting|implemention

===================================Luogu=====================================

================================CodeForces===================================


=============================================================================
"""
import unittest
from typing import List


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


class TestGeneral(unittest.TestCase):

    def test_xxxx(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
