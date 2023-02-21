import unittest
from typing import List

"""
算法：环形线性或者区间DP
功能：计算环形数组上的操作，比较简单的方式是将数组复制成两遍进行区间或者线性DP

题目：

===================================力扣===================================
918. 环形子数组的最大和（https://leetcode.cn/problems/maximum-sum-circular-subarray/）枚举可能的最大与最小区间

参考：OI WiKi（xx）
"""


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_918(nums: List[int]) -> int:
        # 模板：环形子数组的最大非空连续子数组和
        s = ceil = floor = pre_ceil = pre_floor = nums[0]
        for num in nums[1:]:
            s += num
            pre_floor = pre_floor if pre_floor < 0 else 0
            pre_floor += num
            floor = floor if floor < pre_floor else pre_floor

            pre_ceil = pre_ceil if pre_ceil > 0 else 0
            pre_ceil += num
            ceil = ceil if ceil > pre_ceil else pre_ceil
        if floor < s:
            return max(ceil, s - floor)
        return ceil


class TestGeneral(unittest.TestCase):

    def test_circle_dp(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
