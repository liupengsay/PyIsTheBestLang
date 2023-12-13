"""
Algorithm：linear_basis|kth_subset_xor|rank_of_xor
Description：subset_xor|kth_xor|rank_of_xor

=====================================LuoGu======================================
P3812（https://www.luogu.com.cn/problem/P3812）linear_basis

"""

from src.mathmatics.linear_basis.template import LinearBasis
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p3812(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3812
        tag: linear_basis
        """
        # 线性基查询数组取任何子集得到的 xor 最大值
        ac.read_int()
        nums = ac.read_list_ints()
        lb = LinearBasis(nums)
        ac.st(lb.query_max())
        return
