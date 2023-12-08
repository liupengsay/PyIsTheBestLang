from src.mathmatics.linear_basis.template import LinearBasis
from src.utils.fast_io import FastIO

"""
Algorithm：线性基也叫Hamel基
Description：求解数组子集的异或和、排第K的异或和、以及异或和排第几、更新线性基即原始数组等

=====================================LuoGu======================================
3812（https://www.luogu.com.cn/problem/P3812）

"""


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p3812(ac=FastIO()):
        # 线性基查询数组取任何子集得到的 xor 最大值
        ac.read_int()
        nums = ac.read_list_ints()
        lb = LinearBasis(nums)
        ac.st(lb.query_max())
        return