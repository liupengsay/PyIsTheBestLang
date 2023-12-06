"""
Algorithm：裴蜀定理
Function：是一个关于最大公约数的定理可以推广到n个数，比如设a、b是不全为零的整数，则存在整数x、y, 使得ax+by=gcd(a,b)

====================================LeetCode====================================
1250（https://leetcode.com/problems/check-if-it-is-a-good-array/）所有元素的最大公约数是否为1

===================================CodeForces===================================
1478D（https://codeforces.com/contest/1478/problem/D）peishu_theorem|number_theory|math

=====================================LuoGu======================================
4549（https://www.luogu.com.cn/problem/P4549）所有元素能|权生成的最小正数和即所有整数的最大公约数


"""

from typing import List

from src.mathmatics.peishu_theorem.template import PeiShuTheorem
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1250(nums: List[int]) -> bool:
        # 转化为裴蜀定理数组最大公约数是否等于 1 求解
        return PeiShuTheorem().get_lst_gcd(nums) == 1

    @staticmethod
    def lg_p4549(ac=FastIO()):
        # 转化为裴蜀定理数组最大公约数求解
        ac.read_int()
        nums = ac.read_list_ints()
        ac.st(PeiShuTheorem().get_lst_gcd(nums))
        return