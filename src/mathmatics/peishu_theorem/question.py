"""
算法：裴蜀定理
功能：是一个关于最大公约数的定理可以推广到n个数，比如设a、b是不全为零的整数，则存在整数x、y, 使得ax+by=gcd(a,b)
题目：

===================================力扣===================================
1250. 检查「好数组」（https://leetcode.cn/problems/check-if-it-is-a-good-array/）计算所有元素的最大公约数是否为1

===================================洛谷===================================
P4549 裴蜀定理（https://www.luogu.com.cn/problem/P4549）计算所有元素能加权生成的最小正数和即所有整数的最大公约数


参考：OI WiKi（https://oi-wiki.org/math/number-theory/bezouts/）
"""

from typing import List

from src.mathmatics.peishu_theorem.template import PeiShuTheorem
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1250(nums: List[int]) -> bool:
        # 模板：转化为裴蜀定理计算数组最大公约数是否等于 1 求解
        return PeiShuTheorem().get_lst_gcd(nums) == 1

    @staticmethod
    def lg_p4549(ac=FastIO()):
        # 模板：转化为裴蜀定理计算数组最大公约数求解
        ac.read_int()
        nums = ac.read_list_ints()
        ac.st(PeiShuTheorem().get_lst_gcd(nums))
        return
