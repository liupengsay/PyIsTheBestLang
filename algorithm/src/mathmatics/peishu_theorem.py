import math
import unittest

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


class PeiShuTheorem:
    def __init__(self, lst):
        self.lst = lst
        self.get_lst_gcd()
        return

    def get_lst_gcd(self):
        self.ans = self.lst[0]
        for g in self.lst[1:]:
            self.ans = math.gcd(self.ans, g)
        return


class TestGeneral(unittest.TestCase):

    def test_peishu_theorem(self):
        lst = [4059, -1782]
        pst = PeiShuTheorem(lst)
        assert pst.ans == 99
        return


if __name__ == '__main__':
    unittest.main()
