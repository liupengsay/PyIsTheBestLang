

import unittest

from algorithm.src.fast_io import FastIO

"""
算法：乘法逆元、组合数求幂快速计算
功能：求逆元取模，注意取的模必须为质数，且不能整除该质数，否则不存在对应的乘法逆元
题目：

===================================洛谷===================================
P3811 乘法逆元（https://www.luogu.com.cn/problem/P3811）使用乘法逆元计算
P5431 乘法逆元（https://www.luogu.com.cn/problem/P5431）使用乘法逆元计算
P2613 有理数取余（https://www.luogu.com.cn/problem/P2613）使用乘法逆元计算

参考：OI WiKi（xx）
"""


class MultiplicativeInverse:
    def __init__(self):
        return

    @staticmethod
    def compute_with_api(a, p):
        return pow(a, -1, p)

    # 扩展欧几里得求逆元
    def ex_gcd(self, a, b):
        if b == 0:
            return 1, 0, a
        else:
            x, y, q = self.ex_gcd(b, a % b)
            x, y = y, (x - (a // b) * y)
            return x, y, q

    # 扩展欧几里得求逆元
    def mod_reverse(self, a, p):
        x, y, q = self.ex_gcd(a, p)
        if q != 1:
            raise Exception("No solution.")
        else:
            return (x + p) % p  # 防止负数(a, p):
            # 注意a和p都为正整数，且p为质数


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p3811(ac=FastIO()):
        n, p = ac.read_ints()
        for i in range(1, n + 1):
            ac.st(MultiplicativeInverse().mod_reverse(i, p))
        return


class TestGeneral(unittest.TestCase):

    def test_multiplicative_inverse(self):
        mt = MultiplicativeInverse()
        assert MultiplicativeInverse().mod_reverse(10, 13) == 4
        assert MultiplicativeInverse().mod_reverse(10, 1) == 0
        return


if __name__ == '__main__':
    unittest.main()
