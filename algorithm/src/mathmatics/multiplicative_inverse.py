

import unittest

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
    def get_result(a, p):
        # 注意a和p都为正整数，且p为质数
        return pow(a, -1, p)


class TestGeneral(unittest.TestCase):

    def test_multiplicative_inverse(self):
        mt = MultiplicativeInverse()
        assert mt.get_result(10, 13) == 4
        assert mt.get_result(10, 1) == 0
        return


if __name__ == '__main__':
    unittest.main()
