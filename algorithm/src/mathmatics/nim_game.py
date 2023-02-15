
import unittest

from functools import reduce
from operator import xor

"""

算法：nim游戏也叫公平组合游戏，属于博弈论范畴
功能：用来判断游戏是否存在必胜态与必输态，博弈DP类型
题目：

===================================洛谷===================================
P2197 【模板】nim 游戏（https://www.luogu.com.cn/problem/P2197）有一个神奇的结论：当n堆石子的数量异或和等于0时，先手必胜，否则先手必败
参考：OI WiKi（https://oi-wiki.org/graph/lgv/）


"""


class Nim:
    def __init__(self, lst):
        self.lst = lst
        return

    def gen_result(self):
        return reduce(xor, self.lst) != 0


class TestGeneral(unittest.TestCase):
    def test_euler_phi(self):
        nim = Nim([0, 2, 3])
        assert nim.gen_result() == True
        return


if __name__ == '__main__':
    unittest.main()
