import unittest

from functools import lru_cache

"""
算法：博弈类DP、玩游戏、必胜态、必输态
功能：通常使用枚举、区间DP加模拟贪心的方式，和记忆化搜索进行状态转移
题目：

===================================力扣===================================
375. 猜数字大小 II（https://leetcode.cn/problems/guess-number-higher-or-lower-ii/）使用区间DP求解的典型博弈DP

===================================洛谷===================================
P1290 欧几里德的游戏（https://www.luogu.com.cn/problem/P1290）典型的博弈DP题
P5635 【CSGRound1】天下第一（https://www.luogu.com.cn/problem/P5635）博弈DP模拟与手写记忆化搜索，避免陷入死循环
P3150 pb的游戏（1）（https://www.luogu.com.cn/problem/P3150）博弈分析必胜策略与最优选择，只跟奇数偶数有关
P4702 取石子（https://www.luogu.com.cn/problem/P4702）博弈分析必胜策略与最优选择，只跟奇数偶数有关

参考：OI WiKi（xx）
"""


class GameDP:
    def __init__(self):
        return

    @staticmethod
    def main_p1280(x, y):

        @lru_cache(None)
        def dfs(a, b):
            if a < b:
                a, b = b, a
            if a % b == 0:
                return True
            for i in range(1, a // b + 1):
                if not dfs(a - i * b, b):
                    return True
            return False

        ans = dfs(x, y)
        return ans


class TestGeneral(unittest.TestCase):

    def test_game_dp(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
