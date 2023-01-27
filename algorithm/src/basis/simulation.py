import random
import unittest

"""
算法：模拟
功能：根据题意进行模拟
题目：
L2296 设计一个文本编辑器（https://leetcode.cn/problems/design-a-text-editor/）使用指针维护结果进行模拟
P1815 蠕虫游戏（https://www.luogu.com.cn/problem/P1815）模拟类似贪吃蛇的移动
P1538 迎春舞会之数字舞蹈（https://www.luogu.com.cn/problem/P1538）模拟数字文本的打印
P1535 [USACO08MAR]Cow Travelling S（https://www.luogu.com.cn/problem/P1535）动态规划模拟计数
P2239 [NOIP2014 普及组] 螺旋矩阵（https://www.luogu.com.cn/problem/P2239）模拟螺旋矩阵的赋值
54. 螺旋矩阵（https://leetcode.cn/problems/spiral-matrix/）https://leetcode.cn/problems/spiral-matrix/
59. 螺旋矩阵 II（https://leetcode.cn/problems/spiral-matrix-ii/）
2326. 螺旋矩阵 IV（https://leetcode.cn/problems/spiral-matrix-iv/）
P2338 [USACO14JAN]Bessie Slows Down S（https://www.luogu.com.cn/problem/P2338）按照题意进行时间与距离的模拟

P2366 yyy2015c01的IDE之Watches（https://www.luogu.com.cn/problem/P2366）字符串模拟与变量赋值计算
P2552 [AHOI2001]团体操队形（https://www.luogu.com.cn/problem/P2552）经典矩阵赋值模拟
P2696 慈善的约瑟夫（https://www.luogu.com.cn/problem/P2696）约瑟夫环模拟与差分计算
剑指 Offer 62. 圆圈中最后剩下的数字（https://leetcode.cn/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/）约瑟夫环
P1234 小A的口头禅（https://www.luogu.com.cn/problem/P1234）计算矩阵每个点四个方向特定长为4的单词个数

P1166 打保龄球（https://www.luogu.com.cn/problem/P1166）按照题意复杂的模拟题
P1076 [NOIP2012 普及组] 寻宝（https://www.luogu.com.cn/problem/P1076）模拟进行操作即可
P8924 「GMOI R1-T1」Perfect Math Class（https://www.luogu.com.cn/problem/P8924）模拟的同时使用进制的思想进行求解

P8889 狠狠地切割(Hard Version)（https://www.luogu.com.cn/problem/P8889）经典01序列分段计数

P8870 [传智杯 #5 初赛] B-莲子的机械动力学（https://www.luogu.com.cn/problem/P8870）按照进制进行加法模拟
P3880 [JLOI2008]提示问题（https://www.luogu.com.cn/problem/P3880）按照题意模拟加密字符串
P3111 [USACO14DEC]Cow Jog S（https://www.luogu.com.cn/problem/P3111）逆向思维使用行进距离模拟分组，类似力扣车队题目
P4346 [CERC2015]ASCII Addition（https://www.luogu.com.cn/problem/P4346）模拟数字与字符串的转换与进行加减
P5079 Tweetuzki 爱伊图（https://www.luogu.com.cn/problem/P5079）字符串模拟

参考：OI WiKi（xx）
"""


class SpiralMatrix:
    def __init__(self):
        return

    @staticmethod
    def joseph_ring(n, m):
        # 模板：约瑟夫环计算最后的幸存者
        # 0.1..m-1每次选取第m个消除之后剩下的编号
        f = 0
        for x in range(2, n + 1):
            f = (m + f) % x
        return f


    @staticmethod
    def num_to_loc(m, n, num):
        # 根据从左往右从上往下的顺序生成给定数字的行列索引
        # 0123、4567
        return [num // n, num % n]

    @staticmethod
    def loc_to_num(r, c, m, n):
        # 根据从左往右从上往下的顺序给定的行列索引生成数字
        return r * n + n

    @staticmethod
    def get_spiral_matrix_num1(m, n, r, c):  # 顺时针螺旋
        # 获取 m*n 矩阵的 [r, c] 元素位置（元素从 1 开始索引从 1 开始）
        num = 1
        while r not in [1, m] and c not in [1, n]:
            num += 2 * m + 2 * n - 4
            r -= 1
            c -= 1
            n -= 2
            m -= 2

        # 复杂度 O(m+n)
        x = y = 1
        direc = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        d = 0
        while [x, y] != [r, c]:
            a, b = direc[d]
            if not (1 <= x + a <= m and 1 <= y + b <= n):
                d += 1
                a, b = direc[d]
            x += a
            y += b
            num += 1
        return num

    @staticmethod
    def get_spiral_matrix_num2(m, n, r, c):  # 顺时针螺旋
        # 获取 m*n 矩阵的 [r, c] 元素位置（元素从 1 开始索引从 1 开始）

        rem = min(r - 1, m - r, c - 1, n - c)
        num = 2 * rem * (m - rem + 1) + 2 * rem * (n - rem + 1) - 4 * rem
        m -= 2 * rem
        n -= 2 * rem
        r -= rem
        c -= rem

        # 复杂度 O(1)
        if r == 1:
            num += c
        elif 1 < r <= m and c == n:
            num += n + (r - 1)
        elif r == m and 1 <= c <= n - 1:
            num += n + (m - 1) + (n - c)
        else:
            num += n + (m - 1) + (n - 1) + (m - r)
        return num

    @staticmethod
    def get_spiral_matrix_loc(m, n, num):  # 顺时针螺旋
        # 获取 m*n 矩阵的元素 num 位置

        def check(x):
            res = 2 * x * (m - x + 1) + 2 * x * (n - x + 1) - 4 * x
            return res < num

        low = 0
        high = max(m, n)
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                low = mid
            else:
                high = mid
        rem = low if check(low) else high
        num -= 2 * rem * (m - rem + 1) + 2 * rem * (n - rem + 1) - 4 * rem
        assert num > 0
        m -= 2 * rem
        n -= 2 * rem
        r = c = rem

        if num <= n:
            a = 1
            b = num
        elif n < num <= n + m - 1:
            a = num - n + 1
            b = n
        elif n + (m - 1) < num <= n + (m - 1) + (n - 1):
            a = m
            b = n - (num - n - (m - 1))
        else:
            a = m - (num - n - (n - 1) - (m - 1))
            b = 1
        return [r + a, c + b]


class TestGeneral(unittest.TestCase):

    def test_spiral_matrix(self):
        sm = SpiralMatrix()
        nums = [[1, 2, 3, 4], [12, 13, 14, 5], [11, 16, 15, 6], [10, 9, 8, 7]]
        m = len(nums)
        n = len(nums[0])
        for i in range(m):
            for j in range(n):
                assert sm.get_spiral_matrix_num1(
                    m, n, i + 1, j + 1) == nums[i][j]
                assert sm.get_spiral_matrix_num2(
                    m, n, i + 1, j + 1) == nums[i][j]

        nums = [[1, 2, 3, 4, 5, 6], [14, 15, 16, 17, 18, 7],
                [13, 12, 11, 10, 9, 8]]
        m = len(nums)
        n = len(nums[0])
        for i in range(m):
            for j in range(n):
                assert sm.get_spiral_matrix_num1(
                    m, n, i + 1, j + 1) == nums[i][j]
                assert sm.get_spiral_matrix_num2(
                    m, n, i + 1, j + 1) == nums[i][j]

        for _ in range(10):
            m = random.randint(5, 100)
            n = random.randint(5, 100)
            for i in range(m):
                for j in range(n):
                    num = sm.get_spiral_matrix_num1(m, n, i + 1, j + 1)
                    assert sm.get_spiral_matrix_num2(m, n, i + 1, j + 1) == num
                    assert sm.get_spiral_matrix_loc(
                        m, n, num) == [i + 1, j + 1]

        return


if __name__ == '__main__':
    unittest.main()
