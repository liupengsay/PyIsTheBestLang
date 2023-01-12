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

参考：OI WiKi（xx）
"""


class SpiralMatrix:
    def __init__(self):
        return

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
