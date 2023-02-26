import math
import unittest
from collections import defaultdict
from typing import List

from algorithm.src.fast_io import FastIO

"""
算法：计算几何、设计到平面坐标系上的一些问题求解
功能：xxx
题目：

===================================力扣===================================
149. 直线上最多的点数（https://leetcode.cn/problems/max-points-on-a-line/）用直线斜率判断一条线上最多的点数

===================================洛谷===================================
P1665 正方形计数（https://www.luogu.com.cn/problem/P1665）枚举正方形对角线顶点计算可行个数
P2313 [HNOI2005]汤姆的游戏（https://www.luogu.com.cn/problem/P2313）判断点在矩形中或者圆形中
P2358 蚂蚁搬家（https://www.luogu.com.cn/problem/P2358）计算几何判断正方体上表面的点到下表面的点最短距离
P2665 [USACO08FEB]Game of Lines S（https://www.luogu.com.cn/problem/P2665）不同的斜率计算
P1355 神秘大三角（https://www.luogu.com.cn/problem/P1355）使用三角形面积计算判断点与三角形的位置关系
P1142 轰炸（https://www.luogu.com.cn/problem/P1142）利用斜率计算一条直线上最多的点
P2778 [AHOI2016初中组]迷宫（https://www.luogu.com.cn/problem/P2778）枚举圆与点的位置关系
P3021 [USACO11MAR]Bovine Bridge Battle S（https://www.luogu.com.cn/problem/P3021）容斥原理计数加枚举中心对称点

================================CodeForces================================
D. Pair Of Lines (https://codeforces.com/contest/961/problem/D) 抽屉原理枚举初始共线点并计算其他点的共线性情况
参考：OI WiKi（xx）
"""


class Geometry:
    def __init__(self):
        return

    @staticmethod
    def same_line(point1, point2, point3):
        # 模板: 计算三点共线
        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3
        return (x2 - x1) * (y3 - y2) == (x3 - x2) * (y2 - y1)

    @staticmethod
    def compute_slope2(point1, point2):
        assert point1 != point2
        # 模板: 根据两不同的点确定直线斜率
        x1, y1 = point1
        x2, y2 = point2
        a, b = x2 - x1, y2 - y1
        g = math.gcd(a, b)
        a //= g
        b //= g
        if a < 0:
            a *= -1
            b *= -1
        elif a == 0:  # 注意此时的正负号
            b = abs(b)
        return a, b

    @staticmethod
    def compute_slope(x1, y1, x2, y2):
        assert [x1, y1] != [x2, y2]
        # 模板: 根据两不同的点确定直线斜率
        if x1 == x2:
            ans = "x"
        else:
            a = y2 - y1
            b = x2 - x1
            g = math.gcd(a, b)
            if b < 0:
                a *= -1
                b *= -1
            # 使用最简分数来表示斜率
            ans = [a // g, b // g]
        return ans

    @staticmethod
    def compute_square_point(x0, y0, x2, y2):
        assert [x0, y0] != [x2, y2]
        # 模板：已知正方形对角线上的两个点且保证两点不同，求另外两个点的坐标
        x1 = (x0 + x2 + y2 - y0) / 2
        y1 = (y0 + y2 + x0 - x2) / 2
        x3 = (x0 + x2 - y2 + y0) / 2
        y3 = (y0 + y2 - x0 + x2) / 2
        return (x1, y1), (x3, y3)

    @staticmethod
    def compute_square_area(x0, y0, x2, y2):
        # 已知正方形对角线上的点，求正方形面积，注意是整数
        ans = (x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2)
        return ans // 2

    @staticmethod
    def compute_triangle_area(x1, y1, x2, y2, x3, y3):
        # 可用于判断点与三角形的位置关系
        return abs((x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y1 - x1 * y3)) / 2


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_149(points: List[List[int]]) -> int:
        # 模板：计算两个不相同的点组成的直线斜率
        ans = 0
        n = len(points)
        gm = Geometry()
        for i in range(n):
            dct = defaultdict(int)
            dct[0] = 0
            for j in range(i+1, n):
                dct[gm.compute_slope2(points[i], points[j])] += 1
            ans = max(ans, max(dct.values())+1)
        return ans

    @staticmethod
    def lg_p1665(ac=FastIO()):
        # 模板：计算可以组成正方形的点的个数
        n = ac.read_int()
        lst = [ac.read_list_ints() for _ in range(n)]
        dct = set(tuple(p) for p in lst)
        ans = 0
        m = len(lst)
        for i in range(m):
            x1, y1 = lst[i]
            for j in range(i+1, m):
                x2, y2 = lst[j]
                point1, point2 = Geometry.compute_square_point(x1, y1, x2, y2)

                a, b = point1
                if int(a) != a or int(b) != b:
                    continue
                point1 = (int(a), int(b))

                a, b = point2
                if int(a) != a or int(b) != b:
                    continue
                point2 = (int(a), int(b))

                if point1 in dct and point2 in dct:
                    ans += 1
        ac.st(ans//2)
        return


class TestGeneral(unittest.TestCase):

    def test_geometry(self):
        gm = Geometry()
        assert gm.compute_square_point(1, 1, 6, 6) == ((6.0, 1.0), (1.0, 6.0))
        assert gm.compute_square_point(0, 0, 0, 2) == ((1.0, 1.0), (-1.0, 1.0))

        assert gm.compute_triangle_area(0, 0, 2, 0, 1, 1) == 1.0
        return


if __name__ == '__main__':
    unittest.main()
