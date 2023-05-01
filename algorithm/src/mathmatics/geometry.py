import math
import unittest
from audioop import add
from collections import defaultdict
from typing import List
import random

from cytoolz import accumulate

from algorithm.src.fast_io import FastIO, inf
from algorithm.src.data_structure.sorted_list import LocalSortedList

"""
算法：计算几何、设计到平面坐标系上的一些问题求解
功能：xxx
题目：

===================================力扣===================================
149. 直线上最多的点数（https://leetcode.cn/problems/max-points-on-a-line/）用直线斜率判断一条线上最多的点数
面试题 16.03. 交点（https://leetcode.cn/problems/intersection-lcci/）计算两条线段最靠左靠下的交点
面试题 16.14. 最佳直线（https://leetcode.cn/problems/best-line-lcci/）用直线斜率判断一条线上最多的点数

===================================洛谷===================================
P1665 正方形计数（https://www.luogu.com.cn/problem/P1665）枚举正方形对角线顶点计算可行个数
P2313 [HNOI2005]汤姆的游戏（https://www.luogu.com.cn/problem/P2313）判断点在矩形中或者圆形中
P2358 蚂蚁搬家（https://www.luogu.com.cn/problem/P2358）计算几何判断正方体上表面的点到下表面的点最短距离
P2665 [USACO08FEB]Game of Lines S（https://www.luogu.com.cn/problem/P2665）不同的斜率计算
P1355 神秘大三角（https://www.luogu.com.cn/problem/P1355）使用三角形面积计算判断点与三角形的位置关系
P1142 轰炸（https://www.luogu.com.cn/problem/P1142）利用斜率计算一条直线上最多的点
P2778 [AHOI2016初中组]迷宫（https://www.luogu.com.cn/problem/P2778）枚举圆与点的位置关系
P3021 [USACO11MAR]Bovine Bridge Battle S（https://www.luogu.com.cn/problem/P3021）容斥原理计数加枚举中心对称点
P1257 平面上的最接近点对（https://www.luogu.com.cn/problem/P1257）经典平面点集最近点对问题使用分治求解、还有哈希分块、有序列表
P7883 平面最近点对（加强加强版）（https://www.luogu.com.cn/problem/P7883）经典平面点集最近点对问题使用分治求解、还有哈希分块、有序列表
P1429 平面最近点对（加强版）（https://www.luogu.com.cn/problem/P1429）经典平面点集最近点对问题使用分治求解、还有哈希分块、有序列表



================================CodeForces================================
D. Pair Of Lines (https://codeforces.com/contest/961/problem/D) 抽屉原理枚举初始共线点并计算其他点的共线性情况
D. Tricky Function（https://codeforces.com/problemset/problem/429/D）经典平面点集最近点对

================================AcWing====================================
119. 袭击（https://www.acwing.com/problem/content/121/）经典平面点集最近点对问题使用分治求解、还有哈希分块、有序列表


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

    @staticmethod
    def line_intersection_line(start1: List[int], end1: List[int], start2: List[int], end2: List[int]) -> List[float]:
        # 模板：计算两条线段最靠下最靠左的交点，没有交点则返回空
        x1, y1 = start1
        x2, y2 = end1
        x3, y3 = start2
        x4, y4 = end2
        det = lambda a, b, c, d: a * d - b * c
        d = det(x1 - x2, x4 - x3, y1 - y2, y4 - y3)
        p = det(x4 - x2, x4 - x3, y4 - y2, y4 - y3)
        q = det(x1 - x2, x4 - x2, y1 - y2, y4 - y2)
        if d != 0:
            lam, eta = p / d, q / d
            if not (0 <= lam <= 1 and 0 <= eta <= 1):
                return []
            return [lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2]
        if p != 0 or q != 0:
            return []
        t1, t2 = sorted([start1, end1]), sorted([start2, end2])
        if t1[1] < t2[0] or t2[1] < t1[0]:
            return []
        return max(t1[0], t2[0])


class ClosetPair:
    def __init__(self):
        return

    @staticmethod
    def bucket_grid(n: int, nums: List[List[int]]):

        # 模板：使用随机增量法分网格计算平面最近的点对
        def dis(p1, p2):
            return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])

        def check(p):
            nonlocal ss
            return p[0] // ss, p[1] // ss

        def update(buck, ind):
            nonlocal dct
            if buck not in dct:
                dct[buck] = []
            dct[buck].append(ind)
            return

        assert n >= 2
        random.shuffle(nums)

        # 初始化
        dct = dict()
        ans = dis(nums[0], nums[1])
        ss = ans ** 0.5
        update(check(nums[0]), 0)
        update(check(nums[1]), 1)
        if ans == 0:
            return 0

        # 遍历进行随机增量
        for i in range(2, n):
            a, b = check(nums[i])
            res = ans
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    cur = (x + a, y + b)
                    if cur in dct:
                        for j in dct[cur]:
                            now = dis(nums[i], nums[j])
                            res = res if res < now else now
            if res == 0:  # 距离为 0 直接返回
                return 0
            if res < ans:
                # 重置初始化
                ans = res
                ss = ans ** 0.5
                dct = dict()
                for x in range(i + 1):
                    update(check(nums[x]), x)
            else:
                update(check(nums[i]), i)
        # 返回值为欧几里得距离的平方
        return ans

    @staticmethod
    def divide_and_conquer(lst):

        # 模板：使用分治求解平面最近点对
        lst.sort(key=lambda p: p[0])

        def distance(p1, p2):
            return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])

        def conquer(point_set, min_dis, mid):
            smallest = min_dis
            point_mid = point_set[mid]
            pt = []
            for point in point_set:
                if (point[0] - point_mid[0]) ** 2 <= min_dis:
                    pt.append(point)
            pt.sort(key=lambda x: x[1])
            for i in range(len(pt)):
                for j in range(i + 1, len(pt)):
                    if (pt[i][1] - pt[j][1]) ** 2 >= min_dis:
                        break
                    cur = distance(pt[i], pt[j])
                    smallest = smallest if smallest < cur else cur
            return smallest

        def check(point_set):
            min_dis = float("inf")
            if len(point_set) <= 3:
                n = len(point_set)
                for i in range(n):
                    for j in range(i + 1, n):
                        cur = distance(point_set[i], point_set[j])
                        min_dis = min_dis if min_dis < cur else cur
                return min_dis
            mid = len(point_set) // 2
            left = point_set[:mid]
            right = point_set[mid + 1:]
            min_left_dis = check(left)
            min_right_dis = check(right)
            min_dis = min_left_dis if min_left_dis < min_right_dis else min_right_dis
            merge_dis = conquer(point_set, min_dis, mid)
            return min_dis if min_dis < merge_dis else merge_dis

        return check(lst)

    @staticmethod
    def sorted_pair(points) -> float:
        # 模板：使用有序列表进行平面最近点对计算
        def dis(p1, p2):
            return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])

        points.sort(key=lambda p: [p[0], [1]])
        lst1 = LocalSortedList()
        lst2 = LocalSortedList()
        ans = inf
        ss = ans ** 0.5
        n = len(points)
        for i in range(n):
            x, y = points[i]
            while lst1 and abs(x - lst1[0][0]) >= ss:
                a, b = lst1.pop(0)
                lst2.discard((b, a))
            while lst1 and abs(x - lst1[-1][0]) >= ss:
                a, b = lst1.pop()
                lst2.discard((b, a))

            ind = lst2.bisect_left((y - ss, -inf))
            while ind < len(lst2) and abs(y - lst2[ind][0]) <= ss:
                res = dis([x, y], lst2[ind][::-1])
                ans = ans if ans < res else res
                ind += 1

            ss = ans ** 0.5
            lst1.add((x, y))
            lst2.add((y, x))
        return ans

    @staticmethod
    def bucket_grid_inter_set(n: int, nums1: List[List[int]], nums2):

        # 模板：使用随机增量法分网格计算两个平面点集最近的点对
        def dis(p1, p2):
            return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])

        def check(p):
            nonlocal ss
            return p[0] // ss, p[1] // ss

        def update(buck, ind):
            nonlocal dct
            if buck not in dct:
                dct[buck] = []
            dct[buck].append(ind)
            return

        random.shuffle(nums1)
        random.shuffle(nums2)

        # 初始化
        dct = dict()
        ans = inf
        for i in range(n):
            cur = dis(nums1[i], nums2[0])
            ans = cur if ans > cur else ans
        ss = ans ** 0.5
        if ans == 0:
            return 0
        for i in range(n):
            update(check(nums1[i]), i)
        # 遍历进行随机增量
        for i in range(1, n):
            a, b = check(nums2[i])
            res = ans
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    cur = (x + a, y + b)
                    if cur in dct:
                        for j in dct[cur]:
                            now = dis(nums2[i], nums1[j])
                            res = res if res < now else now
            if res == 0:  # 距离为 0 直接返回
                return 0
            if res < ans:
                # 重置初始化
                ans = res
                ss = ans ** 0.5
                dct = dict()
                for x in range(n):
                    update(check(nums1[x]), x)
        # 返回值为欧几里得距离的平方
        return ans


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1603(start1: List[int], end1: List[int], start2: List[int], end2: List[int]) -> List[float]:
        # 模板：计算两条线段之间的最靠左靠下的交点
        gm = Geometry()
        return gm.line_intersection_line(start1, end1, start2, end2)

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

    @staticmethod
    def cf_429d(ac=FastIO()):
        # 模板：转换为求解平面最近点对
        n = ac.read_int()
        nums = ac.read_list_ints()
        n = int(n)
        nums = list(accumulate(nums, add))
        nums = [[i, nums[i]] for i in range(n)]
        # ans = ClosetPair().bucket_grid(n, nums) ** 0.5
        # ans = ClosetPair().divide_and_conquer(nums) ** 0.5
        ans = ClosetPair().sorted_pair(nums)
        ac.st(ans)
        return

    @staticmethod
    def ac_119(ac=FastIO()):
        # 模板：使用随机增量法经典计算平面两个点集之间的最近距离
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums1 = [ac.read_list_ints() for _ in range(n)]
            nums2 = [ac.read_list_ints() for _ in range(n)]
            ans = ClosetPair().bucket_grid_inter_set(n, nums1, nums2)
            ac.st("%.3f" % (ans**0.5))
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
