"""
Algorithm：geometry|plane|closest_pair
Description：triangle|rectangle|square|line|circle|cube

====================================LeetCode====================================
149（https://leetcode.cn/problems/max-points-on-a-line/）line_slope|brute_force|classical
1453（https://leetcode.cn/problems/maximum-number-of-darts-inside-of-a-circular-dartboard/）circle|classical|circle_center
939（https://leetcode.cn/problems/minimum-area-rectangle/）brute_force|rectangle
16（https://leetcode.cn/problems/intersection-lcci/）line_segment|intersection
16（https://leetcode.cn/problems/best-line-lcci/）line_slope|brute_force|classical
2013（https://leetcode.cn/problems/detect-squares/）brute_force|hash|counter|square
2280（https://leetcode.cn/problems/minimum-lines-to-represent-a-line-chart/）line_slope
1401（https://leetcode.cn/problems/circle-and-rectangle-overlapping/）geometry|rectangle

=====================================LuoGu======================================
P1665（https://www.luogu.com.cn/problem/P1665）brute_force|diagonal|square
P2313（https://www.luogu.com.cn/problem/P2313）square|circle
P2358（https://www.luogu.com.cn/problem/P2358）geometry|cube
P2665（https://www.luogu.com.cn/problem/P2665）slope
P1355（https://www.luogu.com.cn/problem/P1355）triangle|area|location
P1142（https://www.luogu.com.cn/problem/P1142）line_slope|brute_force|classical
P2778（https://www.luogu.com.cn/problem/P2778）brute_force|circle|location
P3021（https://www.luogu.com.cn/problem/P3021）inclusion_exclusion|counter|brute_force
P1257（https://www.luogu.com.cn/problem/P1257）closet_pair|divide_and_conquer|hash|block_plane|sorted_list|classical
P7883（https://www.luogu.com.cn/problem/P7883）closet_pair|divide_and_conquer|hash|block_plane|sorted_list|classical
P1429（https://www.luogu.com.cn/problem/P1429）closet_pair|divide_and_conquer|hash|block_plane|sorted_list|classical


===================================CodeForces===================================
961D（https://codeforces.com/contest/961/problem/D)）pigeonhole_principle|brute_force|line_slope|collinearity
429D（https://codeforces.com/problemset/problem/429/D）closet_pair|divide_and_conquer|hash|block_plane|sorted_list|classical

=====================================AcWing=====================================
119（https://www.acwing.com/problem/content/121/）closet_pair|divide_and_conquer|hash|block_plane|sorted_list|classical
4309（https://www.acwing.com/problem/content/4312/）line_slope
4499（https://www.acwing.com/problem/content/4502/）geometry|equation
（https://www.hackerrank.com/contests/2023-1024-1/challenges/challenge-4219）collinearity|random

"""
import math
from audioop import add
from collections import defaultdict
from itertools import pairwise
from typing import List

from cytoolz import accumulate

from src.mathmatics.geometry.template import Geometry, ClosetPair
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1603(start1: List[int], end1: List[int], start2: List[int], end2: List[int]) -> List[float]:
        # 两条线段之间的最靠左靠下的交点
        gm = Geometry()
        return gm.line_intersection_line(start1, end1, start2, end2)

    @staticmethod
    def lc_2280(stock: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-lines-to-represent-a-line-chart/
        tag: line_slope
        """
        # 分数代表斜率
        stock.sort()
        pre = (-1, -1)
        ans = 0
        for (x, y), (a, b) in pairwise(stock):
            if x == a:
                cur = (x, -1)
            else:
                g = math.gcd(b - y, a - x)
                bb = (b - y) // g
                aa = (a - x) // g
                if aa < 0:
                    aa *= -1
                    bb *= -1
                cur = (bb, aa)
            ans += pre != cur
            pre = cur
        return ans

    @staticmethod
    def lc_149(points: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/max-points-on-a-line/
        tag: line_slope|brute_force|classical
        """
        # 两个不相同的点组成的line_slope
        ans = 0
        n = len(points)
        gm = Geometry()
        for i in range(n):
            dct = defaultdict(int)
            dct[0] = 0
            for j in range(i + 1, n):
                dct[gm.compute_slope2(points[i], points[j])] += 1
            ans = max(ans, max(dct.values()) + 1)
        return ans

    @staticmethod
    def lg_p1665(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1665
        tag: brute_force|diagonal|square
        """
        # 可以组成正方形的点的个数
        n = ac.read_int()
        lst = [ac.read_list_ints() for _ in range(n)]
        dct = set(tuple(p) for p in lst)
        ans = 0
        m = len(lst)
        for i in range(m):
            x1, y1 = lst[i]
            for j in range(i + 1, m):
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
        ac.st(ans // 2)
        return

    @staticmethod
    def cf_429d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/429/D
        tag: closet_pair|divide_and_conquer|hash|block_plane|sorted_list|classical
        """
        # 转换为求解closest_pair
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
        """
        url: https://www.acwing.com/problem/content/121/
        tag: closet_pair|divide_and_conquer|hash|block_plane|sorted_list|classical
        """
        # 随机增量法平面两个点集之间的最近距离
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums1 = [ac.read_list_ints() for _ in range(n)]
            nums2 = [ac.read_list_ints() for _ in range(n)]
            ans = ClosetPair().bucket_grid_inter_set(n, nums1, nums2)
            ac.st("%.3f" % (ans ** 0.5))
        return

    @staticmethod
    def lc_1453(darts: List[List[int]], r: int) -> int:
        """
        url: https://leetcode.cn/problems/maximum-number-of-darts-inside-of-a-circular-dartboard/
        tag: circle|classical|circle_center
        """
        # 经过两个不同的点与确定半径的两处圆心
        n = len(darts)
        ans = 1
        go = Geometry()
        for i in range(n):
            x1, y1 = darts[i]
            for j in range(i + 1, n):
                x2, y2 = darts[j]
                if (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) > 4 * r * r:
                    continue
                for x, y in go.compute_center(x1, y1, x2, y2, r):
                    cur = sum((x - x0) * (x - x0) + (y - y0) * (y - y0) <= r * r for x0, y0 in darts)
                    ans = ans if ans > cur else cur
        return ans

    @staticmethod
    def ac_4309(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4312/
        tag: line_slope
        """
        # line_slope
        n, x0, y0 = ac.read_list_ints()
        dct = set()
        for _ in range(n):
            x, y = ac.read_list_ints()
            g = math.gcd(x - x0, y - y0)
            a = (x - x0) // g
            b = (y - y0) // g
            if a == 0:
                dct.add(0)
                continue
            if a < 0:
                b *= -1
                a *= -1
            dct.add((a, b))
        ac.st(len(dct))
        return

    @staticmethod
    def ac_4499(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4502/
        tag: geometry|equation
        """
        # geometry|，一元二次方程求解
        r, x1, y1, x2, y2 = ac.read_list_ints()
        if (x1 - x2) ** 2 + (y1 - y2) ** 2 > r * r:
            ans = [x1, y1, r]
            ac.lst(["%.6f" % x for x in ans])
            return

        dis = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 + r
        ans = [0, 0, dis / 2]
        if x1 == x2:
            if y1 > y2:
                x0, y0 = x2, y2 + dis / 2
            else:
                x0, y0 = x2, y2 - dis / 2
            ans[0] = x0
            ans[1] = y0
        else:
            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1

            aa = k ** 2 + 1
            bb = -2 * k * (y2 - b) - 2 * x2
            cc = (y2 - b) ** 2 - dis ** 2 + x2 ** 2
            for xx in [(-bb + (bb * bb - 4 * aa * cc) ** 0.5) / 2 / aa,
                       (-bb - (bb * bb - 4 * aa * cc) ** 0.5) / 2 / aa]:
                yy = k * xx + b
                if int(x2 - xx > 0) == int(x2 - x1 > 0):
                    ans[0] = (xx + x2) / 2
                    ans[1] = (yy + y2) / 2
                    break
        ac.lst(["%.6f" % x for x in ans])
        return