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
1603（https://leetcode.cn/problems/intersection-lcci/description/）geometry

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
P2449（https://www.luogu.com.cn/problem/P2449）rectangle_overlap|rectangle_edge_touch|rectangle_corner_touch|geometry
P3844（https://www.luogu.com.cn/problem/P3844）geometry|implemention
P6341（https://www.luogu.com.cn/problem/P6341）line_scope|brute_force|right_triangle

===================================CodeForces===================================
961D（https://codeforces.com/contest/961/problem/D)）pigeonhole_principle|brute_force|line_slope|collinearity
429D（https://codeforces.com/contest/429/problem/D）closet_pair|divide_and_conquer|hash|block_plane|sorted_list|classical
1133D（https://codeforces.com/contest/1133/problem/D）line_slope
1979E（https://codeforces.com/contest/1979/problem/E）manhattan_distance|chebyshev_distance|brute_force|two_pointers|map
1C（https://codeforces.com/contest/1/problem/C）geometry|circle|triangle
1354C1（https://codeforces.com/problemset/problem/1354/C1）geometry
1354C2（https://codeforces.com/problemset/problem/1354/C2）geometry
1552C（https://codeforces.com/problemset/problem/1552/C）geometry
598c（https://codeforces.com/problemset/problem/598/C）math|geometry|high_precision|angle_with_x_axis|angle_between_vectors

===================================AtCoder===================================
ABC343E（https://atcoder.jp/contests/abc343/tasks/abc343_e）brute_force|brain_teaser|inclusion_exclusion|math|classical
ABC292F（https://atcoder.jp/contests/abc292/tasks/abc292_f）brain_teaser|math
ABC275C（https://atcoder.jp/contests/abc275/tasks/abc275_c）brute_force|geometry|square|angle|classical
ABC266C（https://atcoder.jp/contests/abc266/tasks/abc266_c）math|geometry|is_convex_quad|classical
ABC250F（https://atcoder.jp/contests/abc250/tasks/abc250_f）geometry|circular_array|two_pointers|brain_teaser
ABC234H（https://atcoder.jp/contests/abc234/tasks/abc234_h）closest_pair|brain_teaser|classical
ABC351E（https://atcoder.jp/contests/abc351/tasks/abc351_e）chebyshev_distance|manhattan_distance|brain_teaser|tree_array|classical
ABC226D（https://atcoder.jp/contests/abc226/tasks/abc226_d）geometry|linear_scope|classical
ABC224C（https://atcoder.jp/contests/abc224/tasks/abc224_c）geometry
ABC218D（https://atcoder.jp/contests/abc218/tasks/abc218_d）brute_force|rectangle
ABC354D（https://atcoder.jp/contests/abc354/tasks/abc354_d）brute_force|inclusion_exclusion
ABC361C（https://atcoder.jp/contests/abc361/tasks/abc361_c）geometry
ABC362B（https://atcoder.jp/contests/abc362/tasks/abc362_b）geometry|linear_scope|classical|vertical_triangular
ABC197D（https://atcoder.jp/contests/abc197/tasks/abc197_d）geometry

=====================================AcWing=====================================
119（https://www.acwing.com/problem/content/121/）closet_pair|divide_and_conquer|hash|block_plane|sorted_list|classical
4309（https://www.acwing.com/problem/content/4312/）line_slope
4499（https://www.acwing.com/problem/content/4502/）geometry|equation
（https://www.hackerrank.com/contests/2023-1024-1/challenges/challenge-4219）collinearity|random

"""
import math
from collections import defaultdict, Counter
from itertools import accumulate, pairwise
from math import inf
from typing import List

from src.graph.union_find.template import UnionFind
from src.mathmatics.geometry.template import Geometry, ClosetPair
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1603(start1: List[int], end1: List[int], start2: List[int], end2: List[int]) -> List[float]:
        """
        url: https://leetcode.cn/problems/intersection-lcci/
        tag: geometry
        """
        gm = Geometry()
        return gm.line_intersection_line(start1, end1, start2, end2)

    @staticmethod
    def lc_2280(stock: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-lines-to-represent-a-line-chart/
        tag: line_slope
        """
        stock.sort()
        gm = Geometry()
        pre = [-1, -1]
        ans = 0
        for (x1, y1), (x2, y2) in pairwise(stock):
            cur = gm.compute_slope(x1, y1, x2, y2)
            ans += pre != cur
            pre = cur
        return ans

    @staticmethod
    def lc_149(points: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/max-points-on-a-line/
        tag: line_slope|brute_force|classical
        """
        ans = 0
        n = len(points)
        gm = Geometry()
        for i in range(n):
            dct = defaultdict(int)
            dct[0] = 0
            x1, y1 = points[i]
            for x2, y2 in points[i + 1:]:
                dct[gm.compute_slope(x1, y1, x2, y2)] += 1
            ans = max(ans, max(dct.values()) + 1)
        return ans

    @staticmethod
    def lg_p1665(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1665
        tag: brute_force|diagonal|square
        """

        n = ac.read_int()
        lst = [ac.read_list_ints() for _ in range(n)]
        dct = set(tuple(p) for p in lst)
        ans = 0
        m = len(lst)
        gm = Geometry()
        for i in range(m):
            x1, y1 = lst[i]
            for j in range(i + 1, m):
                x2, y2 = lst[j]
                point1, point2 = gm.compute_square_point_non_vertical(x1, y1, x2, y2)

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
        url: https://codeforces.com/contest/429/problem/D
        tag: closet_pair|divide_and_conquer|hash|block_plane|sorted_list|classical
        """

        n = ac.read_int()
        nums = ac.read_list_ints()
        n = int(n)
        nums = list(accumulate(nums))
        nums = [[i, nums[i]] for i in range(n)]
        # ans = ClosetPair().bucket_grid(n, nums)
        # ans = ClosetPair().divide_and_conquer(nums)
        ans = ClosetPair().sorted_pair(nums)
        ac.st(ans)
        return

    @staticmethod
    def ac_119(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/121/
        tag: closet_pair|divide_and_conquer|hash|block_plane|sorted_list|classical
        """

        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums1 = [ac.read_list_ints() for _ in range(n)]
            nums2 = [ac.read_list_ints() for _ in range(n)]
            ans = ClosetPair().bucket_grid_between_two_sets(n, nums1, nums2)
            ac.st("%.3f" % (ans ** 0.5))
        return

    @staticmethod
    def lc_1453(darts: List[List[int]], r: int) -> int:
        """
        url: https://leetcode.cn/problems/maximum-number-of-darts-inside-of-a-circular-dartboard/
        tag: circle|classical|circle_center
        """

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
        n, x1, y1 = ac.read_list_ints()
        dct = set()
        gm = Geometry()
        for _ in range(n):
            x2, y2 = ac.read_list_ints()
            dct.add(gm.compute_slope(x1, y1, x2, y2))
        ac.st(len(dct))
        return

    @staticmethod
    def ac_4499(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4502/
        tag: geometry|equation
        """

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

    @staticmethod
    def abc_343e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc343/tasks/abc343_e
        tag: brute_force|brain_teaser|inclusion_exclusion|math|classical
        """

        def three(x, y, z, xx, yy, zz, xxx, yyy, zzz):
            res = 1
            res *= max(0, min(x, xx, xxx) + 7 - max(x, xx, xxx))
            res *= max(0, min(y, yy, yyy) + 7 - max(y, yy, yyy))
            res *= max(0, min(z, zz, zzz) + 7 - max(z, zz, zzz))
            return res

        def two(x, y, z, xx, yy, zz):
            res = 1
            res *= max(0, min(x, xx) + 7 - max(x, xx))
            res *= max(0, min(y, yy) + 7 - max(y, yy))
            res *= max(0, min(z, zz) + 7 - max(z, zz))
            return res

        a1 = b1 = c1 = 0
        low = -1
        high = 7
        v1, v2, v3 = ac.read_list_ints()

        for a2 in range(low, high + 1):
            for b2 in range(low, high + 1):
                for c2 in range(low, high + 1):
                    for a3 in range(low, high + 1):
                        for b3 in range(low, high + 1):
                            for c3 in range(low, high + 1):
                                inter3 = three(a1, b1, c1, a2, b2, c2, a3, b3, c3)
                                inter2 = (two(a1, b1, c1, a2, b2, c2)
                                          + two(a1, b1, c1, a3, b3, c3)
                                          + two(a2, b2, c2, a3, b3, c3)
                                          - inter3 * 3)
                                inter1 = 3 * (7 * 7 * 7) - 2 * inter2 - 3 * inter3
                                if (v1, v2, v3) == (inter1, inter2, inter3):
                                    ac.yes()
                                    ac.lst([a1, b1, c1, a2, b2, c2, a3, b3, c3])
                                    return
        ac.no()
        return

    @staticmethod
    def abc_275c(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc275/tasks/abc275_c
        tag: brute_force|geometry|square|angle|classical
        """
        grid = [ac.read_str() for _ in range(9)]

        ind = []
        for i in range(9):
            for j in range(9):
                if grid[i][j] == "#":
                    ind.append((i, j))

        def dis(p1, p2):
            return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

        gm = Geometry()
        points = set(ind)
        ans = 0
        k = len(ind)
        for i in range(k):
            x0, y0 = ind[i]
            for j in range(i + 1, k):
                x2, y2 = ind[j]
                (x1, y1), (x3, y3) = gm.compute_square_point_non_vertical(x0, y0, x2, y2)
                x1 = int(x1)
                y1 = int(y1)
                x3 = int(x3)
                y3 = int(y3)
                if (x1, y1) in points and (x3, y3) in points:
                    perm = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
                    if len(set(dis(perm[i], perm[i + 1]) for i in range(3))) == 1:
                        if gm.vertical_angle(perm[0], perm[1], perm[2]):
                            ans += 1
        ac.st(ans // 2)
        return

    @staticmethod
    def abc_266c(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc266/tasks/abc266_c
        tag: math|geometry|is_convex_quad|classical
        """
        points = [ac.read_list_ints() for _ in range(4)]
        ac.st("Yes" if Geometry().is_convex_quad(points) else "No")
        return

    @staticmethod
    def abc_248e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc248/tasks/abc248_e
        tag: linear_scope|compute_slope|geometry|brute_force
        """
        n, k = ac.read_list_ints()
        points = [ac.read_list_ints() for _ in range(n)]
        dct = Counter((x, y) for x, y in points)
        if max(dct.values()) >= k:
            ac.st("Infinity")
            return
        ans = 0
        gm = Geometry()
        points = list(dct.keys())
        n = len(points)
        pre = set()
        for i in range(n):
            x1, y1 = points[i]
            cur = defaultdict(lambda: [(x1, y1)])
            for j in range(i + 1, n):
                x2, y2 = points[j]
                s = gm.compute_slope(x1, y1, x2, y2)
                if (x1, y1, s) not in pre:
                    cur[s].append((x2, y2))
            for s in cur:
                tot = sum(dct[t] for t in cur[s])
                ans += tot >= k
                pre |= {(x, y, s) for x, y in cur[s]}
        ac.st(ans)
        return

    @staticmethod
    def abc_250f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc250/tasks/abc250_f
        tag: geometry|circular_array|two_pointers|brain_teaser
        """
        gm = Geometry()
        n = ac.read_int()
        points = [ac.read_list_ints() for _ in range(n)]
        points += points
        tot = 0
        x1, y1 = points[0]
        for i in range(1, n - 1):
            x2, y2 = points[i]
            x3, y3 = points[i + 1]
            tot += gm.compute_triangle_area_double(x1, y1, x2, y2, x3, y3)
        ans2 = inf
        j = pre = 0
        for i in range(len(points) - 2):
            x1, y1 = points[i]
            if j < i + 1:
                j = i + 1
                x2, y2 = points[j]
                x3, y3 = points[j + 1]
                pre = gm.compute_triangle_area_double(x1, y1, x2, y2, x3, y3)
            ans2 = min(ans2, abs(tot - 4 * pre))
            while pre * 4 < tot and j + 2 < len(points):
                j += 1
                x2, y2 = points[j]
                x3, y3 = points[j + 1]
                pre += gm.compute_triangle_area_double(x1, y1, x2, y2, x3, y3)
                if pre < tot:
                    ans2 = min(ans2, abs(tot - 4 * pre))
            x2, y2 = points[i + 1]
            x3, y3 = points[j + 1]
            pre -= gm.compute_triangle_area_double(x1, y1, x2, y2, x3, y3)
        ac.st(ans2)
        return

    @staticmethod
    def abc_234h(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc234/tasks/abc234_h
        tag: closest_pair|brain_teaser|classical
        """
        n, k = ac.read_list_ints()
        points = [ac.read_list_ints() for _ in range(n)]
        dct = defaultdict(list)
        ans = []
        for i, (x, y) in enumerate(points):
            a, b = x // k, y // k
            for aa in range(-1, 2):
                for bb in range(-1, 2):
                    for j in dct[(a + aa, b + bb)]:
                        x0, y0 = points[j]
                        if (x - x0) * (x - x0) + (y - y0) * (y - y0) <= k * k:
                            ans.append((j + 1, i + 1))
            dct[(a, b)].append(i)
        ac.st(len(ans))
        ans.sort()
        for a in ans:
            ac.lst(a)
        return

    @staticmethod
    def abc_351e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc351/tasks/abc351_e
        tag: chebyshev_distance|manhattan_distance|brain_teaser|tree_array|classical
        """
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]

        def check(lst):
            if not lst:
                return 0
            x, y = lst[0]
            w = (x - y) % 2
            cur = [(x - y + w) // 2 for x, y in lst]
            cur.sort()
            res = pre = 0
            for i, val in enumerate(cur):
                pre += val
                res += (i + 1) * val - pre
            cur = [(x + y + w) // 2 for x, y in lst]
            cur.sort()
            pre = 0
            for i, val in enumerate(cur):
                pre += val
                res += (i + 1) * val - pre
            return res

        ans = sum(check([[x, y] for x, y in nums if (x - y) % 2 == w]) for w in range(2))
        ac.st(ans)
        return

    @staticmethod
    def abc_226d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc226/tasks/abc226_d
        tag: geometry|linear_scope|classical
        """
        n = ac.read_int()
        points = [ac.read_list_ints() for _ in range(n)]
        ans = set()
        for i in range(n):
            x1, y1 = points[i]
            for j in range(n):
                if i == j:
                    continue
                x2, y2 = points[j]
                a = x2 - x1
                b = y2 - y1
                if b == 0:
                    a = -1 if a < 0 else 1
                elif a == 0:
                    b = -1 if b < 0 else 1
                else:
                    g = math.gcd(abs(a), abs(b))
                    a //= g
                    b //= g
                ans.add((a, b))
        ac.st(len(ans))
        return

    @staticmethod
    def abc_224c(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc224/tasks/abc224_c
        tag: geometry
        """
        n = ac.read_int()
        ans = 0
        points = [ac.read_list_ints() for _ in range(n)]
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    if not Geometry().same_line(points[i], points[j], points[k]):
                        ans += 1
        ac.st(ans)
        return

    @staticmethod
    def abc_224c_2(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc224/tasks/abc224_c
        tag: geometry
        """
        n = ac.read_int()
        points = [ac.read_list_ints() for _ in range(n)]
        ans = n * (n - 1) * (n - 2) // 6
        for i in range(n - 1):
            dct = defaultdict(int)
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dct[Geometry().compute_slope(x1, y1, x2, y2)] += 1
            for va in dct.values():
                if va >= 2:
                    ans -= va * (va - 1) // 2
        ac.st(ans)
        return

    @staticmethod
    def abc_354d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc354/tasks/abc354_d
        tag: brute_force|inclusion_exclusion
        """
        a, b, c, d = ac.read_list_ints()
        c -= 1
        d -= 1
        w = c - a + 1
        h = d - b + 1
        ww = 4 * ((w + 3) // 4)
        hh = 4 * ((h + 3) // 4)
        grid = [[1, 2, 1, 0], [2, 1, 0, 1], [1, 2, 1, 0], [2, 1, 0, 1]]
        ans = 8 * (ww // 4) * (hh // 4) * 2
        lst = [3, 3, 1, 1]
        for rr in range(a + w, a + ww):
            ans -= (hh // 4) * 2 * lst[rr % 4]
        lst = [2, 2, 2, 2]
        for cc in range(b + h, b + hh):
            ans -= (ww // 4) * 2 * lst[cc % 4]

        for xx in range(a + w, a + ww):
            for yy in range(b + h, b + hh):
                rr = xx % 4
                cc = yy % 4
                ans += grid[-cc - 1][rr]
        ac.st(ans)
        return

    @staticmethod
    def cf_1979e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1979/problem/E
        tag: manhattan_distance|chebyshev_distance|brute_force|two_pointers|map
        """

        def check():
            n, d = ac.read_list_ints()
            nums = []
            for _ in range(n):
                x, y = ac.read_list_ints()
                nums.append((x + y, x - y))

            for _ in range(2):
                dct = defaultdict(list)
                for i, (x, y) in enumerate(nums):
                    dct[x].append((i, y))
                for x in dct:
                    dct[x].sort(key=lambda it: it[1])
                keys = set(dct.keys())
                for x in keys:
                    for w in [x - d, x + d]:
                        if w in keys:
                            m = len(dct[w])
                            pre = dict()
                            j = 0
                            for i, y in dct[x]:
                                while j < m and dct[w][j][1] <= y + d:
                                    if dct[w][j][1] >= y and dct[w][j][1] - d in pre:
                                        return [i + 1, dct[w][j][0] + 1, pre[dct[w][j][1] - d] + 1]
                                    pre[dct[w][j][1]] = dct[w][j][0]
                                    j += 1
                nums = [ls[::-1] for ls in nums]
            return [0, 0, 0]

        for _ in range(ac.read_int()):
            ac.lst(check())
        return

    @staticmethod
    def cf_1c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1/problem/C
        tag: geometry|circle|triangle
        """

        x1, y1 = [float(x) for x in ac.read_list_strs()]
        x2, y2 = [float(x) for x in ac.read_list_strs()]
        x3, y3 = [float(x) for x in ac.read_list_strs()]

        x, y, r = Geometry().circumscribed_circle_of_triangle(x1, y1, x2, y2, x3, y3)

        theta1 = Geometry().get_circle_sector_angle(x1, y1, x2, y2, r)
        theta2 = Geometry().get_circle_sector_angle(x1, y1, x3, y3, r)
        theta3 = 2 * math.pi - theta1 - theta2

        lst = theta1 * 0.5 / math.pi, theta2 * 0.5 / math.pi, theta3 * 0.5 / math.pi

        n = 0
        error = 1e-5
        for d in range(3, 201):
            if all(abs(p * d - int(p * d + error)) < error for p in lst):
                n = d
                break
        theta = 2 * math.pi / n
        ans = n * 0.5 * (r ** 2) * math.sin(theta)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2449(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2449
        tag: rectangle_overlap|rectangle_edge_touch|rectangle_corner_touch|geometry
        """
        n = ac.read_int()
        uf = UnionFind(n)
        pre = []

        def check_overlap(rec1, rec2):
            x1, y1, x2, y2 = rec1
            x3, y3, x4, y4 = rec2
            if x2 <= x3 or x4 <= x1 or y4 <= y1 or y2 <= y3:
                return False
            return True

        def check_edge_touch(rec1, rec2):
            x1, y1, x2, y2 = rec1
            a1, b1, a2, b2 = rec2
            edge_touch = ((x1 == a2 or x2 == a1) and (y1 < b2 and y2 > b1)) or \
                         ((y1 == b2 or y2 == b1) and (x1 < a2 and x2 > a1))
            return edge_touch

        for i in range(n):
            a, b, c, d = ac.read_list_ints()
            pre.append([a, b, c, d])
            for j in range(i):
                if check_overlap(pre[i], pre[j]) or check_edge_touch(pre[i], pre[j]):
                    uf.union(i, j)
        ac.st(uf.part)
        return

    @staticmethod
    def lg_p3844(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3844
        tag: geometry|implemention
        """
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda it: it[2])
        color = [0] * n
        parent = [-1] * n
        area = [0] * n
        pre = []
        for i, (x, y, r) in enumerate(nums):
            cur = r * r
            for j in range(i):
                xx, yy, rr = nums[j]
                if (x - xx) ** 2 + (y - yy) ** 2 <= r * r:
                    if parent[j] == -1:
                        parent[j] = i
                        cur -= rr * rr
                    color[j] = 1 - color[j]
            color[i] = 1
            area[i] = cur
        ans = sum(area[i] for i in range(n) if color[i]) * math.pi
        ans = str(round(ans, 2))
        if "." not in ans:
            ans += ".00"
        else:
            while len(ans) - ans.index(".") < 3:
                ans += "0"
        ac.st(ans)
        return

    @staticmethod
    def lg_p6341(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6341
        tag: line_scope|brute_force|right_triangle
        """
        n = ac.read_int()  # MLE
        pos = [ac.read_list_ints() for _ in range(n)]
        row = defaultdict(int)
        col = defaultdict(int)
        for x, y in pos:
            row[x] += 1
            col[y] += 1
        ans = 0
        for x, y in pos:
            ans += (row[x] - 1) * (col[y] - 1)

        for i in range(n):
            pre = dict()
            x, y = pos[i]
            for j in range(n):
                if j != i:
                    a, b = pos[j]
                    if a == x or b == y:
                        continue
                    g = math.gcd(x - a, y - b)
                    aa = (x - a) // g
                    bb = (y - b) // g
                    if bb < 0:
                        bb *= -1
                        aa *= -1
                    aaa = -bb
                    bbb = aa
                    if bbb < 0:
                        bbb *= -1
                        aaa *= -1
                    ans += pre.get((aaa, bbb), 0)
                    pre[(aa, bb)] = pre.get((aa, bb), 0) + 1
        ac.st(ans)
        return

    @staticmethod
    def cf_598c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/598/C
        tag: math|geometry|high_precision|angle_with_x_axis|angle_between_vectors
        """
        gm = Geometry()
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        angle = [gm.angle_with_x_axis(x, y) for x, y in nums]
        ind = [(angle[i], i) for i in range(n)]
        ind.sort()
        ind = [i for _, i in ind]
        res = gm.angle_between_vector(nums[ind[0]], nums[ind[-1]])
        ans = [ind[0], ind[n - 1]]
        for i in range(1, n):
            cur = gm.angle_between_vector(nums[ind[i - 1]], nums[ind[i]])
            k1, k2 = cur
            k3, k4 = res
            if k1 * k4 > k2 * k3:
                res = cur
                ans = [ind[i - 1], ind[i]]
        ac.lst([x + 1 for x in ans])
        return
