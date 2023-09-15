import bisect
import math
import unittest
import random
from collections import defaultdict
from typing import List
from decimal import Decimal
from src.fast_io import FastIO
from math import inf


"""
算法：三分查找求一维极值、三分套三分求二维极值、梯度下降法、爬山法、ternary search三元搜索
功能：用来寻找区间至多具有一个峰顶点或者一个谷底点的函数极值解
题目：

===================================洛谷===================================
1515. 服务中心的最佳位置（https://leetcode.cn/problems/best-position-for-a-service-centre/）三分套三分求凸函数极小值，也可以使用梯度下降法与爬山法求解

===================================洛谷===================================
P3382 【模板】三分法（https://www.luogu.com.cn/problem/P3382）利用三分求区间函数极值点
P1883 函数（https://www.luogu.com.cn/problem/P1883）三分求下凸函数最小值

================================CodeForces================================
E. Maximize!（https://codeforces.com/problemset/problem/939/E）贪心使用双指针或者三分进行求解，整数函数最大值
D. Devu and his Brother（http://codeforces.com/problemset/problem/439/D）利用单调性变换使用三分查找求解

================================AtCoder================================
F - Minimum Bounding Box（https://atcoder.jp/contests/abc130/tasks/abc130_f）三分模板题求函数最小值需要高精度

参考：OI WiKi（xx）
"""


class TriPartSearch:
    # 三分查找
    def __init__(self):
        return

    @staticmethod
    def find_ceil_point_float(fun, left, right, error=1e-9, high_precision=False):

        # 求解上凸函数函数取得最大值时的点
        while left < right - error:
            diff = Decimal(right - left) / 3 if high_precision else (right - left) / 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 > dist2:
                right = mid2
            elif dist1 < dist2:
                left = mid1
            else:
                left = mid1
                right = mid2
        return left

    @staticmethod
    def find_ceil_point_int(fun, left, right, error=1):
        # 求解上凸函数函数取得最大值时的点
        while left < right - error:
            diff = (right - left) // 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 > dist2:
                right = mid2
            elif dist1 < dist2:
                left = mid1
            else:
                left = mid1
                right = mid2
        return left

    @staticmethod
    def find_floor_point_float(fun, left, right, error=1e-9, high_precision=False):

        # 求解下凸函数取得最小值时的点
        while left < right - error:
            diff = Decimal(right - left) / 3 if high_precision else (right - left) / 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 < dist2:
                right = mid2
            elif dist1 > dist2:
                left = mid1
            else:
                left = mid1
                right = mid2
        return left

    @staticmethod
    def find_ceil_value_float(fun, left, right, error=1e-9, high_precision=False):

        # 求解上凸函数取得的最大值
        f1, f2 = fun(left), fun(right)
        while abs(f1 - f2) > error:
            diff = Decimal(right - left) / 3 if high_precision else (right - left) / 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 > dist2:
                right = mid2
                f2 = dist2
            elif dist1 < dist2:
                left = mid1
                f1 = dist1
            else:
                left = mid1
                right = mid2
                f1, f2 = dist1, dist2
        return (f1 + f2) / 2

    @staticmethod
    def find_floor_value_float(fun, left, right, error=1e-9, high_precision=False):

        # 求解下凸函数取得的最小值
        f1, f2 = fun(left), fun(right)
        while abs(f1 - f2) > error:
            diff = Decimal(right - left) / 3 if high_precision else (right - left) / 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 < dist2:
                right = mid2
                f2 = dist2
            elif dist1 > dist2:
                left = mid1
                f1 = dist1
            else:
                left = mid1
                right = mid2
                f1, f2 = dist1, dist2
        return (f1 + f2) / 2


class TriPartPackTriPart:
    # 模板：三分套三分
    def __init__(self):
        return

    @staticmethod
    def find_floor_point_float(target, left_x, right_x, low_y, high_y):
        # 求最小的坐标[x,y]使得目标函数target最小
        error = 5e-8
        
        def optimize(y):
            # 套三分里面的损失函数
            low_ = left_x
            high_ = right_x
            while low_ < high_ - error:
                diff_ = (high_ - low_) / 3
                mid1_ = low_ + diff_
                mid2_ = low_ + 2 * diff_
                dist1_ = target(mid1_, y)
                dist2_ = target(mid2_, y)
                if dist1_ < dist2_:
                    high_ = mid2_
                elif dist1_ > dist2_:
                    low_ = mid1_
                else:
                    low_ = mid1_
                    high_ = mid2_
            return low_, target(low_, y)

        low = low_y
        high = high_y
        while low < high - error:
            diff = (high - low) / 3
            mid1 = low + diff
            mid2 = low + 2 * diff
            _, dist1 = optimize(mid1)
            _, dist2 = optimize(mid2)
            if dist1 < dist2:
                high = mid2
            elif dist1 > dist2:
                low = mid1
            else:
                low = mid1
                high = mid2
        res_x, r = optimize(low)
        res_y = low
        return [res_x, res_y, math.sqrt(r)]


class Solution:
    def __init__(self):
        return

    @staticmethod
    def abc_130f(ac=FastIO()):
        # 模板：三分模板题求函数最小值需要高精度
        n = ac.read_int()
        ind = {"L": [-1, 0], "R": [1, 0], "U": [0, 1], "D": [0, -1]}
        dct_x = defaultdict(lambda: [inf, -inf])
        dct_y = defaultdict(lambda: [inf, -inf])
        for _ in range(n):
            x, y, d = ac.read_list_strs()
            x = int(x)
            y = int(y)

            a, b = dct_x[d]
            a = a if a < x else x
            b = b if b > x else x
            dct_x[d] = [a, b]

            a, b = dct_y[d]
            a = a if a < y else y
            b = b if b > y else y
            dct_y[d] = [a, b]

        lst_x = []
        for d in ind:
            if dct_x[d][0] < inf:
                for x in dct_x[d]:
                    lst_x.append([x, ind[d][0]])

        lst_y = []
        for d in ind:
            if dct_y[d][0] < inf:
                for y in dct_y[d]:
                    lst_y.append([y, ind[d][1]])

        def check(t):
            dis_x = [xx+t*aa for xx, aa in lst_x]
            dis_y = [xx+t*aa for xx, aa in lst_y]
            x_low = min(dis_x)
            x_high = max(dis_x)
            y_low = min(dis_y)
            y_high = max(dis_y)
            return (x_high-x_low)*(y_high-y_low)

        ans = TriPartSearch().find_floor_value_float(check, 0, 10**8, error=1e-10, high_precision=True)
        ac.st(ans)
        return

    @staticmethod
    def lg_1883(ac=FastIO()):
        # 模板：三分模板题求函数最小值，不需要高精度
        t = ac.read_int()

        for _ in range(t):
            n = ac.read_int()
            nums = [ac.read_list_ints() for _ in range(n)]

            def fun(x):
                return max(a * x * x + b * x + c for a, b, c in nums)

            ans = TriPartSearch().find_floor_value_float(fun, 0, 1000)
            ans = ac.round_5(ans * 10 ** 4)
            ans /= 10 ** 4
            ac.st("%.4f" % ans)
        return

    @staticmethod
    def lg_p3382(ac=FastIO()):
        # 模板：三分查找取得最大值的函数点
        n, l, r = ac.read_floats()
        n = int(n)
        lst = ac.read_list_floats()
        lst.reverse()

        def check(x):
            res = lst[0]
            mul = 1
            for i in range(1, n + 1):
                mul *= x
                res += mul * lst[i]
            return res

        ans = TriPartSearch().find_ceil_point_float(check, l, r)
        ac.st(ans)
        return

    @staticmethod
    def cf_439d(ac=FastIO()):
        # 模板：求函数取得最小值时的点
        n, m = ac.read_ints()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        a.sort()
        b.sort()

        pre_a = [0] * (n + 1)
        for i in range(n):
            pre_a[i + 1] = pre_a[i] + a[i]

        pre_b = [0] * (m + 1)
        for i in range(m):
            pre_b[i + 1] = pre_b[i] + b[i]

        floor = min(a)
        ceil = max(b)
        if floor >= ceil:
            ac.st(0)
            return

        def check(xx):
            ii = bisect.bisect_right(a, xx)
            cost = ii * xx - pre_a[ii]

            ii = bisect.bisect_left(b, xx)
            cost += pre_b[-1] - pre_b[ii] - (m - ii) * xx
            return cost

        point = int(TriPartSearch().find_floor_point_float(check, floor, ceil, 1))

        ans = inf
        for x in [-1, 0, 1]:
            if floor <= point + x <= ceil:
                ans = ac.min(ans, check(point + x))
        ac.st(ans)
        return

    @staticmethod
    def cf_939e(ac=FastIO()):
        # 模板：整数三分查找，计算上凸函数最大值
        nums = []
        pre = [0]

        def check(xx):
            return nums[-1] - (pre[xx] + nums[-1]) / (xx + 1)

        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            if len(lst) == 2:
                nums.append(lst[1])
                pre.append(pre[-1] + nums[-1])
            else:
                n = len(nums)
                x = TriPartSearch().find_ceil_point_int(check, 0, n - 1)
                ans = -inf
                for y in [x - 1, x, x + 1, x + 2]:
                    if 0 <= y <= n - 1:
                        ans = ac.max(ans, check(y))
                ac.st(ans)
        return

    @staticmethod
    def lc_1515_1(stack: List[List[int]]) -> float:

        # 模板：三分套三分求凸函数极值
        def target(x, y):
            return sum([math.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2) for p in stack])

        low_x = min([p[0] for p in stack])
        high_x = max([p[0] for p in stack])
        low_y = min([p[1] for p in stack])
        high_y = max([p[1] for p in stack])
        _, _, ans = TriPartPackTriPart().find_floor_point_float(target, low_x, high_x, low_y, high_y)
        return ans ** 2

    @staticmethod
    def lc_1515_2(positions: List[List[int]]) -> float:
        # 模板：梯度下降法求解凸函数极值
        eps = 1e-10
        alpha = 1.0
        decay = 0.001
        n = len(positions)
        batch_size = n
        x = sum(pos[0] for pos in positions) / n
        y = sum(pos[1] for pos in positions) / n

        while True:
            x_pre, y_pre = x, y
            random.shuffle(positions)
            for i in range(0, n, batch_size):
                dx = dy = 0.0
                j = i+batch_size if i+batch_size<n else n
                for k in range(i, j):
                    pos = positions[k]
                    dx += (x-pos[0])/((x-pos[0])*(x-pos[0])+(y-pos[1])*(y-pos[1])+eps)**0.5
                    dy += (y-pos[1])/((x-pos[0])*(x-pos[0])+(y-pos[1])*(y-pos[1])+eps)**0.5
                x -= alpha*dx
                y -= alpha*dy
            alpha *= (1-decay)
            if ((x-x_pre)*(x-x_pre)+(y-y_pre)*(y-y_pre))**0.5 < eps:
                break

        ans = sum(((x-x0)*(x-x0)+(y-y0)*(y-y0))**0.5 for x0, y0 in positions)
        return ans

    @staticmethod
    def lc_1515_3(positions: List[List[int]]) -> float:
        # 模板：爬山法计算凸函数极小值
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        eps = 1e-6
        step = 1.0
        decay = 0.8
        n = len(positions)

        x = sum(pos[0] for pos in positions) / n
        y = sum(pos[1] for pos in positions) / n

        def get_dis(xc, yc):
            return sum(((xc - x0) * (xc - x0) + (yc - y0) * (yc - y0)) ** 0.5 for x0, y0 in positions)

        while step > eps:
            for dx, dy in dirs:
                x_next = x + step * dx
                y_next = y + step * dy
                if get_dis(x_next, y_next) < get_dis(x, y):
                    x, y = x_next, y_next
                    break
            else:
                step *= (1.0 - decay)
        return get_dis(x, y)


class TestGeneral(unittest.TestCase):

    def test_tri_part_search(self):
        tps = TriPartSearch()
        def fun1(x): return (x - 1) * (x - 1)
        assert abs(tps.find_floor_point_float(fun1, -5, 100) - 1) < 1e-5

        def fun2(x): return -(x - 1) * (x - 1)
        assert abs(tps.find_ceil_point_float(fun2, -5, 100) - 1) < 1e-5
        return

    def test_tri_part_pack_tri_part(self):
        tpt = TriPartPackTriPart()
        nodes = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        # 定义目标函数
        def target(x, y): return max([(x - p[0]) ** 2 + (y - p[1]) ** 2 for p in nodes])
        x0, y0, _ = tpt.find_floor_point_float(target, -10, 10, -10, 10)
        assert abs(x0) < 1e-5 and abs(y0) < 1e-5
        return


if __name__ == '__main__':
    unittest.main()
