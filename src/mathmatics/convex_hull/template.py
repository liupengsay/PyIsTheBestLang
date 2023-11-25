import math
import random
from typing import List


class MinCircleOverlap:
    def __init__(self):
        self.pi = math.acos(-1)
        self.esp = 10 ** (-10)
        return

    def get_min_circle_overlap(self, points: List[List[int]]):
        # 模板：随机增量法求解最小圆覆盖

        def cross(a, b):
            return a[0] * b[1] - b[0] * a[1]

        def intersection_point(p1, v1, p2, v2):
            # 求解两条直线的交点
            u = (p1[0] - p2[0], p1[1] - p2[1])
            t = cross(v2, u) / cross(v1, v2)
            return p1[0] + v1[0] * t, p1[1] + v1[1] * t

        def is_point_in_circle(circle_x, circle_y, circle_r, x, y):
            res = math.sqrt((x - circle_x) ** 2 + (y - circle_y) ** 2)
            if abs(res - circle_r) < self.esp:
                return True
            if res < circle_r:
                return True
            return False

        def vec_rotate(v, theta):
            x, y = v
            return x * math.cos(theta) + y * math.sin(theta), -x * math.sin(theta) + y * math.cos(theta)

        def get_out_circle(x1, y1, x2, y2, x3, y3):
            xx1, yy1 = (x1 + x2) / 2, (y1 + y2) / 2
            vv1 = vec_rotate((x2 - x1, y2 - y1), self.pi / 2)
            xx2, yy2 = (x1 + x3) / 2, (y1 + y3) / 2
            vv2 = vec_rotate((x3 - x1, y3 - y1), self.pi / 2)
            pp = intersection_point((xx1, yy1), vv1, (xx2, yy2), vv2)
            res = math.sqrt((pp[0] - x1) ** 2 + (pp[1] - y1) ** 2)
            return pp[0], pp[1], res

        random.shuffle(points)
        n = len(points)
        p = points

        # 圆心与半径
        cc1 = (p[0][0], p[0][1], 0)
        for ii in range(1, n):
            if not is_point_in_circle(cc1[0], cc1[1], cc1[2], p[ii][0], p[ii][1]):
                cc2 = (p[ii][0], p[ii][1], 0)
                for jj in range(ii):
                    if not is_point_in_circle(cc2[0], cc2[1], cc2[2], p[jj][0], p[jj][1]):
                        dis = math.sqrt((p[jj][0] - p[ii][0]) ** 2 + (p[jj][1] - p[ii][1]) ** 2)
                        cc3 = ((p[jj][0] + p[ii][0]) / 2, (p[jj][1] + p[ii][1]) / 2, dis / 2)
                        for kk in range(jj):
                            if not is_point_in_circle(cc3[0], cc3[1], cc3[2], p[kk][0], p[kk][1]):
                                cc3 = get_out_circle(p[ii][0], p[ii][1], p[jj][0], p[jj][1], p[kk][0], p[kk][1])
                        cc2 = cc3
                cc1 = cc2

        return cc1
