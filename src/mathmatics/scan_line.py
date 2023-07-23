import unittest

from typing import List

import heapq

from src.fast_io import FastIO

"""
算法：扫描线
功能：计算平面几何面积或者立体体积
题目：

===================================力扣===================================
218. 天际线问题（https://leetcode.cn/problems/the-skyline-problem/）扫描线计算建筑物的轮廓
850. 矩形面积 II（https://leetcode.cn/problems/rectangle-area-ii/）扫描线计算覆盖面积，线段树加离散化应该有 nlogn 的解法

===================================洛谷===================================
P6265 [COCI2014-2015#3] SILUETA（https://www.luogu.com.cn/problem/P6265）计算建筑物的扫描线轮廓
P5490 【模板】扫描线（https://www.luogu.com.cn/problem/P5490）扫描线计算覆盖面积
P1884 [USACO12FEB] Overplanting S（https://www.luogu.com.cn/problem/P1884）扫描线计算覆盖面积
P1904 天际线（https://www.luogu.com.cn/problem/P1904）扫描线计算建筑物的轮廓

参考：[OI WiKi]（https://oi-wiki.org/geometry/scanning/)
"""


class ScanLine:
    def __init__(self):
        return

    @staticmethod
    def get_sky_line(buildings: List[List[int]]) -> List[List[int]]:

        # 模板：扫描线提取建筑物轮廓
        events = []
        # 生成左右端点事件并排序
        for left, right, height in buildings:
            # [x,y,h]分别为左右端点与高度
            events.append([left, -height, right])
            events.append([right, 0, 0])
        events.sort()

        # 初始化结果与前序高度
        res = [[0, 0]]
        stack = [[0, float('inf')]]
        for left, height, right in events:
            # 超出管辖范围的先出队
            while left >= stack[0][1]:
                heapq.heappop(stack)
            # 加入备选天际线队列
            if height < 0:
                heapq.heappush(stack, [height, right])
            # 高度发生变化出现新的关键点
            if res[-1][1] != -stack[0][0]:
                res.append([left, -stack[0][0]])
        return res[1:]

    @staticmethod
    def get_rec_area(rectangles: List[List[int]]) -> int:

        # 模板：扫描线提取矩形x轴的端点并排序（也可以取y轴的端点是一个意思）
        axis = set()
        # [x1,y1,x2,y2] 为左下角到右上角坐标
        for rec in rectangles:
            axis.add(rec[0])
            axis.add(rec[2])
        axis = sorted(list(axis))
        ans = 0
        n = len(axis)
        for i in range(n - 1):

            # 枚举两个点之间的宽度
            x1, x2 = axis[i], axis[i + 1]
            width = x2 - x1
            if not width:
                continue

            # 这里本质上是求一维区间的合并覆盖长度作为高度
            items = [[rec[1], rec[3]] for rec in rectangles if rec[0] < x2 and rec[2] > x1]
            items.sort(key=lambda x: [x[0], -x[1]])
            height = low = high = 0
            for y1, y2 in items:
                if y1 >= high:
                    height += high - low
                    low, high = y1, y2
                else:
                    high = high if high > y2 else y2
            height += high - low

            # 表示区[x1,x2]内的矩形覆盖高度为height
            ans += width * height
        return ans


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1884(ac=FastIO()):
        # 模板：计算矩形覆盖面积
        n = ac.read_int()
        lst = []
        for _ in range(n):
            lst.append(ac.read_list_ints())
        low_x = min(min(ls[0], ls[2]) for ls in lst)
        low_y = min(min(ls[1], ls[3]) for ls in lst)
        # 注意挪到坐标原点
        lst = [[ls[0] - low_x, ls[1] - low_y, ls[2] - low_x, ls[3] - low_y] for ls in lst]
        lst = [[ls[0], ls[3], ls[2], ls[1]] for ls in lst]
        ans = ScanLine().get_rec_area(lst)
        ac.st(ans)
        return


class TestGeneral(unittest.TestCase):
    def test_euler_phi(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
