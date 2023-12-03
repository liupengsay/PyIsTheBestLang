"""
算法：扫描线
功能：计算平面几何面积或者立体体积
题目：

===================================LeetCode===================================
218（https://leetcode.com/problems/the-skyline-problem/）扫描线计算建筑物的轮廓
850（https://leetcode.com/problems/rectangle-area-ii/）扫描线计算覆盖面积，线段树加离散化应该有 nlogn 的解法

===================================LuoGu==================================
6265（https://www.luogu.com.cn/problem/P6265）计算建筑物的扫描线轮廓
5490（https://www.luogu.com.cn/problem/P5490）扫描线计算覆盖面积
1884（https://www.luogu.com.cn/problem/P1884）扫描线计算覆盖面积
1904（https://www.luogu.com.cn/problem/P1904）扫描线计算建筑物的轮廓

参考：[OI WiKi]（https://oi-wiki.org/geometry/scanning/)
"""
from src.mathmatics.scan_line.template import ScanLine
from src.utils.fast_io import FastIO


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