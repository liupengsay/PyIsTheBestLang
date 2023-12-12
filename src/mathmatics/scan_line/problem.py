"""
Algorithm：scan_line
Description：plane|cube

====================================LeetCode====================================
218（https://leetcode.cn/problems/the-skyline-problem/）scan_line
850（https://leetcode.cn/problems/rectangle-area-ii/）scan_line|segment_tree|discretization|O(nlogn)

=====================================LuoGu======================================
P6265（https://www.luogu.com.cn/problem/P6265）scan_line
P5490（https://www.luogu.com.cn/problem/P5490）scan_line
P1884（https://www.luogu.com.cn/problem/P1884）scan_line
P1904（https://www.luogu.com.cn/problem/P1904）scan_line

"""
from src.mathmatics.scan_line.template import ScanLine
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1884(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1884
        tag: scan_line
        """
        # 矩形覆盖面积
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