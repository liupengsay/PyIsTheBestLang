
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

