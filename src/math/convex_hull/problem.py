"""
Algorithm：convex_hull|minimum_circle_coverage|random_increment_method
Description：convex_hull

====================================LeetCode====================================
1924（https://leetcode.cn/problems/erect-the-fence-ii/）convex_hull|tripart_pack_tripart|minimum_circle_coverage

=====================================LuoGu======================================
P1742（https://www.luogu.com.cn/problem/P1742）random_increment_method|minimum_circle_coverage
P3517（https://www.luogu.com.cn/problem/P3517）binary_search_of_binary_search|random_increment_method|minimum_circle_coverage

"""

from typing import List

from src.math.convex_hull.template import MinCircleOverlap
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1924(trees: List[List[int]]) -> List[float]:
        """
        url: https://leetcode.cn/problems/erect-the-fence-ii/
        tag: convex_hull|tripart_pack_tripart|minimum_circle_coverage
        """
        ans = MinCircleOverlap().get_min_circle_overlap(trees)
        return list(ans)

    @staticmethod
    def lg_p1742(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1742
        tag: random_increment_method|minimum_circle_coverage
        """
        n = ac.read_int()
        nums = [ac.read_list_floats() for _ in range(n)]
        x, y, r = MinCircleOverlap().get_min_circle_overlap(nums)
        ac.st(r)
        ac.lst([x, y])
        return

    @staticmethod
    def lg_3517(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3517
        tag: binary_search_of_binary_search|random_increment_method|minimum_circle_coverage
        """
        n, m = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]

        def check(r):

            def circle(lst):
                x, y, rr = MinCircleOverlap().get_min_circle_overlap(lst)
                return x, y, rr

            cnt = i = 0
            res = []
            while i < n:
                left = i
                right = n - 1
                while left < right - 1:
                    mm = left + (right - left) // 2
                    if circle(nums[i:mm + 1])[2] <= r:
                        left = mm
                    else:
                        right = mm
                ll = circle(nums[i:right + 1])
                if ll[2] > r:
                    ll = circle(nums[i:left + 1])
                    i = left + 1
                else:
                    i = right + 1
                res.append(ll[:-1])
                cnt += 1
            return res, cnt <= m

        low = 0
        high = 4 * 10 ** 6
        error = 10 ** (-6)
        while low < high - error:
            mid = low + (high - low) / 2
            if check(mid)[1]:
                high = mid
            else:
                low = mid

        nodes, flag = check(low)
        rrr = low
        if not flag:
            nodes, flag = check(high)
            rrr = high
        ac.st(rrr)
        ac.st(len(nodes))
        for a in nodes:
            ac.lst([round(a[0], 10), round(a[1], 10)])
        return
