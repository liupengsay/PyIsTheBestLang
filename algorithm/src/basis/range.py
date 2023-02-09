import unittest

"""
算法：区间合并、区间覆盖、区间计数
功能：涉及到区间的一些合并查询和操作，也可以使用差分数组与树状数组、线段树进行解决
题目：

===================================力扣===================================
435. 无重叠区间（https://leetcode.cn/problems/non-overlapping-intervals/）最多不相交的区间，使用贪心或者二分DP
763. 划分字母区间（https://leetcode.cn/problems/partition-labels/）经典将区间合并为不相交的区间

===================================洛谷===================================
P2082 区间覆盖（加强版）（https://www.luogu.com.cn/problem/P2082）经典区间合并确定覆盖范围
P2434 [SDOI2005]区间（https://www.luogu.com.cn/problem/P2434）经典区间合并为不相交的区间
P2684 搞清洁（https://www.luogu.com.cn/problem/P2684）最小区间覆盖，选取最少的区间来进行覆盖
P2970 [USACO09DEC]Selfish Grazing S（https://www.luogu.com.cn/problem/P2970）最多不相交的区间，使用贪心或者二分DP
P6123 [NEERC2016]Hard Refactoring（https://www.luogu.com.cn/problem/P6123）区间合并变形问题

================================CodeForces================================
https://codeforces.com/problemset/problem/827/A（区间合并为不相交的区间，再贪心赋值）


参考：OI WiKi（xx）
"""


class RangeCoverCount:
    def __init__(self):
        return

    @staticmethod
    def range_cover_less(t, nums):
        # 模板: 计算nums的最少区间数进行覆盖 [1,t]
        nums.sort(key=lambda x: [x[0], -x[1]])
        if nums[0][0] != 1:
            return -1
        # 起点
        ans = 1  # 可以换为最后选取的区间返回
        end = nums[0][1]
        cur = -1
        for a, b in nums[1:]:
            if end >= t:
                return ans
            # 可以作为下一个交集
            if end >= a - 1:
                cur = cur if cur > b else b
            else:
                if cur <= end:
                    return -1
                # 新增一个最远交集区间
                ans += 1
                end = cur
                cur = -1
                if end >= t:
                    return ans
                if end >= a - 1:
                    # 可以再交
                    cur = cur if cur > b else b
                else:
                    # 不行
                    return -1
        # 可以再交
        if cur >= t:
            ans += 1
            return ans
        return -1

    @staticmethod
    def range_merge(lst):
        # 模板: 合并线性区间为不相交的区间
        lst.sort(key=lambda it: it[0])
        ans = []
        x, y = lst[0]
        for a, b in lst[1:]:
            if a <= y:
                y = y if y > b else b
            else:
                ans.append([x, y])
                x, y = a, b
        ans.append([x, y])
        return ans


class TestGeneral(unittest.TestCase):

    def test_range_cover_count(self):
        rcc = RangeCoverCount()
        lst = [[1, 4], [2, 5], [3, 6], [8, 9]]
        assert rcc.range_merge(lst) == [[1, 6], [8, 9]]
        return


if __name__ == '__main__':
    unittest.main()
