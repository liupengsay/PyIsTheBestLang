import unittest
from collections import defaultdict
from typing import List

from algorithm.src.fast_io import FastIO

"""
算法：区间合并、区间覆盖、区间计数
功能：涉及到区间的一些合并查询和操作，也可以使用差分数组与树状数组、线段树进行解决
题目：

===================================力扣===================================
45. 跳跃游戏 II（https://leetcode.cn/problems/jump-game-ii/）转换为最小区间覆盖问题
1326. 灌溉花园的最少水龙头数目（https://leetcode.cn/problems/minimum-number-of-taps-to-open-to-water-a-garden/）转换为最小区间覆盖问题
1024. 视频拼接（https://leetcode.cn/problems/video-stitching/）转换为最小区间覆盖问题

435. 无重叠区间（https://leetcode.cn/problems/non-overlapping-intervals/）最多不相交的区间，使用贪心或者二分DP
763. 划分字母区间（https://leetcode.cn/problems/partition-labels/）经典将区间合并为不相交的区间
6313. 统计将重叠区间合并成组的方案数（https://leetcode.cn/contest/biweekly-contest-99/problems/count-ways-to-group-overlapping-ranges/）经典将区间合并为不相交的区间，再使用快速幂计数
2345. 寻找可见山的数量（https://leetcode.cn/problems/finding-the-number-of-visible-mountains/）二维偏序，转换为区间包含问题
757. 设置交集大小至少为2（https://leetcode.cn/problems/set-intersection-size-at-least-two/）贪心选取最少的点集合，使得每个区间包含其中至少两个点
2589. 完成所有任务的最少时间（https://leetcode.cn/problems/minimum-time-to-complete-all-tasks/）贪心选取最少的点集合，使得每个区间包含其中要求的k个点
LCP 32. 批量处理任务（https://leetcode.cn/problems/t3fKg1/）贪心选取最少的点集合，使得每个区间包含其中要求的k个点

===================================洛谷===================================
P2082 区间覆盖（加强版）（https://www.luogu.com.cn/problem/P2082）经典区间合并确定覆盖范围
P2434 [SDOI2005]区间（https://www.luogu.com.cn/problem/P2434）经典区间合并为不相交的区间
P2970 [USACO09DEC]Selfish Grazing S（https://www.luogu.com.cn/problem/P2970）最多不相交的区间，使用贪心或者二分DP
P6123 [NEERC2016]Hard Refactoring（https://www.luogu.com.cn/problem/P6123）区间合并变形问题
P2684 搞清洁（https://www.luogu.com.cn/problem/P2684）最小区间覆盖，选取最少的区间来进行覆盖
P1233 木棍加工（https://www.luogu.com.cn/problem/P1233）按照一个维度排序后计算另一个维度的最长严格递增子序列的长度，二位偏序，转换为区间包含问题
================================CodeForces================================
A. String Reconstruction（https://codeforces.com/problemset/problem/827/A）区间合并为不相交的区间，再贪心赋值
D. Nested Segments（https://codeforces.com/problemset/problem/652/D）二位偏序，转换为区间包含问题
D. Non-zero Segments（https://codeforces.com/problemset/problem/1426/D）贪心选取最少的点集合，使得每个区间包含其中至少一个点


参考：OI WiKi（xx）
"""


class Range:
    def __init__(self):
        return

    @staticmethod
    def merge(lst):
        # 模板: 区间合并为不相交的区间
        lst.sort(key=lambda it: it[0])
        ans = []
        x, y = lst[0]
        for a, b in lst[1:]:
            # 注意此处开闭区间 [1, 3] + [3, 4] = [1, 4]
            if a <= y:  # 如果是 [1, 2] + [3, 4] = [1, 4] 则需要改成 a <= y-1
                y = y if y > b else b
            else:
                ans.append([x, y])
                x, y = a, b
        ans.append([x, y])
        return ans

    @staticmethod
    def cover_less(s, t, lst, inter=True):
        # 模板: 计算nums的最少区间数进行覆盖 [s, t] 即最少区间覆盖
        lst.sort(key=lambda x: [x[0], -x[1]])
        if lst[0][0] != s:
            return -1
        if lst[0][1] >= t:
            return 1
        # 起点
        ans = 1  # 可以换为最后选取的区间返回
        end = lst[0][1]
        cur = -1
        for a, b in lst[1:]:
            # 注意此处开闭区间 [1, 3] + [3, 4] = [1, 4]
            if end >= t:
                return ans
            # 可以作为下一个交集
            if (end >= a and inter) or (not inter and end >= a - 1): # 如果是 [1, 2] + [3, 4] = [1, 4] 则需要改成 end >= a-1
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
                # 如果是 [1, 2] + [3, 4] = [1, 4] 则需要改成 end >= a-1
                if (end >= a and inter) or (not inter and end >= a - 1):
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
    def disjoint_most(lst):
        # 模板：最多不相交的区间
        lst.sort(key=lambda x: x[1])
        n = len(lst)
        ans = 0
        end = float("-inf")
        for a, b in lst:
            if a >= end:
                ans += 1
                end = b
        return ans


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_45(nums):
        n = len(nums)
        lst = [[i, min(n - 1, i + nums[i])] for i in range(n)]
        if n == 1:  # 注意特判
            return 0
        return Range().cover_less(0, n - 1, lst)

    @staticmethod
    def lc_1326(n, ranges):
        m = n + 1
        lst = []
        for i in range(m):
            lst.append([max(i - ranges[i], 0), i + ranges[i]])
        return Range().cover_less(0, n, lst)

    @staticmethod
    def lc_1024(clips, time) -> int:
        return Range().cover_less(0, time, clips)

    @staticmethod
    def lg_p2684(ac=FastIO()):
        n, t = ac.read_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        ac.st(Range().cover_less(1, t, nums))
        return

    @staticmethod
    def lc_435(intervals):
        # 模板：合并区间
        n = len(intervals)
        return n - Range().disjoint_most(intervals)

    @staticmethod
    def lc_763(s: str) -> List[int]:
        # 模板：合并区间
        dct = defaultdict(list)
        for i, w in enumerate(s):
            if len(dct[w]) >= 2:
                dct[w].pop()
            dct[w].append(i)
        lst = []
        for w in dct:
            lst.append([dct[w][0], dct[w][-1]])
        ans = Range().merge(lst)
        return [y-x+1 for x, y in ans]

    @staticmethod
    def lc_6313(ranges: List[List[int]]) -> int:
        # 模板：合并为不相交的区间
        cnt = len(Range().merge(ranges))
        mod = 10 ** 9 + 7
        return pow(2, cnt, mod)

    @staticmethod
    def cf_1426d(ac=FastIO()):
        # 模板：选取最少的点集合，使得每个区间包含其中至少一个点
        n = ac.read_int()
        nums = ac.read_list_ints()
        pre = 0
        dct = dict()
        dct[pre] = -1
        lst = []
        for i in range(n):
            pre += nums[i]
            if pre in dct:
                lst.append([dct[pre], i])
            dct[pre] = i
        if not lst:
            ac.st(0)
            return
        lst.sort(key=lambda it: [it[1], it[0]])
        ans = 1
        a, b = lst[0]
        for c, d in lst[1:]:
            if b > c + 1:
                continue
            else:
                ans += 1
                b = d
        ac.st(ans)
        return


class TestGeneral(unittest.TestCase):

    def test_range_cover_count(self):
        rcc = Range()
        lst = [[1, 4], [2, 5], [3, 6], [8, 9]]
        assert rcc.merge(lst) == [[1, 6], [8, 9]]
        return


if __name__ == '__main__':
    unittest.main()
