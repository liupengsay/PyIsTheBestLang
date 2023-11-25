"""
算法：离线查询、预处理排序
功能：通过指针移动来进行查询值更新
参考：
题目：

===================================力扣===================================
100110. 找到 Alice 和 Bob 可以相遇的建筑（https://leetcode.cn/contest/weekly-contest-372/problems/find-building-where-alice-and-bob-can-meet/）offline query after sorting

===================================洛谷===================================
xx（xxx）xxxxxxxxxxxxxxxxxxxx

================================CodeForces================================

"""
from typing import List

from sortedcontainers import SortedList

from tests.leetcode.template import ac_max


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_100110(heights: List[int], queries: List[List[int]]) -> List[int]:

        m = len(queries)
        for i in range(m):
            x, y = queries[i]
            queries[i] = (i, x, y, ac_max(heights[x], heights[y]))
        queries.sort(key=lambda it: -it[-1])
        ans = [-1] * m

        n = len(heights)
        original = heights[:]
        heights = [(i, heights[i]) for i in range(n)]
        heights.sort(key=lambda it: -it[-1])
        j = 0
        lst = SortedList()
        for i, x, y, c in queries:
            if x < y and original[x] < original[y]:
                ans[i] = y
                continue
            if y < x and original[y] < original[x]:
                ans[i] = x
                continue
            if x == y:
                ans[i] = y
                continue
            while j < n and heights[j][1] > c:
                lst.add(heights[j][0])
                j += 1
            k = lst.bisect_right(ac_max(x, y))
            if 0 <= k < len(lst):
                ans[i] = lst[k]
        return ans
