import bisect
import unittest
from collections import defaultdict, deque
from math import inf
from typing import List

from src.data_structure.sorted_list import LocalSortedList
from utils.fast_io import FastIO




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
        if not lst:
            return -1
        # inter=True 默认为[1, 3] + [3, 4] = [1, 4]
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
            if (end >= a and inter) or (not inter and end >= a - 1):  # 如果是 [1, 2] + [3, 4] = [1, 4] 则需要改成 end >= a-1
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
    def minimum_interval_coverage(clips: List[List[int]], time: int, inter=True) -> int:
        # 模板：最少区间覆盖问题，从 clips 中选出最小的区间数覆盖 [0, time] 当前只能处理 inter=True
        assert inter
        assert time >= 0
        if not clips:
            return -1
        if time == 0:
            if min(x for x, _ in clips) > 0:
                return -1
            return 1

        if inter:
            # 当前只能处理 inter=True 即 [1, 3] + [3, 4] = [1, 4]
            post = [0]*time
            for a, b in clips:
                if a < time:
                    post[a] = post[a] if post[a] > b else b
            if not post[0]:
                return -1

            ans = right = pre_end = 0
            for i in range(time):
                right = right if right > post[i] else post[i]
                if i == right:
                    return -1
                if i == pre_end:  # 也可以输出具体方案，注意此时要求区间左右点相同即 [1, 3] + [3, 4] = [1, 4]
                    ans += 1
                    pre_end = right
        else:
            ans = -1
        return ans

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


