import bisect
import random
import unittest
from bisect import bisect_left
from collections import deque, defaultdict
from typing import List

from src.data_structure.segment_tree import RangeAscendRangeMax
from src.data_structure.tree_array import PointAscendPreMax
from utils.fast_io import FastIO


class LongestIncreasingSubsequence:
    def __init__(self):
        return

    @staticmethod
    def definitely_increase(nums):
        # 最长单调递增子序列（严格上升）
        dp = []
        for num in nums:
            i = bisect.bisect_left(dp, num)
            if 0 <= i < len(dp):
                dp[i] = num
            else:
                dp.append(num)
        return len(dp)

    @staticmethod
    def definitely_not_reduce(nums):
        # 最长单调不减子序列（不降）
        dp = []
        for num in nums:
            i = bisect.bisect_right(dp, num)
            if 0 <= i < len(dp):
                dp[i] = num
            else:
                dp.append(num)
        return len(dp)

    def definitely_reduce(self, nums):
        # 最长单调递减子序列（严格下降）
        nums = [-num for num in nums]
        return self.definitely_increase(nums)

    def definitely_not_increase(self, nums):
        # 最长单调不增子序列（不升）
        nums = [-num for num in nums]
        return self.definitely_not_reduce(nums)


class LcsLis:
    def __init__(self):
        return

    def longest_common_subsequence(self, s1, s2) -> int:
        # 使用LIS的办法求LCS
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        m = len(s2)
        mapper = defaultdict(list)
        for i in range(m - 1, -1, -1):
            mapper[s2[i]].append(i)
        nums = []
        for c in s1:
            if c in mapper:
                nums.extend(mapper[c])

        return self.longest_increasing_subsequence(nums)

    @staticmethod
    def longest_increasing_subsequence(nums: List[int]) -> int:
        # 使用贪心二分求LIS
        stack = []
        for x in nums:
            idx = bisect_left(stack, x)
            if idx < len(stack):
                stack[idx] = x
            else:
                stack.append(x)
        # 还可以返回stack获得最长公共子序列
        return len(stack)

    def longest_common_subsequence_stack(self, s1, s2) -> List[int]:
        # 使用LIS的办法求LCS
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        m = len(s2)
        mapper = defaultdict(list)
        for i in range(m - 1, -1, -1):
            mapper[s2[i]].append(i)
        nums = []
        for c in s1:
            if c in mapper:
                nums.extend(mapper[c])
        # 这里返回的是 s2 的索引不是具体的值
        res = self.longest_increasing_subsequence_stack(nums)
        return res

    @staticmethod
    def longest_increasing_subsequence_stack(nums: List[int]) -> List[int]:
        # 使用贪心二分求LIS
        if not nums:
            return []
        n = len(nums)
        tops = [nums[0]]
        piles = [0] * n
        piles[0] = 0

        for i in range(1, n):
            if nums[i] > tops[-1]:
                piles[i] = len(tops)
                tops.append(nums[i])
            else:
                j = bisect.bisect_left(tops, nums[i])
                piles[i] = j
                tops[j] = nums[i]

        lis = []
        j = len(tops) - 1
        for i in range(n - 1, -1, -1):
            if piles[i] == j:
                lis.append(nums[i])
                j -= 1
        lis.reverse()  # 反转列表，输出字典序最小的方案
        # 还可以返回stack获得最长公共子序列
        return lis

    @staticmethod
    def longest_increasing_subsequence_max_sum(nums):
        # 模板：经典最长上升且和最大的子序列的长度与和（同理可扩展到下降、不严格、最小和等）
        dp = []  # 维护 LIS 数组
        q = []  # 长度对应的末尾值与最大和
        for num in nums:
            if not dp or num > dp[-1]:
                dp.append(num)
                length = len(dp)
            else:
                i = bisect.bisect_left(dp, num)
                dp[i] = num
                length = i + 1
            while len(q) <= len(dp):
                q.append(deque())

            if length == 1:
                q[length].append([num, num])
            else:
                # 使用队列与计数加和维护
                while q[length - 1] and q[length - 1][0][0] >= num:
                    q[length - 1].popleft()
                cur = q[length - 1][0][1] + num
                while q[length] and q[length][-1][1] <= cur:
                    q[length].pop()
                q[length].append([num, cur])
        # 可以进一步变换求非严格递增子序列的个数
        return q[-1][0][1]

    @staticmethod
    def longest_increasing_subsequence_cnt(nums):
        # 模板：经典求 LIS 子序列的个数 O(nlogn) 做法模板题
        dp = []  # 维护 LIS 数组
        s = []  # 长度对应的方案和
        q = []  # 长度对应的末尾值与个数
        for num in nums:
            if not dp or num > dp[-1]:
                dp.append(num)
                length = len(dp)
            else:
                i = bisect.bisect_left(dp, num)
                dp[i] = num
                length = i + 1
            while len(s) <= len(dp):
                s.append(0)
            while len(q) <= len(dp):
                q.append(deque())

            if length == 1:
                s[length] += 1
                q[length].append([num, 1])
            else:
                # 使用队列与计数加和维护
                while q[length - 1] and q[length - 1][0][0] >= num:
                    s[length - 1] -= q[length - 1].popleft()[1]
                s[length] += s[length - 1]
                q[length].append([num, s[length - 1]])
        # 可以进一步变换求非严格递增子序列的个数
        return s[-1]

    @staticmethod
    def longest_common_subsequence_length_and_cnt(s1, s2, mod=10**9+7):
        # 模板：经典使用求 LIS 子序列的个数 O(nlogn) 做法求解 LCS 的长度与个数

        # 使用LIS的办法求LCS生成索引数组
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        m = len(s2)
        mapper = defaultdict(list)
        for i in range(m - 1, -1, -1):
            mapper[s2[i]].append(i)
        nums = []
        for c in s1:
            if c in mapper:
                nums.extend(mapper[c])

        dp = []  # 维护 LIS 数组
        s = []  # 长度对应的方案和
        q = []  # 长度对应的末尾值与个数
        for num in nums:
            if not dp or num > dp[-1]:
                dp.append(num)
                length = len(dp)
            else:
                i = bisect.bisect_left(dp, num)
                dp[i] = num
                length = i + 1
            while len(s) <= len(dp):
                s.append(0)
            while len(q) <= len(dp):
                q.append(deque())

            if length == 1:
                s[length] += 1
                q[length].append([num, 1])
            else:
                # 使用队列与计数加和维护
                while q[length - 1] and q[length - 1][0][0] >= num:
                    s[length - 1] -= q[length - 1].popleft()[1]
                s[length] += s[length - 1]
                s[length] %= mod
                q[length].append([num, s[length - 1]])
        return len(dp), s[-1]


