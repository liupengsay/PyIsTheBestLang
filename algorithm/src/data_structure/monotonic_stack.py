"""

"""
"""
算法：单调栈
功能：用来计算数组前后的更大值更小值信息
题目：

L0085 最大矩形（https://leetcode.cn/problems/maximal-rectangle/）枚举矩形下边界结合单调栈计算最大面积 
P1950 长方形（https://www.luogu.com.cn/problem/P1950）通过枚举下边界结合单调栈计算矩形个数
L2334 元素值大于变化阈值的子数组（https://leetcode.cn/problems/subarray-with-elements-greater-than-varying-threshold/）排序后枚举最小值左右两边的影响范围
L2281 巫师的总力量（https://leetcode.cn/problems/sum-of-total-strength-of-wizards/）枚举当前元素作为最小值的子数组和并使用前缀和的前缀和计算
L2262 字符串的总引力（https://leetcode.cn/problems/total-appeal-of-a-string/）计算下一个或者上一个不同字符的位置
P1901 发射站（https://www.luogu.com.cn/problem/P1901）由不相同的数组成的数组求其前后的更大值
P2866 [USACO06NOV]Bad Hair Day S（https://www.luogu.com.cn/problem/P2866）单调栈
P2947 [USACO09MAR]Look Up S（https://www.luogu.com.cn/problem/P2947）单调栈裸题
P4147 玉蟾宫（https://www.luogu.com.cn/problem/P4147）枚举矩形的下边界，使用单调栈计算最大矩形面积



参考：OI WiKi（xx）
"""




import bisect
import random
import re
import unittest
from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from operator import xor
from functools import lru_cache
import random
from itertools import permutations, combinations
import numpy as np
from decimal import Decimal
import heapq
import copy


class MonotonicStack:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)

        # 视情况可给不存在前序相关最值的值赋 i 或者 0
        self.pre_bigger = [-1] * self.n  # 上一个更大值
        self.pre_bigger_equal = [-1] * self.n  # 上一个大于等于值
        self.pre_smaller = [-1] * self.n  # 上一个更小值
        self.pre_smaller_equal = [-1] * self.n  # 上一个小于等于值

        # 视情况可给不存在前序相关最值的值赋 i 或者 n-1
        self.post_bigger = [-1] * self.n  # 下一个更大值
        self.post_bigger_equal = [-1] * self.n  # 下一个大于等于值
        self.post_smaller = [-1] * self.n  # 下一个更小值
        self.post_smaller_equal = [-1] * self.n   # 下一个小于等于值

        self.gen_result()
        return

    def gen_result(self):

        # 从前往后遍历
        stack = []
        for i in range(self.n):
            while stack and self.nums[i] >= self.nums[stack[-1]]:
                self.post_bigger_equal[stack.pop()] = i  # 有时也用 i-1 作为边界
            if stack:
                self.pre_bigger[i] = stack[-1]  # 有时也用 stack[-1]+1 做为边界
            stack.append(i)

        stack = []
        for i in range(self.n):
            while stack and self.nums[i] <= self.nums[stack[-1]]:
                self.post_smaller_equal[stack.pop()] = i  # 有时也用 i-1 作为边界
            if stack:
                self.pre_smaller[i] = stack[-1]  # 有时也用 stack[-1]+1 做为边界
            stack.append(i)

        # 从后往前遍历
        stack = []
        for i in range(self.n - 1, -1, -1):
            while stack and self.nums[i] >= self.nums[stack[-1]]:
                self.pre_bigger_equal[stack.pop()] = i  # 有时也用 i-1 作为边界
            if stack:
                self.post_bigger[i] = stack[-1]  # 有时也用 stack[-1]-1 做为边界
            stack.append(i)

        stack = []
        for i in range(self.n - 1, -1, -1):
            while stack and self.nums[i] <= self.nums[stack[-1]]:
                self.pre_smaller_equal[stack.pop()] = i  # 有时也用 i-1 作为边界
            if stack:
                self.post_smaller[i] = stack[-1]  # 有时也用 stack[-1]-1 做为边界
            stack.append(i)

        return


class TestGeneral(unittest.TestCase):

    def test_monotonic_stack(self):
        n = 1000
        nums = [random.randint(0, n) for _ in range(n)]
        ms = MonotonicStack(nums)
        for i in range(n):

            # 上一个最值
            pre_bigger = pre_bigger_equal = pre_smaller = pre_smaller_equal = -1
            for j in range(i - 1, -1, -1):
                if nums[j] > nums[i]:
                    pre_bigger = j
                    break
            for j in range(i - 1, -1, -1):
                if nums[j] >= nums[i]:
                    pre_bigger_equal = j
                    break
            for j in range(i - 1, -1, -1):
                if nums[j] < nums[i]:
                    pre_smaller = j
                    break
            for j in range(i - 1, -1, -1):
                if nums[j] <= nums[i]:
                    pre_smaller_equal = j
                    break
            assert pre_bigger == ms.pre_bigger[i]
            assert pre_bigger_equal == ms.pre_bigger_equal[i]
            assert pre_smaller == ms.pre_smaller[i]
            assert pre_smaller_equal == ms.pre_smaller_equal[i]

            # 下一个最值
            post_bigger = post_bigger_equal = post_smaller = post_smaller_equal = - 1
            for j in range(i + 1, n):
                if nums[j] > nums[i]:
                    post_bigger = j
                    break
            for j in range(i + 1, n):
                if nums[j] >= nums[i]:
                    post_bigger_equal = j
                    break
            for j in range(i + 1, n):
                if nums[j] < nums[i]:
                    post_smaller = j
                    break
            for j in range(i + 1, n):
                if nums[j] <= nums[i]:
                    post_smaller_equal = j
                    break
            assert post_bigger == ms.post_bigger[i]
            assert post_bigger_equal == ms.post_bigger_equal[i]
            assert post_smaller == ms.post_smaller[i]
            assert post_smaller_equal == ms.post_smaller_equal[i]

        return


if __name__ == '__main__':
    unittest.main()
