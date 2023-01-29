
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
from functools import cmp_to_key

"""
算法：各种排序、冒泡排序、归并排序（期望比较次数最少）、快速排序（期望性能最好）
功能：xxx
题目：xx（xx）

L0045 把数组排成最小的数（https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/）自定义排序携程快排
P2310 loidc，看看海（https://www.luogu.com.cn/problem/P2310）预处理排序之后进行遍历
912. 排序数组（https://leetcode.cn/problems/sort-an-array/）快速排序
P4378 [USACO18OPEN]Out of Sorts S（https://www.luogu.com.cn/problem/P4378）枚举元素向左冒泡的移动轮数，计算最大轮数
P5626 【AFOI-19】数码排序（https://www.luogu.com.cn/problem/P5626）分治DP，归并排序需要的比较次数最少，但是可能内存占用超过快排


参考：OI WiKi（xx）
"""


class VariousSort:
    def __init__(self):
        return

    @staticmethod
    def insertion_sort(nums):
        # 插入排序
        n = len(nums)
        for i in range(1, n):
            key = nums[i]
            j = i - 1
            while j >= 0 and nums[j] > key:
                nums[j + 1] = nums[j]
                j = j - 1
            nums[j + 1] = key
        return nums

    @staticmethod
    def counting_sort(nums):
        # 计数排序
        count = Counter(nums)
        keys = sorted(count.keys())
        rank = 0
        for key in keys:
            while count[key]:
                nums[rank] = key
                count[key] -= 1
                rank += 1
        return nums

    # @staticmethod
    # def quick_sort(nums):
        # 快速排序
        # def recursion(first, last):
        #     if first >= last:
        #         return
        #     mid_value = nums[first]
        #     low = first
        #     high = last
        #     while low < high:
        #         while low < high and nums[high] >= mid_value:
        #             high -= 1
        #         nums[low] = nums[high]
        #         while low < high and nums[low] < mid_value:
        #             low += 1
        #         nums[high] = nums[low]
        #     nums[low] = mid_value
        #     recursion(first, low - 1)
        #     recursion(low + 1, last)
        #
        # recursion(0, len(nums) - 1)
        # return nums

    @staticmethod
    def sortArray(self, lst: List[int]) -> List[int]:
        # 两路快排
        n = len(lst)

        def quick_sort(i, j):
            if i >= j:
                return
            val = lst[random.randint(i, j)]
            left = i
            for k in range(i, j + 1):
                if lst[k] < val:
                    lst[k], lst[left] = lst[left], lst[k]
                    left += 1

            quick_sort(i, left - 1)
            for k in range(i, j + 1):
                if lst[k] == val:
                    lst[k], lst[left] = lst[left], lst[k]
                    left += 1
            quick_sort(left, j)
            return

        quick_sort(0, n - 1)
        return lst

    # @staticmethod
    # def quick_sort_three(nums):
    #     # 三路快排
    #     def recursion(first, last):
    #         if first >= last:
    #             return
    #         random_index = random.randint(first, last)
    #         pivot = nums[random_index]
    #         nums[first], nums[random_index] = nums[random_index], nums[first]
    #         i = first + 1
    #         j = first
    #         k = last + 1
    #         while i < k:
    #             if nums[i] < pivot:
    #                 nums[i], nums[j + 1] = nums[j + 1], nums[i]
    #                 j += 1
    #                 i += 1
    #             elif nums[i] > pivot:
    #                 nums[i], nums[k - 1] = nums[k - 1], nums[i]
    #                 k -= 1
    #             else:
    #                 i += 1
    #         nums[first], nums[j] = nums[j], nums[first]
    #         recursion(first, j - 1)
    #         recursion(k, last)
    #
    #     recursion(0, len(nums) - 1)
    #     return nums


    def merge_sort(self, nums):
        # 归并排序
        if len(nums) > 1:
            mid = len(nums) // 2
            left = nums[:mid]
            right = nums[mid:]

            self.merge_sort(left)
            self.merge_sort(right)

            i = j = k = 0
            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    nums[k] = left[i]
                    i += 1
                else:
                    nums[k] = right[j]
                    j += 1
                k += 1
            while i < len(left):
                nums[k] = left[i]
                i += 1
                k += 1

            while j < len(right):
                nums[k] = right[j]
                j += 1
                k += 1
        return nums

    @staticmethod
    def merge_sort_inverse_pair(nums, n):

        # 使用归并排序计算在只交换相邻元素的情况下至少需要多少次才能使数组变得有序
        # 也可以使用 2*n 长度的树状数组与线段树进行模拟计算
        # 结果等同于数组逆序对的数目
        # 参考题目【P1774 最接近神的人】

        def merge(left, right):
            nonlocal ans
            if left >= right:
                return

            # 递归进行排序
            mid = (left + right) // 2
            merge(left, mid)
            merge(mid + 1, right)

            # 合并有序列表
            i, j = left, mid + 1
            k = left
            while i <= mid and j <= right:
                if nums[i] <= nums[j]:
                    arr[k] = nums[i]
                    i += 1
                else:
                    arr[k] = nums[j]
                    j += 1
                    # 此时出现了逆序对移动记录次数
                    ans += mid - i + 1
                k += 1
            while i <= mid:
                arr[k] = nums[i]
                i += 1
                k += 1
            while j <= right:
                arr[k] = nums[j]
                j += 1
                k += 1

            # 将值赋到原数组
            for i in range(left, right + 1):
                nums[i] = arr[i]
            return

        # 使用归并排序进行求解
        ans = 0
        arr = [0] * n
        merge(0, n - 1)
        return ans


    @staticmethod
    def heap_sort(nums):
        # 堆排序
        def sift_down(start, end):
            # 计算父结点和子结点的下标
            parent = int(start)
            child = int(parent * 2 + 1)
            # 子结点下标在范围内才做比较
            while child <= end:
                # 先比较两个子结点大小，选择最大的
                if child + 1 <= end and nums[child] < nums[child + 1]:
                    child += 1
                # 如果父结点比子结点大，代表调整完毕，直接跳出函数
                if nums[parent] >= nums[child]:
                    return
                # 否则交换父子内容，子结点再和孙结点比较
                else:
                    nums[parent], nums[child] = nums[child], nums[parent]
                    parent = child
                    child = int(parent * 2 + 1)
            return

        length = len(nums)
        # 从最后一个节点的父节点开始 sift down 以完成堆化 (heapify)
        i = (length - 1 - 1) / 2
        while i >= 0:
            sift_down(i, length - 1)
            i -= 1
        # 先将第一个元素和已经排好的元素前一位做交换，再重新调整（刚调整的元素之前的元素），直到排序完毕
        i = length - 1
        while i > 0:
            nums[0], nums[i] = nums[i], nums[0]
            sift_down(0, i - 1)
            i -= 1
        return nums

    @staticmethod
    def shell_sort(nums):
        # 希尔排序
        length = len(nums)
        h = 1
        while h < length / 3:
            h = int(3 * h + 1)
        while h >= 1:
            for i in range(h, length):
                j = i
                while j >= h and nums[j] < nums[j - h]:
                    nums[j], nums[j - h] = nums[j - h], nums[j]
                    j -= h
            h = int(h / 3)
        return nums

    @staticmethod
    def bucket_sort(nums):
        # 桶排序
        min_num = min(nums)
        max_num = max(nums)
        # 桶的大小
        bucket_range = (max_num - min_num) / len(nums)
        # 桶数组
        count_list = [[] for i in range(len(nums) + 1)]
        # 向桶数组填数
        for i in nums:
            count_list[int((i - min_num) // bucket_range)].append(i)
        nums.clear()
        # 回填，这里桶内部排序直接调用了sorted
        for i in count_list:
            for j in sorted(i):
                nums.append(j)
        return nums

    @staticmethod
    def bubble_sort(nums):
        # 冒泡排序
        n = len(nums)
        flag = True
        while flag:
            flag = False
            for i in range(n - 1):
                if nums[i] > nums[i + 1]:
                    flag = True
                    nums[i], nums[i + 1] = nums[i + 1], nums[i]
        return nums

    def selection_sort(self, nums):
        n = len(nums)
        for i in range(n):
            ith = i
            for j in range(i + 1, n):
                if nums[j] < nums[ith]:
                    ith = j
            nums[i], nums[ith] = nums[ith], nums[i]
        return nums


    def largestNumber(self, nums: List[int]) -> str:
        def quick_sort(l, r):
            if l >= r:
                return
            i, j = l, r
            while i < j:
                while strs[j] + strs[l] <= strs[l] + strs[j] and i < j:
                    j -= 1
                while strs[i] + strs[l] >= strs[l] + strs[i] and i < j:
                    i += 1
                strs[i], strs[j] = strs[j], strs[i]
            strs[i], strs[l] = strs[l], strs[i]
            quick_sort(l, i - 1)
            quick_sort(i + 1, r)
            return

        strs = [str(num) for num in nums]
        quick_sort(0, len(strs) - 1)
        return str(int(''.join(strs)))


    def largestNumber2(self, nums: List[int]) -> str:
        # 自定义排序获得最大数字
        def compare(x, y):
            return int(y + x) - int(x + y)

        nums = [str(num) for num in nums]
        nums.sort(key=cmp_to_key(compare))
        # print(nums)
        return str(int(''.join(nums)))

    def minNumber(self, nums: List[int]) -> str:
        # 自定义排序拼接数字字符串使得最后的数字字典序最小
        def compare(x, y):
            return (x+y) < (y+x)
        nums = [str(num) for num in nums]
        nums.sort(key=cmp_to_key(compare))
        return "".join(nums)


class TestGeneral(unittest.TestCase):

    def test_xxx(self):
        vs = VariousSort()
        #assert nt.gen_result(10 ** 11 + 131) == 66666666752
        return


if __name__ == '__main__':
    unittest.main()
