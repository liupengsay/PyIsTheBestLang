"""
Algorithm：sort|bubble_sort|merge_sort(minimum_num_comparisons)|quick_sort(expected_fast)|custom_sort
Description：reverse_order_pair

====================================LeetCode====================================
164（https://leetcode.cn/problems/maximum-gap/）bucket_sort
179（https://leetcode.cn/problems/largest-number/）custom_sort|maximum
912（https://leetcode.cn/problems/sort-an-array/）quick_sort
1585（https://leetcode.cn/problems/check-if-string-is-transformable-with-substring-sort-operations/）bubble_sort|implemention
45（https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/）custom_sort|minimum
2412（https://leetcode.cn/problems/minimum-money-required-before-transactions/）custom_sort|greedy
1665（https://leetcode.cn/problems/minimum-initial-energy-to-finish-tasks/）custom_sort|greedy|sorting

=====================================LuoGu======================================
P2310（https://www.luogu.com.cn/problem/P2310）sorting
P4378（https://www.luogu.com.cn/problem/P4378）brute_force|bubble_sort
P6243（https://www.luogu.com.cn/problem/P6243）greedy|custom_sort
P1774（https://www.luogu.com.cn/problem/P1774）merge_sort|reverse_order_pair
P1177（https://www.luogu.com.cn/problem/P1177）quick_sort

===================================CodeForces===================================
922D（https://codeforces.com/problemset/problem/922/D）greedy|custom_sort

====================================AtCoder=====================================
ABC042B（https://atcoder.jp/contests/abc042/tasks/abc042_b）custom_sort

=====================================AcWing=====================================
113（https://www.acwing.com/problem/content/description/115/）custom_sort

"""
import random
from functools import cmp_to_key
from typing import List

from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_912(lst: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/sort-an-array/
        tag: quick_sort
        """
        # quick_sort两路手动实现
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

    @staticmethod
    def ac_113(compare, n):
        """
        url: https://www.acwing.com/problem/content/description/115/
        tag: custom_sort
        """

        def compare_(x, y):
            if compare(x, y):
                return -1
            return 1

        nums = list(range(1, n + 1))
        nums.sort(key=cmp_to_key(compare_))
        return nums

    @staticmethod
    def abc_42b(ac=FastIO()):
        # custom_sort
        n, m = ac.read_list_ints()
        nums = [ac.read_str() for _ in range(n)]

        def compare(a, b):
            # 比较函数
            if a + b < b + a:
                return -1
            elif a + b > b + a:
                return 1
            return 0

        nums.sort(key=cmp_to_key(compare))
        ac.st("".join(nums))
        return

    @staticmethod
    def lc_179(nums: List[int]) -> str:
        """
        url: https://leetcode.cn/problems/largest-number/
        tag: custom_sort|maximum
        """

        # 模板: custom_sort拼接最大数
        def compare(a, b):
            # 比较函数
            x = int(a + b)
            y = int(b + a)
            if x > y:
                return -1
            elif x < y:
                return 1
            return 0

        nums = [str(x) for x in nums]
        nums.sort(key=cmp_to_key(compare))
        return str(int("".join(nums)))

    @staticmethod
    def lg_1177(ac=FastIO()):
        # quick_sort迭代实现
        n = ac.read_int()
        nums = ac.read_list_ints()
        stack = [[0, n - 1]]
        while stack:
            left, right = stack.pop()
            mid = nums[random.randint(left, right)]
            i, j = left, right
            while i <= j:
                while nums[i] < mid:
                    i += 1
                while nums[j] > mid:
                    j -= 1
                if i <= j:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
                    j -= 1
            if left < j:
                stack.append([left, j])
            if i < right:
                stack.append([i, right])
        ac.lst(nums)
        return

    @staticmethod
    def lc_1665(tasks: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-initial-energy-to-finish-tasks/
        tag: custom_sort|greedy|sorting
        """

        # 模板: custom_sort

        def compare(aa, bb):
            # 比较函数
            a1, m1 = aa
            a2, m2 = bb
            s12 = m1 if m1 > a1 + m2 else a1 + m2
            s21 = m2 if m2 > a2 + m1 else a2 + m1
            if s12 < s21:
                return -1
            elif s12 > s21:
                return 1
            return 0

        tasks.sort(key=cmp_to_key(compare))
        ans = cur = 0
        for a, m in tasks:
            if cur < m:
                ans += m - cur
                cur = m
            cur -= a
        return ans

    @staticmethod
    def lc_2412(transactions: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-money-required-before-transactions/
        tag: custom_sort|greedy
        """

        def check(it):
            cos, cash = it[:]
            if cos > cash:
                return [-1, cash]
            return [1, -cos]

        transactions.sort(key=check)
        ans = cur = 0
        for a, b in transactions:
            if cur < a:
                ans += a - cur
                cur = a
            cur += b - a
        return ans
