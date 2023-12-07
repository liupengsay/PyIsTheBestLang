"""
Algorithm：sorting、冒泡sorting、归并sorting（期望比较次数最少）、快速sorting（期望性能最好）、自定义sorting（灵活）
Function：各种sorting的实现以及特点变形题目，如reverse_pair|

====================================LeetCode====================================
164（https://leetcode.com/problems/maximum-gap/）桶sorting
179（https://leetcode.com/problems/largest-number/）自定义拼接最大数
912（https://leetcode.com/problems/sort-an-array/）快速sorting
1585（https://leetcode.com/problems/check-if-string-is-transformable-with-substring-sort-operations/）冒泡sorting思想implemention
面试题45（https://leetcode.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/）自定义拼接成最小的数
2412（https://leetcode.com/problems/minimum-money-required-before-transactions/）自定义sortinggreedy选择顺序
1665（https://leetcode.com/problems/minimum-initial-energy-to-finish-tasks/）自定义sorting确定greedysorting公式

=====================================LuoGu======================================
2310（https://www.luogu.com.cn/problem/P2310）预处理sorting之后遍历
4378（https://www.luogu.com.cn/problem/P4378）brute_force元素向左冒泡的移动轮数，最大轮数
6243（https://www.luogu.com.cn/problem/P6243）greedy举例之后自定义sorting
1774（https://www.luogu.com.cn/problem/P1774）归并sorting确定在只交换相邻元素的情况下最少的交换次数使得数组有序
1177（https://www.luogu.com.cn/problem/P1177）快速sorting

===================================CodeForces===================================
922D（https://codeforces.com/problemset/problem/922/D）greedy|自定义sorting

====================================AtCoder=====================================
B - Iroha Loves Strings（https://atcoder.jp/contests/abc042/tasks/abc042_b）自定义sorting

=====================================AcWing=====================================
113（https://www.acwing.com/problem/content/description/115/）自定义sorting调用函数比较

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
        # 快速sorting两路手动实现
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

        def compare_(x, y):
            if compare(x, y):
                return -1
            return 1

        nums = list(range(1, n + 1))
        nums.sort(key=cmp_to_key(compare_))
        return nums

    @staticmethod
    def abc_42b(ac=FastIO()):
        # 自定义sorting
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

        # 模板: 自定义sorting拼接最大数
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
        # 快速sorting迭代实现
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
        # 模板: 自定义sorting

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