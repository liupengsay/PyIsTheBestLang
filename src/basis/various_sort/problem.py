"""
算法：排序、冒泡排序、归并排序（期望比较次数最少）、快速排序（期望性能最好）、自定义排序（灵活）
功能：各种排序的实现以及特点变形题目，如逆序对
题目：xx（xx）

===================================力扣===================================
164. 最大间距（https://leetcode.cn/problems/maximum-gap/）经典桶排序
179. 最大数（https://leetcode.cn/problems/largest-number/）自定义拼接最大数
912. 排序数组（https://leetcode.cn/problems/sort-an-array/）快速排序
1585. 检查字符串是否可以通过排序子字符串得到另一个字符串（https://leetcode.cn/problems/check-if-string-is-transformable-with-substring-sort-operations/）经典冒泡排序思想进行模拟
面试题45. 把数组排成最小的数（https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/）自定义拼接成最小的数
2412. 完成所有交易的初始最少钱数（https://leetcode.cn/problems/minimum-money-required-before-transactions/）自定义排序贪心选择顺序
1665. 完成所有任务的最少初始能量（https://leetcode.cn/problems/minimum-initial-energy-to-finish-tasks/）自定义排序确定贪心排序公式

===================================洛谷===================================
P2310 loidc，看看海（https://www.luogu.com.cn/problem/P2310）预处理排序之后进行遍历
P4378 [USACO18OPEN]Out of Sorts S（https://www.luogu.com.cn/problem/P4378）枚举元素向左冒泡的移动轮数，计算最大轮数
P5626 【AFOI-19】数码排序（https://www.luogu.com.cn/problem/P5626）分治DP，归并排序需要的比较次数最少，但是可能内存占用超过快排
P6243 [USACO06OPEN]The Milk Queue G（https://www.luogu.com.cn/problem/P6243）经典贪心举例之后进行自定义排序
P1774 最接近神的人（https://www.luogu.com.cn/problem/P1774）使用归并排序确定在只交换相邻元素的情况下最少的交换次数使得数组有序
P1177 【模板】快速排序（https://www.luogu.com.cn/problem/P1177）快速排序

================================CodeForces================================
https://codeforces.com/problemset/problem/922/D（贪心加自定义排序）

================================AtCoder================================
B - Iroha Loves Strings（https://atcoder.jp/contests/abc042/tasks/abc042_b）自定义排序

================================AcWing====================================
113. 特殊排序（https://www.acwing.com/problem/content/description/115/）自定义排序调用函数进行比较

参考：OI WiKi（xx）
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
        # 模板：快速排序两路手动实现
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
        # 模板：自定义排序
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

        # 模板: 自定义排序拼接最大数
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
        # 模板：快速排序迭代实现
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
        # 模板: 自定义排序

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
