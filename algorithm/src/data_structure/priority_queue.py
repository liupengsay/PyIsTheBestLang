import unittest
from collections import deque
import random
from typing import List

from algorithm.src.fast_io import FastIO

"""
算法：单调队列、双端队列
功能：维护单调性，计算滑动窗口最大值最小值
题目：

===================================力扣===================================
239. 滑动窗口最大值（https://leetcode.cn/problems/sliding-window-maximum/）滑动区间最大值

===================================洛谷===================================
P2251 质量检测（https://www.luogu.com.cn/problem/P2251）滑动区间最小值
P2032 扫描（https://www.luogu.com.cn/problem/P2032）滑动区间最大值
P1750 出栈序列（https://www.luogu.com.cn/problem/P1750）经典题目，滑动指针窗口栈加队列
P2311 loidc，想想看（https://www.luogu.com.cn/problem/P2311）不定长滑动窗口最大值索引
P7175 [COCI2014-2015#4] PŠENICA（https://www.luogu.com.cn/problem/P7175）使用有序优先队列进行模拟
P7793 [COCI2014-2015#7] ACM（https://www.luogu.com.cn/problem/P7793）双端单调队列，进行最小值计算
P2216 [HAOI2007]理想的正方形（https://www.luogu.com.cn/problem/P2216）二维区间的滑动窗口最大最小值
P1886 滑动窗口 /【模板】单调队列（https://www.luogu.com.cn/problem/P1886）计算滑动窗口的最大值与最小值
P1714 切蛋糕（https://www.luogu.com.cn/problem/P1714）前缀和加滑动窗口最小值
P1725 琪露诺（https://www.luogu.com.cn/problem/P1725）单调队列和指针维护滑动窗口最大值加线性DP
P2827 [NOIP2016 提高组] 蚯蚓（https://www.luogu.com.cn/problem/P2827）经典单调队列

参考：OI WiKi（xx）
"""


class PriorityQueue:
    def __init__(self):
        return

    @staticmethod
    def sliding_window(nums: List[int], k: int, method="max") -> List[int]:
        assert k >= 1
        # 模板: 计算滑动窗口最大值与最小值
        if method == "min":
            nums = [-num for num in nums]
        n = len(nums)
        stack = deque()
        ans = []
        for i in range(n):
            while stack and stack[0][1] <= i - k:
                stack.popleft()
            while stack and stack[-1][0] <= nums[i]:
                stack.pop()
            stack.append([nums[i], i])
            if i >= k - 1:
                ans.append(stack[0][0])
        if method == "min":
            ans = [-num for num in ans]
        return ans

    @staticmethod
    def sliding_window_all(nums: List[int], k: int, method="max") -> List[int]:
        assert k >= 1
        # 模板: 计算滑动窗口最大值与最小值
        if method == "min":
            nums = [-num for num in nums]
        n = len(nums)
        stack = deque()
        ans = []
        for i in range(n):
            while stack and stack[0][1] <= i - k:
                stack.popleft()
            while stack and stack[-1][0] <= nums[i]:
                stack.pop()
            stack.append([nums[i], i])
            ans.append(stack[0][0])
        if method == "min":
            ans = [-num for num in ans]
        return ans


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1725(ac=FastIO()):

        # 模板：单调队列和指针维护滑动窗口最大值加线性DP
        inf = float("-inf")
        n, low, high = ac.read_ints()
        n += 1
        nums = ac.read_list_ints()
        dp = [-inf] * n
        dp[0] = nums[0]
        j = 0
        stack = deque()
        for i in range(1, n):
            while stack and stack[0][0] < i - high:
                stack.popleft()
            while j < n and j <= i - low:
                while stack and stack[-1][1] <= dp[j]:
                    stack.pop()
                stack.append([j, dp[j]])
                j += 1
            if stack:
                dp[i] = stack[0][1] + nums[i]
        ans = max(dp[x] for x in range(n) if x + high >= n)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1714(ac=FastIO()):

        # 模板：前缀和加滑动窗口最小值
        n, m = ac.read_ints()
        nums = ac.read_list_ints()
        ans = max(nums)
        pre = 0
        stack = deque([[-1, 0]])
        for i in range(n):
            pre += nums[i]
            while stack and stack[0][0] <= i-m-1:
                stack.popleft()
            while stack and stack[-1][1] >= pre:
                stack.pop()
            stack.append([i, pre])
            if stack:
                ans = ac.max(ans, pre-stack[0][1])
        ac.st(ans)
        return

    @staticmethod
    def lc_239(self, nums: List[int], k: int) -> List[int]:
        # 模板：滑动窗口最大值
        return PriorityQueue().sliding_window(nums, k)

    @staticmethod
    def lg_p2032(ac=FastIO()):
        # 模板：滑动窗口最大值
        n, k = ac.read_ints()
        nums = ac.read_list_ints()
        ans = PriorityQueue().sliding_window(nums, k)
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lg_p2251(ac=FastIO()):
        # 模板：滑动窗口最小值
        n, m = ac.read_ints()
        nums = ac.read_list_ints()
        ans = PriorityQueue().sliding_window(nums, m, "min")
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lg_p2216(ac=FastIO()):

        # 模板：二维滑动窗口最大值与滑动窗口最小值
        m, n, k = ac.read_ints()
        grid = [ac.read_list_ints() for _ in range(m)]

        ceil = [[0]*n for _ in range(m)]
        floor = [[0]*n for _ in range(m)]
        pq = PriorityQueue()
        for i in range(m):
            ceil[i] = pq.sliding_window_all(grid[i], k, "max")
            floor[i] = pq.sliding_window_all(grid[i], k, "min")
        for j in range(n):
            lst = pq.sliding_window_all([ceil[i][j] for i in range(m)], k, "max")
            for i in range(m):
                ceil[i][j] = lst[i]
            lst = pq.sliding_window_all([floor[i][j] for i in range(m)], k, "min")
            for i in range(m):
                floor[i][j] = lst[i]
        ans = ceil[k-1][k-1] - floor[k-1][k-1]
        for i in range(k-1, m):
            for j in range(k-1, n):
                ans = ac.min(ans, ceil[i][j]-floor[i][j])
        ac.st(ans)
        return

    @staticmethod
    def lg_p1886(ac=FastIO()):
        # 模板：计算滑动窗口最大最小值
        n, k = ac.read_ints()
        nums = ac.read_list_ints()
        ans1 = []
        ans2 = []
        ceil = deque()
        floor = deque()
        for i in range(n):
            while ceil and ceil[0] < i-k+1:
                ceil.popleft()
            while ceil and nums[ceil[-1]] <= nums[i]:
                ceil.pop()
            ceil.append(i)

            while floor and floor[0] < i-k+1:
                floor.popleft()
            while floor and nums[floor[-1]] >= nums[i]:
                floor.pop()
            floor.append(i)

            if i >= k-1:
                ans1.append(nums[floor[0]])
                ans2.append(nums[ceil[0]])
        ac.lst(ans1)
        ac.lst(ans2)
        return


class TestGeneral(unittest.TestCase):

    def test_priority_queue(self):
        pq = PriorityQueue()

        for _ in range(10):
            n = random.randint(100, 1000)
            nums = [random.randint(1, n) for _ in range(n)]
            k = random.randint(1, n)
            ans = pq.sliding_window(nums, k, "max")
            for i in range(n-k+1):
                assert ans[i] == max(nums[i:i+k])

            ans = pq.sliding_window(nums, k, "min")
            for i in range(n - k + 1):
                assert ans[i] == min(nums[i:i + k])
        return


if __name__ == '__main__':
    unittest.main()
