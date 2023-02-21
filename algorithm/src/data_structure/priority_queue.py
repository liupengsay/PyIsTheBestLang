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
P1750 出栈序列（https://www.luogu.com.cn/problem/P1750）经典题目，滑动指针窗口栈加队列
P2311 loidc，想想看（https://www.luogu.com.cn/problem/P2311）不定长滑动窗口最大值索引
P7175 [COCI2014-2015#4] PŠENICA（https://www.luogu.com.cn/problem/P7175）使用有序优先队列进行模拟
P7793 [COCI2014-2015#7] ACM（https://www.luogu.com.cn/problem/P7793）双端单调队列，进行最小值计算

参考：OI WiKi（xx）
"""


class PriorityQueue:
    def __init__(self):
        return

    @staticmethod
    def sliding_window(nums, k, method="max"):
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


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_239(self, nums: List[int], k: int) -> List[int]:
        # 模板：滑动窗口最大值
        return PriorityQueue().sliding_window(nums, k)

    @staticmethod
    def lg_p2251(ac=FastIO()):
        # 模板：滑动窗口最小值
        n, m = ac.read_ints()
        nums = ac.read_list_ints()
        ans = PriorityQueue().sliding_window(nums, m, "min")
        for a in ans:
            ac.st(a)
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
