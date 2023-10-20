from collections import deque
from typing import List


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
