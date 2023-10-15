import math
import random
import unittest
from collections import defaultdict, Counter
from functools import reduce
from math import gcd
from itertools import accumulate
from typing import List
from operator import mul, add, xor, and_, or_
from src.fast_io import FastIO
from math import inf


INF = int(1e64)


class TwoPointer:
    def __init__(self):
        return

    @staticmethod
    def window(nums):
        n = len(nums)
        ans = j = 0
        dct = dict()
        for i in range(n):
            while j < n and (nums[j] in dct or not dct or (abs(max(dct) - nums[j]) <= 2 and abs(min(dct) - nums[j]) <= 2)):
                dct[nums[j]] = dct.get(nums[j], 0) + 1
                j += 1
            ans += j - i
            dct[nums[i]] -= 1
            if not dct[nums[i]]:
                del dct[nums[i]]
        return ans

    @staticmethod
    def circle_array(arr):
        # 模板：环形数组指针移动
        n = len(arr)
        ans = 0
        for i in range(n):
            ans = max(ans, arr[i] + arr[(i + n - 1) % n])
        return ans

    @staticmethod
    def fast_and_slow(head):
        # 模板：快慢指针判断链表是否存在环
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False

    @staticmethod
    def same_direction(nums):
        # 模板: 相同方向双指针（寻找最长不含重复元素的子序列）
        n = len(nums)
        ans = j = 0
        pre = set()
        for i in range(n):
            # 特别注意指针的移动情况
            while j < n and nums[j] not in pre:
                pre.add(nums[j])
                j += 1
            # 视情况更新返回值
            ans = ans if ans > j - i else j - i
            pre.discard(nums[i])
        return ans

    @staticmethod
    def opposite_direction(nums, target):
        # 模板: 相反方向双指针（寻找升序数组是否存在两个数和为target）
        n = len(nums)
        i, j = 0, n - 1
        while i < j:
            cur = nums[i] + nums[j]
            if cur > target:
                j -= 1
            elif cur < target:
                i += 1
            else:
                return True
        return False


class SlidingWindowAggregation:
    """SlidingWindowAggregation

    Api:
    1. append value to tail,O(1).
    2. pop value from head,O(1).
    3. query aggregated value in window,O(1).
    """

    def __init__(self, e, op):
        # 模板：滑动窗口维护和查询聚合信息
        """
        Args:
            e: unit element
            op: merge function
        """
        self.stack0 = []
        self.agg0 = []
        self.stack2 = []
        self.stack3 = []
        self.e = e
        self.e0 = self.e
        self.e1 = self.e
        self.size = 0
        self.op = op

    def append(self, value) -> None:
        if not self.stack0:
            self.push0(value)
            self.transfer()
        else:
            self.push1(value)
        self.size += 1

    def popleft(self) -> None:
        if not self.size:
            return
        if not self.stack0:
            self.transfer()
        self.stack0.pop()
        self.stack2.pop()
        self.e0 = self.stack2[-1] if self.stack2 else self.e
        self.size -= 1

    def query(self):
        return self.op(self.e0, self.e1)

    def push0(self, value):
        self.stack0.append(value)
        self.e0 = self.op(value, self.e0)
        self.stack2.append(self.e0)

    def push1(self, value):
        self.agg0.append(value)
        self.e1 = self.op(self.e1, value)
        self.stack3.append(self.e1)

    def transfer(self):
        while self.agg0:
            self.push0(self.agg0.pop())
        while self.stack3:
            self.stack3.pop()
        self.e1 = self.e

    def __len__(self):
        return self.size




