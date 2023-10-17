import heapq
from typing import List


class HeapqMedian:
    def __init__(self, mid):
        # 使用两个堆动态维护奇数长度数组的中位数
        self.mid = mid
        self.left = []
        self.right = []
        return

    def add(self, num):
        # 根据大小先放入左右两边的堆
        if num > self.mid:
            heapq.heappush(self.right, num)
        else:
            heapq.heappush(self.left, -num)
        n = len(self.left) + len(self.right)

        if n % 2 == 0:
            # 如果是奇数长度则更新中位数并保持左右两边数组长度相等
            if len(self.left) > len(self.right):
                heapq.heappush(self.right, self.mid)
                self.mid = -heapq.heappop(self.left)
            elif len(self.right) > len(self.left):
                heapq.heappush(self.left, -self.mid)
                self.mid = heapq.heappop(self.right)
        return

    def query(self):
        return self.mid


class KthLargest:
    # 使用一个堆维护第k大的元素
    def __init__(self, k: int, nums: List[int]):
        self.heap = [num for num in nums]
        self.k = k
        heapq.heapify(self.heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        while len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]


class MedianFinder:
    # 使用两个堆动态维护数组的中位数
    def __init__(self):
        self.pre = []  # 负数
        self.post = []  # 正数（中位数的位置）

    def add_num(self, num: int) -> None:
        if len(self.pre) != len(self.post):
            heapq.heappush(self.pre, -heapq.heappushpop(self.post, num))
        else:
            heapq.heappush(self.post, -heapq.heappushpop(self.pre, -num))

    def find_median(self) -> float:
        return self.post[0] if len(self.pre) != len(self.post) else (self.post[0]-self.pre[0])/2
