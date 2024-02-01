import heapq


class HeapqMedian:
    def __init__(self, mid):
        """median maintenance by heapq with odd length array"""
        self.mid = mid
        self.left = []
        self.right = []
        self.left_sum = 0
        self.right_sum = 0
        return

    def add(self, num):

        if num > self.mid:
            heapq.heappush(self.right, num)
            self.right_sum += num
        else:
            heapq.heappush(self.left, -num)
            self.left_sum += num
        n = len(self.left) + len(self.right)

        if n % 2 == 0:
            # maintain equal length
            if len(self.left) > len(self.right):
                self.right_sum += self.mid
                heapq.heappush(self.right, self.mid)
                self.mid = -heapq.heappop(self.left)
                self.left_sum -= self.mid
            elif len(self.right) > len(self.left):
                heapq.heappush(self.left, -self.mid)
                self.left_sum += self.mid
                self.mid = heapq.heappop(self.right)
                self.right_sum -= self.mid
        return

    def query(self):
        return self.mid


class KthLargest:
    def __init__(self, k: int, nums):
        self.heap = [num for num in nums]
        self.k = k
        heapq.heapify(self.heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        while len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]


class MedianFinder:
    def __init__(self):
        self.pre = []
        self.post = []

    def add_num(self, num: int) -> None:
        if len(self.pre) != len(self.post):
            heapq.heappush(self.pre, -heapq.heappushpop(self.post, num))
        else:
            heapq.heappush(self.post, -heapq.heappushpop(self.pre, -num))

    def find_median(self) -> float:
        return self.post[0] if len(self.pre) != len(self.post) else (self.post[0] - self.pre[0]) / 2
