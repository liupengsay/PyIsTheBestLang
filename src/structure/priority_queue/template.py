import heapq
from collections import defaultdict


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


class FindMedian:
    def __init__(self):
        self.small = []
        self.big = []
        self.big_dct = defaultdict(int)
        self.small_dct = defaultdict(int)
        self.big_cnt = 0
        self.small_cnt = 0

    def delete(self):
        while self.small and not self.small_dct[-self.small[0]]:
            heapq.heappop(self.small)
        while self.big and not self.big_dct[self.big[0]]:
            heapq.heappop(self.big)
        return

    def change(self):
        self.delete()
        while self.small and self.big and -self.small[0] > self.big[0]:
            self.small_dct[-self.small[0]] -= 1
            self.big_dct[-self.small[0]] += 1
            self.small_cnt -= 1
            self.big_cnt += 1
            heapq.heappush(self.big, -heapq.heappop(self.small))
            self.delete()
        return

    def balance(self):
        self.delete()
        while self.small_cnt > self.big_cnt:
            self.small_dct[-self.small[0]] -= 1
            self.big_dct[-self.small[0]] += 1
            heapq.heappush(self.big, -heapq.heappop(self.small))
            self.small_cnt -= 1
            self.big_cnt += 1
            self.delete()

        while self.small_cnt < self.big_cnt - 1:
            self.small_dct[self.big[0]] += 1
            self.big_dct[self.big[0]] -= 1
            heapq.heappush(self.small, -heapq.heappop(self.big))
            self.small_cnt += 1
            self.big_cnt -= 1
            self.delete()
        return


    def add(self, num):
        if not self.big or self.big[0] < num:
            self.big_dct[num] += 1
            heapq.heappush(self.big, num)
            self.big_cnt += 1
        else:
            self.small_dct[num] += 1
            heapq.heappush(self.small, -num)
            self.small_cnt += 1
        self.change()
        self.balance()
        return

    def remove(self, num):
        self.change()
        self.balance()
        if self.big_dct[num]:
            self.big_cnt -= 1
            self.big_dct[num] -= 1
        else:
            self.small_cnt -= 1
            self.small_dct[num] -= 1
        self.change()
        self.balance()
        return

    def find_median(self):
        self.change()
        self.balance()
        if self.big_cnt == self.small_cnt:
            return (-self.small[0] + self.big[0]) // 2
        return self.big[0]


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
