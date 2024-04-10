import random
from collections import Counter
from functools import cmp_to_key


class VariousSort:
    def __init__(self):
        return

    @staticmethod
    def insertion_sort(nums):
        n = len(nums)
        for i in range(1, n):
            key = nums[i]
            j = i - 1
            while j >= 0 and nums[j] > key:
                nums[j + 1] = nums[j]
                j = j - 1
            nums[j + 1] = key
        return nums

    @staticmethod
    def counting_sort(nums):
        count = Counter(nums)
        keys = sorted(count.keys())
        rank = 0
        for key in keys:
            while count[key]:
                nums[rank] = key
                count[key] -= 1
                rank += 1
        return nums

    @staticmethod
    def quick_sort_two(lst):
        n = len(lst)

        def quick_sort(i, j):
            if i >= j:
                return

            # First find the smaller divide and conquer sort
            val = lst[random.randint(i, j)]
            left = i
            for k in range(i, j + 1):
                if lst[k] < val:
                    lst[k], lst[left] = lst[left], lst[k]
                    left += 1
            quick_sort(i, left - 1)

            # Then find the larger divide and conquer sort on the right
            for k in range(i, j + 1):
                if lst[k] == val:
                    lst[k], lst[left] = lst[left], lst[k]
                    left += 1
            quick_sort(left, j)
            return

        quick_sort(0, n - 1)
        return lst

    def range_merge_to_disjoint_sort(self, nums):

        if len(nums) > 1:
            mid = len(nums) // 2
            left = nums[:mid]
            right = nums[mid:]

            self.range_merge_to_disjoint_sort(left)
            self.range_merge_to_disjoint_sort(right)

            # Merge ordered lists using pointers
            i = j = k = 0
            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    nums[k] = left[i]
                    i += 1
                else:
                    nums[k] = right[j]
                    j += 1
                k += 1

            while i < len(left):
                nums[k] = left[i]
                i += 1
                k += 1

            while j < len(right):
                nums[k] = right[j]
                j += 1
                k += 1
        return nums

    @staticmethod
    def range_merge_to_disjoint_sort_inverse_pair(nums):
        """Use range_merge_to_disjoint sort to calculate the minimum number of times needed
        to make an array sorted by exchanging only adjacent elements
        which is equal the number of reverse_order_pair
        """

        ans = 0
        n = len(nums)
        arr = [0] * n
        stack = [(0, n - 1)]
        while stack:
            left, right = stack.pop()
            if left >= 0:
                if left >= right:
                    continue
                mid = (left + right) // 2
                stack.append((~left, right))
                stack.append((left, mid))
                stack.append((mid + 1, right))
            else:
                left = ~left
                mid = (left + right) // 2
                i, j = left, mid + 1
                k = left
                while i <= mid and j <= right:
                    if nums[i] <= nums[j]:
                        arr[k] = nums[i]
                        i += 1
                    else:
                        arr[k] = nums[j]
                        j += 1
                        ans += mid - i + 1
                    k += 1
                while i <= mid:
                    arr[k] = nums[i]
                    i += 1
                    k += 1
                while j <= right:
                    arr[k] = nums[j]
                    j += 1
                    k += 1

                for i in range(left, right + 1):
                    nums[i] = arr[i]
        return ans

    @staticmethod
    def heap_sort(nums):

        def sift_down(start, end):

            parent = int(start)
            child = int(parent * 2 + 1)

            while child <= end:

                if child + 1 <= end and nums[child] < nums[child + 1]:
                    child += 1

                if nums[parent] >= nums[child]:
                    return

                else:
                    nums[parent], nums[child] = nums[child], nums[parent]
                    parent = child
                    child = int(parent * 2 + 1)
            return

        length = len(nums)

        i = (length - 1 - 1) / 2
        while i >= 0:
            sift_down(i, length - 1)
            i -= 1

        i = length - 1
        while i > 0:
            nums[0], nums[i] = nums[i], nums[0]
            sift_down(0, i - 1)
            i -= 1
        return nums

    @staticmethod
    def shell_sort(nums):
        length = len(nums)
        h = 1
        while h < length / 3:
            h = int(3 * h + 1)
        while h >= 1:
            for i in range(h, length):
                j = i
                while j >= h and nums[j] < nums[j - h]:
                    nums[j], nums[j - h] = nums[j - h], nums[j]
                    j -= h
            h = int(h / 3)
        return nums

    @staticmethod
    def bucket_sort(nums):
        min_num = min(nums)
        max_num = max(nums)
        bucket_range = (max_num - min_num) / len(nums)
        count_list = [[] for _ in range(len(nums) + 1)]
        for i in nums:
            count_list[int((i - min_num) // bucket_range)].append(i)
        nums.clear()
        for i in count_list:
            for j in sorted(i):
                nums.append(j)
        return nums

    @staticmethod
    def bubble_sort(nums):
        n = len(nums)
        flag = True
        while flag:
            flag = False
            for i in range(n - 1):
                if nums[i] > nums[i + 1]:
                    flag = True
                    nums[i], nums[i + 1] = nums[i + 1], nums[i]
        return nums

    @staticmethod
    def selection_sort(nums):
        n = len(nums)
        for i in range(n):
            ith = i
            for j in range(i + 1, n):
                if nums[j] < nums[ith]:
                    ith = j
            nums[i], nums[ith] = nums[ith], nums[i]
        return nums

    @staticmethod
    def defined_sort(nums):

        def compare(a, b):
            if a < b:
                return -1
            elif a > b:
                return 1
            return 0

        def compare2(x, y):
            a = int(x + y)
            b = int(y + x)
            if a < b:
                return -1
            elif a > b:
                return 1
            return 0

        def compare3(x, y):
            a = x + y
            b = y + x
            if a < b:
                return -1
            elif a > b:
                return 1
            return 0

        nums.sort(key=cmp_to_key(compare3))
        nums.sort(key=cmp_to_key(compare2))
        nums.sort(key=cmp_to_key(compare))
        return nums

    @staticmethod
    def minimum_money(transactions) -> int:

        def check(ls):
            x, y = ls[0], ls[1]
            res = [0, 0]
            if x > y:
                res[0] = 0
                res[1] = y
            else:
                res[0] = 1
                res[1] = -x
            return res

        transactions.sort(key=lambda it: check(it))
        ans = cur = 0
        for a, b in transactions:
            if cur < a:
                ans += a - cur
                cur = a
            cur += b - a
        return ans
