import random
from collections import Counter
from functools import cmp_to_key
from typing import List


class VariousSort:
    def __init__(self):
        return

    @staticmethod
    def insertion_sort(nums):
        # 模板: 插入排序
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
        # 模板: 计数排序
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
    def quick_sort_two(lst: List[int]) -> List[int]:
        # 模板: 比较好理解和记忆的两路快排
        n = len(lst)

        def quick_sort(i, j):
            if i >= j:
                return

            # 先找到左边比较小的分治排序
            val = lst[random.randint(i, j)]
            left = i
            for k in range(i, j + 1):
                if lst[k] < val:
                    lst[k], lst[left] = lst[left], lst[k]
                    left += 1
            quick_sort(i, left - 1)

            # 再找到右边比较大的分治排序
            for k in range(i, j + 1):
                if lst[k] == val:
                    lst[k], lst[left] = lst[left], lst[k]
                    left += 1
            quick_sort(left, j)
            return

        quick_sort(0, n - 1)
        return lst

    def merge_sort(self, nums):
        # 模板: 归并排序
        if len(nums) > 1:
            mid = len(nums) // 2
            left = nums[:mid]
            right = nums[mid:]

            self.merge_sort(left)
            self.merge_sort(right)

            # 使用指针合并有序列表
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
    def merge_sort_inverse_pair(nums, n):

        # 模板: 使用归并排序计算在只交换相邻元素的情况下至少需要多少次才能使数组变得有序

        def merge(left, right):
            nonlocal ans
            if left >= right:
                return

            # 递归进行排序
            mid = (left + right) // 2
            merge(left, mid)
            merge(mid + 1, right)

            # 合并有序列表
            i, j = left, mid + 1
            k = left
            while i <= mid and j <= right:
                if nums[i] <= nums[j]:
                    arr[k] = nums[i]
                    i += 1
                else:
                    arr[k] = nums[j]
                    j += 1
                    # 此时出现了逆序对移动记录次数
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

            # 将值赋到原数组
            for i in range(left, right + 1):
                nums[i] = arr[i]
            return

        # 使用归并排序进行求解
        ans = 0
        arr = [0] * n
        merge(0, n - 1)
        # 也可以使用 2*n 长度的树状数组与线段树进行模拟计算
        # 结果等同于数组逆序对的数目
        # 参考题目P1774
        return ans

    @staticmethod
    def heap_sort(nums):
        # 模板: 堆排序
        def sift_down(start, end):
            # 计算父结点和子结点的下标
            parent = int(start)
            child = int(parent * 2 + 1)
            # 子结点下标在范围内才做比较
            while child <= end:
                # 先比较两个子结点大小，选择最大的
                if child + 1 <= end and nums[child] < nums[child + 1]:
                    child += 1
                # 如果父结点比子结点大，代表调整完毕，直接跳出函数
                if nums[parent] >= nums[child]:
                    return
                # 否则交换父子内容，子结点再和孙结点比较
                else:
                    nums[parent], nums[child] = nums[child], nums[parent]
                    parent = child
                    child = int(parent * 2 + 1)
            return

        length = len(nums)
        # 从最后一个节点的父节点开始 sift down 以完成堆化 (heapify)
        i = (length - 1 - 1) / 2
        while i >= 0:
            sift_down(i, length - 1)
            i -= 1
        # 先将第一个元素和已经排好的元素前一位做交换，再重新调整（刚调整的元素之前的元素），直到排序完毕
        i = length - 1
        while i > 0:
            nums[0], nums[i] = nums[i], nums[0]
            sift_down(0, i - 1)
            i -= 1
        return nums

    @staticmethod
    def shell_sort(nums):
        # 模板: 希尔排序
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
        # 模板: 桶排序
        min_num = min(nums)
        max_num = max(nums)
        # 桶的大小
        bucket_range = (max_num - min_num) / len(nums)
        # 桶数组
        count_list = [[] for _ in range(len(nums) + 1)]
        # 向桶数组填数
        for i in nums:
            count_list[int((i - min_num) // bucket_range)].append(i)
        nums.clear()
        # 回填，这里桶内部排序直接调用了sorted
        for i in count_list:
            for j in sorted(i):
                nums.append(j)
        return nums

    @staticmethod
    def bubble_sort(nums):
        # 模板: 冒泡排序
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
        # 模板: 选择排序
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

        # 模板: 自定义排序

        def compare(a, b):
            # 比较函数
            if a < b:
                return -1
            elif a > b:
                return 1
            return 0

        def compare2(x, y):
            # 比较函数
            a = int(x+y)
            b = int(y+x)
            if a < b:
                return -1
            elif a > b:
                return 1
            return 0

        def compare3(x, y):
            # 比较函数
            a = x+y
            b = y+x
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
    def ac_113(compare, n):

        # 模板：自定义排序

        def compare_(x, y):
            # 比较函数
            if compare(x, y):
                return -1
            return 1

        nums = list(range(1, n+1))
        nums.sort(key=cmp_to_key(compare_))
        return nums

    @staticmethod
    def minimum_money(transactions: List[List[int]]) -> int:
        # 模板：贪心选择顺序，自定义排序方式

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
                ans += a-cur
                cur = a
            cur += b-a
        return ans
