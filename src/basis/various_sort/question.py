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
    def abc_42b(ac=FastIO()):
        # 模板：自定义排序
        n, m = ac.read_ints()
        nums = [ac.read_str() for _ in range(n)]

        def compare(a, b):
            # 比较函数
            if a+b < b+a:
                return -1
            elif a+b > b+a:
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
        stack = [[0, n-1]]
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
                ans += a-cur
                cur = a
            cur += b-a
        return ans

