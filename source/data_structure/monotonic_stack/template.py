import heapq


class QuickMonotonicStack:
    def __init__(self):
        return

    @staticmethod
    def pipline_general(nums):
        """template of index as pre bound and post bound in monotonic stack"""
        n = len(nums)
        post = [n - 1] * n  # initial can be n or n-1 or -1 dependent on usage
        pre = [0] * n  # initial can be 0 or -1 dependent on usage
        stack = []
        for i in range(n):  # can be also range(n-1, -1, -1) dependent on usage
            while stack and nums[stack[-1]] < nums[i]:  # can be < or > or <=  or >=  dependent on usage
                post[stack.pop()] = i - 1  # can be i or i-1 dependent on usage
            if stack:  # which can be done only pre and post are no-repeat such as post bigger and pre not-bigger
                pre[i] = stack[-1] + 1  # can be stack[-1] or stack[-1]-1 dependent on usage
            stack.append(i)

        # strictly smaller at pre or post
        post_min = [n - 1] * n
        pre_min = [0] * n
        stack = []
        for i in range(n):
            while stack and nums[i] < nums[stack[-1]]:
                post_min[stack.pop()] = i - 1
            stack.append(i)
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and nums[i] < nums[stack[-1]]:
                pre_min[stack.pop()] = i + 1
            stack.append(i)

        # strictly bigger at pre or post
        post_max = [n - 1] * n
        pre_max = [0] * n
        stack = []
        for i in range(n):
            while stack and nums[i] > nums[stack[-1]]:
                post_max[stack.pop()] = i - 1
            stack.append(i)
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and nums[i] > nums[stack[-1]]:
                pre_max[stack.pop()] = i + 1
            stack.append(i)
        return

    @staticmethod
    def pipline_general_2(nums):
        """template of post second strictly larger or pre second strictly larger
        which can also be solved by offline queries with sorting and binary search
        """
        n = len(nums)
        # next strictly larger elements
        post = [-1] * n
        # next and next strictly larger elements
        post2 = [-1] * n
        stack1 = []
        stack2 = []
        for i in range(n):
            while stack2 and stack2[0][0] < nums[i]:
                post2[heapq.heappop(stack2)[1]] = i
            while stack1 and nums[stack1[-1]] < nums[i]:
                j = stack1.pop()
                post[j] = i
                heapq.heappush(stack2, [nums[j], j])
            stack1.append(i)

        # previous strictly larger elements
        pre = [-1] * n
        # previous and previous strictly larger elements
        pre2 = [-1] * n
        stack1 = []
        stack2 = []
        for i in range(n - 1, -1, -1):
            while stack2 and stack2[0][0] < nums[i]:
                pre2[heapq.heappop(stack2)[1]] = i
            while stack1 and nums[stack1[-1]] < nums[i]:
                j = stack1.pop()
                pre[j] = i
                heapq.heappush(stack2, [nums[j], j])
            stack1.append(i)
        return


class MonotonicStack:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)

        # 视情况可给不存在前序相关最值的值赋 i 或者 0
        self.pre_bigger = [-1] * self.n  # 上一个更大值
        self.pre_bigger_equal = [-1] * self.n  # 上一个大于等于值
        self.pre_smaller = [-1] * self.n  # 上一个更小值
        self.pre_smaller_equal = [-1] * self.n  # 上一个小于等于值

        # 视情况可给不存在前序相关最值的值赋 i 或者 n-1
        self.post_bigger = [-1] * self.n  # 下一个更大值
        self.post_bigger_equal = [-1] * self.n  # 下一个大于等于值
        self.post_smaller = [-1] * self.n  # 下一个更小值
        self.post_smaller_equal = [-1] * self.n  # 下一个小于等于值

        self.gen_result()
        return

    def gen_result(self):

        # 从前往后遍历
        stack = []
        for i in range(self.n):
            while stack and self.nums[i] >= self.nums[stack[-1]]:
                self.post_bigger_equal[stack.pop()] = i  # 有时也用 i-1 作为边界
            if stack:
                self.pre_bigger[i] = stack[-1]  # 有时也用 stack[-1]+1 做为边界
            stack.append(i)

        stack = []
        for i in range(self.n):
            while stack and self.nums[i] <= self.nums[stack[-1]]:
                self.post_smaller_equal[stack.pop()] = i  # 有时也用 i-1 作为边界
            if stack:
                self.pre_smaller[i] = stack[-1]  # 有时也用 stack[-1]+1 做为边界
            stack.append(i)

        # 从后往前遍历
        stack = []
        for i in range(self.n - 1, -1, -1):
            while stack and self.nums[i] >= self.nums[stack[-1]]:
                self.pre_bigger_equal[stack.pop()] = i  # 有时也用 i-1 作为边界
            if stack:
                self.post_bigger[i] = stack[-1]  # 有时也用 stack[-1]-1 做为边界
            stack.append(i)

        stack = []
        for i in range(self.n - 1, -1, -1):
            while stack and self.nums[i] <= self.nums[stack[-1]]:
                self.pre_smaller_equal[stack.pop()] = i  # 有时也用 i-1 作为边界
            if stack:
                self.post_smaller[i] = stack[-1]  # 有时也用 stack[-1]-1 做为边界
            stack.append(i)

        return


class Rectangle:
    def __init__(self):
        return

    @staticmethod
    def compute_area(pre):
        # 模板：使用单调栈根据高度计算最大矩形面积

        m = len(pre)
        left = [0] * m
        right = [m - 1] * m
        stack = []
        for i in range(m):
            while stack and pre[stack[-1]] > pre[i]:
                right[stack.pop()] = i - 1
            if stack:  # 这里可以同时求得数组前后的下一个大于等于值
                left[i] = stack[-1] + 1  # 这里将相同的值视为右边的更大且并不会影响计算
            stack.append(i)

        ans = 0
        for i in range(m):
            cur = pre[i] * (right[i] - left[i] + 1)
            ans = ans if ans > cur else cur
        return ans

    @staticmethod
    def compute_number(pre):
        # 模板：使用单调栈根据高度计算矩形个数

        n = len(pre)
        right = [n - 1] * n
        left = [0] * n
        stack = []
        for j in range(n):
            while stack and pre[stack[-1]] > pre[j]:
                right[stack.pop()] = j - 1
            if stack:  # 这个单调栈过程和上述求面积的一样
                left[j] = stack[-1] + 1
            stack.append(j)

        ans = 0
        for j in range(n):
            ans += (right[j] - j + 1) * (j - left[j] + 1) * pre[j]
        return ans
