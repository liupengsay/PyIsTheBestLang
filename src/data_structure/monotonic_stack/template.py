
class QuickMonotonicStack:
    def __init__(self):
        return

    @staticmethod
    def pipline_general(nums):
        # 模板：经典单调栈前后边界下标计算
        n = len(nums)
        post = [n - 1] * n   # 这里可以是n/n-1/null，取决于用途
        pre = [0] * n   # 这里可以是0/-1/null，取决于用途
        stack = []
        for i in range(n):  # 这里也可以是从n-1到0倒序计算，取决于用途
            while stack and nums[stack[-1]] < nums[i]:  # 这里可以是"<" ">" "<=" ">="，取决于需要判断的大小关系
                post[stack.pop()] = i - 1  # 这里可以是i或者i-1，取决于是否包含i作为右端点
            if stack:  # 这里不一定可以同时计算，比如前后都是大于等于时，只有前后所求范围互斥时，可以计算
                pre[i] = stack[-1] + 1  # 这里可以是stack[-1]或者stack[-1]+1，取决于是否包含stack[-1]作为左端点
            stack.append(i)

        # 前后严格更小的边界
        post_min = [n - 1] * n  # 这里可以是n/n-1/null，取决于用途
        pre_min = [0] * n  # 这里可以是0/-1/null，取决于用途
        stack = []
        for i in range(n):  # 这里也可以是从n-1到0倒序计算，取决于用途
            while stack and nums[i] < nums[stack[-1]]:  # 这里可以是"<" ">" "<=" ">="，取决于需要判断的大小关系
                post_min[stack.pop()] = i - 1  # 这里可以是i或者i-1，取决于是否包含i作为右端点
            stack.append(i)
        stack = []
        for i in range(n - 1, -1, -1):  # 这里也可以是从n-1到0倒序计算，取决于用途
            while stack and nums[i] < nums[stack[-1]]:  # 这里可以是"<" ">" "<=" ">="，取决于需要判断的大小关系
                pre_min[stack.pop()] = i + 1  # 这里可以是i或者i-1，取决于是否包含i作为右端点
            stack.append(i)

        # 前后严格更大的边界
        post_max = [n - 1] * n  # 这里可以是n/n-1/null，取决于用途
        pre_max = [0] * n  # 这里可以是0/-1/null，取决于用途
        stack = []
        for i in range(n):  # 这里也可以是从n-1到0倒序计算，取决于用途
            while stack and nums[i] > nums[stack[-1]]:  # 这里可以是"<" ">" "<=" ">="，取决于需要判断的大小关系
                post_max[stack.pop()] = i - 1  # 这里可以是i或者i-1，取决于是否包含i作为右端点
            stack.append(i)
        stack = []
        for i in range(n - 1, -1, -1):  # 这里也可以是从n-1到0倒序计算，取决于用途
            while stack and nums[i] > nums[stack[-1]]:  # 这里可以是"<" ">" "<=" ">="，取决于需要判断的大小关系
                pre_max[stack.pop()] = i + 1  # 这里可以是i或者i-1，取决于是否包含i作为右端点
            stack.append(i)
        return

    @staticmethod
    def pipline_general_2(nums):
        # 模板：经典单调栈求下个与下下个严格更大元素与上个与上个个严格更大元素（可使用二分离线查询拓展到 k ）
        n = len(nums)
        post = [-1] * n
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

        pre = [-1] * n
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
        self.post_smaller_equal = [-1] * self.n   # 下一个小于等于值

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


