import random
import sys


class FastIO:
    def __init__(self):
        return

    @staticmethod
    def read_int():
        return int(sys.stdin.readline().strip())

    @staticmethod
    def read_float():
        return float(sys.stdin.readline().strip())

    @staticmethod
    def read_ints():
        return map(int, sys.stdin.readline().strip().split())

    @staticmethod
    def read_floats():
        return map(float, sys.stdin.readline().strip().split())

    @staticmethod
    def read_ints_minus_one():
        return map(lambda x: int(x) - 1, sys.stdin.readline().strip().split())

    @staticmethod
    def read_list_ints():
        return list(map(int, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_list_floats():
        return list(map(float, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_list_ints_minus_one():
        return list(map(lambda x: int(x) - 1,
                        sys.stdin.readline().strip().split()))

    @staticmethod
    def read_str():
        return sys.stdin.readline().strip()

    @staticmethod
    def read_list_strs():
        return sys.stdin.readline().strip().split()

    @staticmethod
    def read_list_str():
        return list(sys.stdin.readline().strip())

    @staticmethod
    def st(x):
        return sys.stdout.write(str(x) + '\n')

    @staticmethod
    def lst(x):
        return sys.stdout.write(" ".join(str(w) for w in x) + '\n')

    @staticmethod
    def round_5(f):
        res = int(f)
        if f - res >= 0.5:
            res += 1
        return res

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    def ask(self, lst):
        # CF交互题输出询问并读取结果
        self.lst(lst)
        sys.stdout.flush()
        res = self.read_int()
        # 记得任何一个输出之后都要 sys.stdout.flush() 刷新
        return res

    def out_put(self, lst):
        # CF交互题输出最终答案
        self.lst(lst)
        sys.stdout.flush()
        return

    @staticmethod
    def accumulate(nums):
        n = len(nums)
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + nums[i]
        return pre

    @staticmethod
    def get_random_seed():
        # 随机种子避免哈希冲突
        return random.randint(0, 10**9 + 7)


class Solution:
    def __init__(self):
        return

    @staticmethod
    def ac_3780(ac=FastIO()):
        # 模板：经典单调栈线性贪心DP构造
        n = ac.read_int()
        nums = ac.read_list_ints()
        if n == 1:
            ac.lst(nums)
            return

        n = len(nums)

        # 后面更小的
        post = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] > nums[i]:
                post[stack.pop()] = i
            stack.append(i)

        # 前面更小的
        pre = [-1] * n
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] > nums[i]:
                pre[stack.pop()] = i
            stack.append(i)

        left = [0] * n
        left[0] = nums[0]
        for i in range(1, n):
            j = pre[i]
            if j != -1:
                left[i] += nums[i] * (i - j) + left[j]
            else:
                left[i] = nums[i]*(i+1)

        right = [0] * n
        right[-1] = nums[-1]
        for i in range(n - 2, -1, -1):
            j = post[i]
            if j != -1:
                right[i] += nums[i] * (j - i) + right[j]
            else:
                right[i] = (n-i)*nums[i]

        # 枚举先上升后下降的最高点
        dp = [left[i] + right[i] - nums[i] for i in range(n)]
        x = dp.index(max(dp))
        for i in range(x + 1, n):
            nums[i] = ac.min(nums[i - 1], nums[i])
        for i in range(x - 1, -1, -1):
            nums[i] = ac.min(nums[i + 1], nums[i])
        ac.lst(nums)
        return


Solution().main()
