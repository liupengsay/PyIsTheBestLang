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
        return list(map(lambda x: int(x) - 1, sys.stdin.readline().strip().split()))

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
        return random.randint(0, 10**9+7)


def check(n, nums):

    if sorted(nums) == nums:
        return []
    #print(nums)
    ans = []
    if max(nums) > 0:
        x = max(nums)
        i = nums.index(x)
        while nums[i] <= 40:
            ans.append([i + 1, i + 1])
            nums[i] *= 2
        if i != n-1:
            while nums[-1] <= nums[i]:
                nums[-1] += nums[i]
                ans.append([n, i + 1])
        ans.append([n, n])
        nums[-1] *= 2
        for j in range(n - 1):
            nums[j] += nums[-1]
            ans.append([j + 1, n])
            ans.append([n, n])
            nums[-1] *= 2

    else:
        x = min(nums)
        i = nums.index(x)
        while nums[i] >= -40:
            ans.append([i + 1, i + 1])
            nums[i] *= 2
        if 0 != i:
            while nums[0] >= nums[i]:
                nums[0] += nums[i]
                ans.append([1, i + 1])

        ans.append([1, 1])
        nums[0] *= 2

        for j in range(n - 1, 0, -1):
            nums[j] += nums[0]
            nums[0] *= 2
            ans.append([j + 1, 1])
            ans.append([1, 1])
    #print(nums)
    assert sorted(nums) == nums
    return ans


class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):

        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()

            ans = check(n, nums)
            ac.st(len(ans))
            for ls in ans:
                ac.lst(ls)
        return


Solution().main()

# for _ in range(20):
#     nums = [random.randint(-20, 20) for _ in range(5)]
#     check(5, nums)
#

# nums = [7, -4, -2, -16, 1]
# check(5, nums)