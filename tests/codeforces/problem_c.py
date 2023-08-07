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

class TreeArrayRangeQueryPointUpdateMax:
    # 模板：树状数组 单点增加 前缀区间查询 最大值
    def __init__(self, n):
        # 索引从 1 到 n
        self.t = [0] * (n + 1)

    @staticmethod
    def lowest_bit(i):
        return i & (-i)

    def query(self, i):
        mx = 0
        while i:
            mx = mx if mx > self.t[i] else self.t[i]
            i -= self.lowest_bit(i)
        return mx

    def update(self, i, mx):
        while i < len(self.t):
            self.t[i] = self.t[i] if self.t[i] > mx else mx
            i += self.lowest_bit(i)
        return

class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):
        n = ac.read_int()
        nums = ac.read_list_ints()
        ind = {num: i for i, num in enumerate(sorted(nums))}
        tree = TreeArrayRangeQueryPointUpdateMax(n)
        for num in nums:
            pre = tree.query(ind[num]+1)
            tree.update(ind[num]+1, pre+num)
        ac.st(tree.query(n))
        return


Solution().main()
