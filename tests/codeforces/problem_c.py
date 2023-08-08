import random
import sys
from math import inf


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
    def ac_3735(ac=FastIO()):
        # 模板：经典倒序状压DP与输出具体方案
        n, m = ac.read_ints()
        if m == n*(n-1)//2:
            ac.st(0)
            return
        group = [0] * n
        for i in range(n):
            group[i] |= (1 << i)
        for _ in range(m):
            i, j = ac.read_ints()
            i -= 1
            j -= 1
            group[i] |= (1 << j)
            group[j] |= (1 << i)

        dp = [inf] * (1 << n)
        pre = [[] for _ in range(1 << n)]
        for i in range(n):
            dp[group[i]] = 1
            pre[group[i]] = [i, -1]  # use, from

        for i in range(1 << n):
            if dp[i] == inf:
                continue

            for j in range(n):
                if i & (1 << j):
                    nex = i | group[j]
                    if dp[nex] > dp[i] + 1:
                        dp[nex] = dp[i] + 1
                        pre[nex] = [j, i]  # use, from

        s = (1 << n) - 1
        ans = []
        while s > 0:
            ans.append(pre[s][0] + 1)
            s = pre[s][1]
        ac.st(len(ans))
        ac.lst(ans)
        return


Solution().main()
