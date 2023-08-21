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


class SegmentTreeRangeUpdateXORSum:
    def __init__(self, n):
        # 模板：区间值01翻转与区间和查询
        self.n = n
        self.cover = [0] * (4 * self.n)  # 区间和
        self.lazy = [0] * (4 * self.n)  # 懒标记
        return

    def build(self, nums) -> None:
        # 使用数组初始化线段树
        stack = [[0, self.n - 1, 1]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = nums[s]
                    continue
                stack.append([s, t, ~i])
                m = s + (t - s) // 2
                stack.append([s, m, 2 * i])
                stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self.cover[i] = self.cover[2*i+1]+self.cover[2*i]
        return

    def push_down(self, i: int, s: int, m: int, t: int) -> None:
        if self.lazy[i]:
            self.cover[2 * i] = m - s + 1 - self.cover[2 * i]
            self.cover[2 * i + 1] = t - m - self.cover[2 * i + 1]

            self.lazy[2 * i] ^= self.lazy[i]  # 注意使用异或抵消查询
            self.lazy[2 * i + 1] ^= self.lazy[i]  # 注意使用异或抵消查询

            self.lazy[i] = 0
        return

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 增减区间值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.cover[i] = t - s + 1 - self.cover[i]
                    self.lazy[i] ^= val  # 注意使用异或抵消查询
                    continue

                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, ~i])

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        return

    def query_sum(self, left: int, right: int, s: int, t: int, i: int) -> int:
        # 查询区间的和
        stack = [[s, t, i]]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i]
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return ans


class SegBitSet:
    # 使用位运算进行区间01翻转操作
    def __init__(self):
        self.val = 0
        return

    def update(self, b, c):
        # 索引从0开始
        p = (1 << (c + 1)) - (1 << b)
        self.val ^= p
        return

    def query(self, b, c):
        p = (1 << (c + 1)) - (1 << b)
        return bin(self.val & p).count("1")


class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):
        n = ac.read_int()
        nums = ac.read_list_ints()
        tree = [SegmentTreeRangeUpdateXORSum(n) for _ in range(22)]
        for j in range(22):
            lst = [1 if nums[i] & (1<<j) else 0 for i in range(n)]
            tree[j].build(lst)
        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                ll, rr = lst[1:]
                ll -= 1
                rr -= 1
                ans = sum((1<<j)*tree[j].query_sum(ll, rr, 0, n-1, 1) for j in range(22))
                ac.st(ans)
            else:
                ll, rr, xx = lst[1:]
                ll -= 1
                rr -= 1
                for j in range(22):
                    if (1<<j) & xx:
                        tree[j].update(ll, rr, 0, n-1, 1, 1)

        return


    @staticmethod
    def main2(ac=FastIO()):
        n = ac.read_int()
        nums = ac.read_list_ints()
        tree = [SegBitSet() for _ in range(22)]
        for i in range(n):
            x = nums[i]
            for j in range(22):
                if x & (1<<j):
                    tree[j].update(i, i)

        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                ll, rr = lst[1:]
                ll -= 1
                rr -= 1
                ans = sum((1<<j)*tree[j].query(ll, rr) for j in range(22))
                ac.st(ans)
            else:
                ll, rr, xx = lst[1:]
                ll -= 1
                rr -= 1
                for j in range(22):
                    if (1<<j) & xx:
                        tree[j].update(ll, rr)

        return

# 5037. 区间异或
# Solution().main()
Solution().main2()
