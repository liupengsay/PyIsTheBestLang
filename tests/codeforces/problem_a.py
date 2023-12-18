import random
from heapq import heappush, heappop
from sys import stdin


class FastIO:
    def __init__(self):
        self.random_seed = random.randint(0, 10 ** 9 + 7)
        return

    @staticmethod
    def read_int():
        return int(stdin.readline().rstrip())

    @staticmethod
    def read_float():
        return float(stdin.readline().rstrip())

    @staticmethod
    def read_list_ints():
        return list(map(int, stdin.readline().rstrip().split()))

    @staticmethod
    def read_list_ints_minus_one():
        return list(map(lambda x: int(x) - 1, stdin.readline().rstrip().split()))

    @staticmethod
    def read_str():
        return stdin.readline().rstrip()

    @staticmethod
    def read_list_strs():
        return stdin.readline().rstrip().split()

    @staticmethod
    def st(x):
        return print(x)

    @staticmethod
    def lst(x):
        return print(*x)

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    @staticmethod
    def ceil(a, b):
        return a // b + int(a % b != 0)

    @staticmethod
    def accumulate(nums):
        n = len(nums)
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + nums[i]
        return pre


class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):

        n, m, p = ac.read_list_ints()
        dp = [1] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] * 2 % p

        def check():
            while stack and not cnt[stack[0]]:
                heappop(stack)
            if ceil >= 3 or cnt[stack[0]] != 1:
                ac.st(0)
            else:
                even = freq[2]
                ac.st(dp[n - even * 2 - 1])
            return

        def add(a):
            nonlocal ceil
            heappush(stack, a)
            cnt[a] += 1
            if ceil < cnt[a]:
                ceil = cnt[a]
            freq[cnt[a]] += 1
            if cnt[a] > 1:
                freq[cnt[a] - 1] -= 1
            return

        def remove(a):
            nonlocal ceil
            freq[cnt[a]] -= 1
            if not freq[ceil]:
                ceil -= 1
            cnt[a] -= 1
            if cnt[a]:
                freq[cnt[a]] += 1
            return

        freq = [0] * (n + 1)
        cnt = [0] * (n + 1)
        ceil = 0
        nums = ac.read_list_ints()
        stack = []
        for num in nums:
            add(num)

        check()
        for _ in range(m):
            x, k = ac.read_list_ints()
            x -= 1
            remove(nums[x])
            nums[x] = k
            add(k)
            check()
        return


Solution().main()
