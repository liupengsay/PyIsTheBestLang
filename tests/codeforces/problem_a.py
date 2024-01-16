from sys import stdin

inf = 1 << 32


class FastIO:
    def __init__(self):
        self.random_seed = 0
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

    def get_random_seed(self):
        import random
        self.random_seed = random.randint(0, 10 ** 9 + 7)
        return

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


class KMP:
    def __init__(self):
        return

    @staticmethod
    def prefix_function(s):
        """calculate the longest common true prefix and true suffix for s [:i+1] and s [:i+1]"""
        n = len(s)
        pi = [0] * n
        for i in range(1, n):
            j = pi[i - 1]
            while j > 0 and s[i] != s[j]:
                j = pi[j - 1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j  # pi[i]<=i
        # pi[0] = 0
        return pi

    @staticmethod
    def z_function(s):
        """calculate the longest common prefix between s[i:] and s"""
        n = len(s)
        z = [0] * n
        left, r = 0, 0
        for i in range(1, n):
            if i <= r and z[i - left] < r - i + 1:
                z[i] = z[i - left]
            else:
                z[i] = max(0, r - i + 1)
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
            if i + z[i] - 1 > r:
                left = i
                r = i + z[i] - 1
        # z[0] = 0
        return z

    def find(self, s1, s2):
        """find the index position of s2 in s1"""
        n, m = len(s1), len(s2)
        pi = self.prefix_function(s2 + "#" + s1)
        ans = []
        for i in range(m + 1, m + n + 1):
            if pi[i] == m:
                ans.append(i - m - m)
        return ans

    def find_lst(self, s1, s2, tag=-1):
        """find the index position of s2 in s1"""
        n, m = len(s1), len(s2)
        pi = self.prefix_function(s2 + [tag] + s1)
        ans = []
        for i in range(m + 1, m + n + 1):
            if pi[i] == m:
                ans.append(i - m - m)
        return ans

    def find_longest_palindrome(self, s, pos="prefix") -> int:
        """calculate the longest prefix and longest suffix palindrome substring"""
        if pos == "prefix":
            return self.prefix_function(s + "#" + s[::-1])[-1]
        return self.prefix_function(s[::-1] + "#" + s)[-1]


class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):
        """
        url: https://codeforces.com/contest/494/problem/B
        tag: kmp|linear_dp|prefix_sum
        """
        s = ac.read_str()
        t = ac.read_str()
        m, n = len(t), len(s)
        pi = KMP().prefix_function(t + "#" + s)
        mod = 10 ** 9 + 7
        dp = [0] * (n + 1)
        pre = [0] * (n + 1)
        dp[0] = pre[0] = 1
        last = -1
        for i in range(1, n + 1):
            if pi[i + m] == m:
                last = i - m + 1
            if last != -1:
                dp[i] = dp[i - 1] + pre[last - 1]
            else:
                dp[i] = dp[i - 1]
            dp[i] %= mod
            pre[i] = (pre[i - 1] + dp[i]) % mod
        ac.st((dp[-1] - 1) % mod)
        return


Solution().main()
