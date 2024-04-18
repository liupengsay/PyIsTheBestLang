from functools import lru_cache
from sys import stdin


class FastIO:
    def __init__(self):
        self.random_seed = 0
        self.flush = False
        self.inf = 1 << 32
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

    def st(self, x):
        return print(x, flush=self.flush)

    def lst(self, x):
        return print(*x, flush=self.flush)

    def flatten(self, lst):
        self.st("\n".join(str(x) for x in lst))
        return

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


class BinarySearch:
    def __init__(self):
        return

    @staticmethod
    def find_int_left(low: int, high: int, check) -> int:
        """find the minimum int x which make check true"""
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                high = mid
            else:
                low = mid
        return low if check(low) else high

    @staticmethod
    def find_int_left_strictly(low: int, high: int, check) -> int:
        """find the minimum int x which make check true"""
        while low < high:
            mid = low + (high - low) // 2
            if check(mid):
                high = mid
            else:
                low = mid + 1
        return low

    @staticmethod
    def find_int_right(low: int, high: int, check) -> int:
        """find the maximum int x which make check true"""
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                low = mid
            else:
                high = mid
        return high if check(high) else low

    @staticmethod
    def find_int_right_strictly(low: int, high: int, check) -> int:
        """find the maximum int x which make check true"""
        while low < high:
            mid = low + (high - low + 1) // 2
            if check(mid):
                low = mid
            else:
                high = mid - 1
        return high

    @staticmethod
    def find_float_left(low: float, high: float, check, error=1e-6) -> float:
        """find the minimum float x which make check true"""
        while low < high - error:
            mid = low + (high - low) / 2
            if check(mid):
                high = mid
            else:
                low = mid
        return low if check(low) else high

    @staticmethod
    def find_float_right(low: float, high: float, check, error=1e-6) -> float:
        """find the maximum float x which make check true"""
        while low < high - error:
            mid = low + (high - low) / 2
            if check(mid):
                low = mid
            else:
                high = mid
        return high if check(high) else low


class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):
        """
        url: url of the problem
        tag: algorithm tag
        """

        @lru_cache(None)
        def dfs(x):
            if x == 1:
                return 1
            return sum(dfs(y) for y in range(1, int(x ** 0.5) + 1))

        pre = 0
        cnt = 0
        for x in range(1, 1000):
            def find(ans):
                if ans <= 3:
                    cc = ans
                else:
                    cc = (ans-3)//2+3
                return (3 + 2 * (cc - 1) + 3)*cc // 2 >= x

            ans1 = dfs(x)
            ans2 = BinarySearch().find_int_left(1, 2**63, find)
            #print(x, ans1, ans2)
            if ans1 == pre:
                cnt += 1
            else:
                print(ans1-pre, pre, cnt)
                pre = ans1
                cnt = 1
            #assert ans1 == ans2

        # for _ in range(ac.read_int()):
        #     x = ac.read_int()
        #
        #     def find(ans):
        #         return (3 + 2 * (ans - 1) + 3) // 2 >= x
        #
        #     ans2 = BinarySearch().find_int_left(1, 2 ** 63, find)
        #     ac.st(ans2)
        return


Solution().main()
