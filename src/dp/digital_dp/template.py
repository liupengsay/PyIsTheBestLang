
import unittest
from functools import lru_cache

from utils.fast_io import FastIO



class DigitalDP:
    def __init__(self):
        return

    @staticmethod
    def count_bin(n):
        # 模板: 计算 1 到 n 的正整数二进制位 1 出现的次数
        @lru_cache(None)
        def dfs(i, is_limit, is_num, cnt):
            if i == m:
                if is_num:
                    return cnt
                return 0
            res = 0
            if not is_num:
                res += dfs(i + 1, False, False, cnt)
            low = 0 if is_num else 1
            high = int(st[i]) if is_limit else 1
            for x in range(low, high + 1):
                res += dfs(i + 1, is_limit and high == x, True, cnt + int(i == w) * x)
            return res

        st = bin(n)[2:]
        m = len(st)
        ans = []  # 从二进制高位到二进制低位
        for w in range(m):
            cur = dfs(0, True, False, 0)
            ans.append(cur)
            dfs.cache_clear()
        return ans

    @staticmethod
    def count_digit(num, d):
        # 模板: 计算 1到 num 内数位 d 出现的个数
        @lru_cache(None)
        def dfs(i, cnt, is_limit, is_num):
            if i == n:
                if is_num:
                    return cnt
                return 0
            res = 0
            if not is_num:
                res += dfs(i + 1, 0, False, False)

            floor = 0 if is_num else 1
            ceil = int(s[i]) if is_limit else 9
            for x in range(floor, ceil + 1):
                res += dfs(i + 1, cnt + int(x == d), is_limit and ceil == x, True)
            return res

        s = str(num)
        n = len(s)
        return dfs(0, 0, True, False)

    @staticmethod
    def count_digit_iteration(num, d):
        # 模板: 计算 1到 num 内数位 d 出现的个数
        assert num >= 1
        s = str(num)
        n = len(s)
        dp = [[[[0] * 2 for _ in range(2)] for _ in range(n + 2)] for _ in range(n + 1)]
        # 数位 计数 是否受限 是否为数字
        for i in range(n, -1, -1):
            for cnt in range(n, -1, -1):
                for is_limit in range(1, -1, -1):
                    for is_num in range(1, -1, -1):
                        if i == n:
                            dp[i][cnt][is_limit][is_num] = cnt if is_num else 0
                            continue
                        res = 0
                        if not is_num:
                            res += dp[i + 1][0][0][0]
                        floor = 0 if is_num else 1
                        ceil = int(s[i]) if is_limit else 9
                        for x in range(floor, ceil + 1):
                            res += dp[i + 1][cnt + int(x == d)][int(is_limit and x == ceil)][1]
                        dp[i][cnt][is_limit][is_num] = res
        return dp[0][0][1][0]

    @staticmethod
    def count_num_base(num, d):
        # 模板: 使用进制计算 1 到 num 内不含数位 d 的数字个数
        assert 1 <= d <= 9  # 不含 0 则使用数位 DP 进行计算
        s = str(num)
        i = s.find(str(d))
        if i != -1:
            if d:
                s = s[:i] + str(d - 1) + (len(s) - i - 1) * "9"
            else:
                s = s[:i - 1] + str(int(s[i - 1]) - 1) + (len(s) - i - 1) * "9"
            num = int(s)

        lst = []
        while num:
            lst.append(num % 10)
            if d and lst[-1] >= d:
                lst[-1] -= 1
            elif not d and lst[-1] == 0:  # 不含 0 应该怎么算
                num *= 10
                num -= 1
                lst.append(num % 10)
            num //= 10
        lst.reverse()

        ans = 0
        for x in lst:
            ans *= 9
            ans += x
        return ans

    @staticmethod
    def count_num_dp(num, d):

        # 模板: 使用进制计算 1 到 num 内不含数位 d 的数字个数
        assert 0 <= d <= 9

        @lru_cache(None)
        def dfs(i: int, is_limit: bool, is_num: bool) -> int:
            if i == m:
                return int(is_num)

            res = 0
            if not is_num:  # 可以跳过当前数位
                res = dfs(i + 1, False, False)
            up = int(s[i]) if is_limit else 9
            for x in range(0 if is_num else 1, up + 1):  # 枚举要填入的数字 d
                if x != d:
                    res += dfs(i + 1, is_limit and x == up, True)
            return res

        s = str(num)
        m = len(s)
        return dfs(0, True, False)

    @staticmethod
    def get_kth_without_d(k, d):
        # 模板: 使用进制计算第 k 个不含数位 d 的数 0<=d<=9
        lst = []
        st = list(range(10))
        st.remove(d)
        while k:
            if d:
                lst.append(k % 9)
                k //= 9
            else:
                lst.append((k - 1) % 9)
                k = (k - 1) // 9
        lst.reverse()
        # 也可以使用二分加数位 DP 进行求解
        ans = [str(st[i]) for i in lst]
        return int("".join(ans))


