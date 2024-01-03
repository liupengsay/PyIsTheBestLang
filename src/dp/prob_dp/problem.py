"""
Algorithm：prob_dp
Description：comb|specific_plan|prob|expectation

====================================LeetCode====================================
1227（https://leetcode.cn/problems/airplane-seat-assignment-probability/）prob_dp

=====================================LuoGu======================================
P1291（https://www.luogu.com.cn/problem/P1291）liner_dp|expectation
P4316（https://www.luogu.com.cn/problem/P4316）expectation|reverse_graph|topological_sort
P6154（https://www.luogu.com.cn/problem/P6154）reverse_graph|expectation|tree_dp|float|mod

=====================================AcWing=====================================
5058（https://www.acwing.com/problem/content/description/5061/）prob_dp


"""
import math
from collections import deque

from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1291(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1291
        tag: liner_dp|expectation
        """
        n = ac.read_int()
        ans = [1, 1]
        for x in range(2, n + 1):
            a, b = ans
            c, d = 1, x
            g = math.gcd(b, d)
            lcm = b * d // g
            a, b = a * lcm // b + c * lcm // d, lcm
            g = math.gcd(a, b)
            ans = [a // g, b // g]
        a, b = ans
        a *= n
        x = a // b
        a %= b
        if a == 0:
            ac.st(x)
            return
        g = math.gcd(a, b)
        ans = [a // g, b // g]
        a, b = ans
        ac.st(len(str(x)) * " " + str(a))
        ac.st(str(x) + "-" * len(str(b)))
        ac.st(len(str(x)) * " " + str(b))
        return

    @staticmethod
    def lg_p4316(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4316
        tag: expectation|reverse_graph|topological_sort|float
        """
        n, m = ac.read_list_ints()
        dp = [0 for _ in range(n)]
        degree = [0] * n
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            a, b, w = ac.read_list_ints()
            a -= 1
            b -= 1
            dct[b][a] = w
            degree[a] += 1
        cnt = degree[:]

        stack = deque([n - 1])
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                dp[j] += dp[i] + dct[i][j]
                degree[j] -= 1
                if not degree[j]:
                    dp[j] /= cnt[j]
                    stack.append(j)
        ans = "%.2f" % (dp[0])
        ac.st(ans)
        return

    @staticmethod
    def lg_p6154(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6154
        tag: reverse_graph|expectation|tree_dp|float|mod
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            dct[y].append(x)
            degree[x] += 1
        length_sum = [0] * n
        path_cnt = [0] * n
        mod = 998244353
        stack = deque([i for i in range(n) if not degree[i]])
        for i in stack:
            path_cnt[i] = 1
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                degree[j] -= 1
                length_sum[j] += path_cnt[i] + length_sum[i]
                path_cnt[j] += path_cnt[i]
                if not degree[j]:
                    path_cnt[j] += 1
                    path_cnt[j] %= mod
                    length_sum[j] %= mod
                    stack.append(j)
        total_length = sum(length_sum) % mod
        total_cnt = sum(path_cnt) % mod
        ac.st(total_length * pow(total_cnt, -1, mod) % mod)
        return

    @staticmethod
    def ac_5058(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/5061/
        tag: prob_dp
        """
        w, b = ac.read_list_ints()
        dp = [[0] * (b + 1) for _ in range(w + 1)]
        for i in range(1, w + 1):
            dp[i][0] = 1
        for i in range(1, w + 1):
            for j in range(1, b + 1):
                p = i / (i + j)
                if j > 1:
                    p += j / (i + j) * (j - 1) / (i + j - 1) * i / (i + j - 2) * dp[i - 1][j - 2]
                if j > 2:
                    p += j / (i + j) * (j - 1) / (i + j - 1) * (j - 2) / (i + j - 2) * dp[i][j - 3]
                dp[i][j] = p
        ac.st(dp[w][b])
        return
