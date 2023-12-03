"""
算法：概率DP
功能：根据组合数与转移方案求解概率或者期望
题目：

====================================LeetCode====================================
1227（https://leetcode.com/problems/airplane-seat-assignment-probability/）概率DP

=====================================LuoGu======================================
2719（https://www.luogu.com.cn/record/list?user=739032&status=12&page=1）二维DP求概率
1291（https://www.luogu.com.cn/problem/P1291）线性DP求期望
4316（https://www.luogu.com.cn/problem/P4316）经典期望 DP 反向建图与拓扑排序
6154（https://www.luogu.com.cn/problem/P6154）经典反向建图期望树形 DP 与有理数取模

=====================================AcWing=====================================
5058（https://www.acwing.com/problem/content/description/5061/）经典概率DP


参考：OI WiKi（xx）
"""
import math
from collections import deque
from functools import lru_cache

from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):

        # 模板：经典记忆化二维 DP 模拟搜索转移计算概率

        @lru_cache(None)
        def dfs(a, b):
            if a + b == 2:
                if a == 0 or b == 0:
                    return 1
                return 0
            if a == 0 or b == 0:
                return 1
            res = (dfs(a - 1, b) + dfs(a, b - 1)) / 2
            return res

        n = ac.read_int() // 2
        if n == 0:
            ans = 0
        else:
            ans = dfs(n, n)
        ac.st("%.4f" % ans)
        return

    @staticmethod
    def lg_p1291(ac=FastIO()):

        # 模板：线性DP求期望，使用分数加减运算
        n = ac.read_int()
        ans = [1, 1]
        for x in range(2, n + 1):
            a, b = ans
            c, d = 1, x
            # 使用gcd进行加减
            g = math.gcd(b, d)
            lcm = b * d // g
            a, b = a * lcm // b + c * lcm // d, lcm
            g = math.gcd(a, b)
            ans = [a // g, b // g]
        # f[i] = f[i-1] + n/(n-i+1) 表示已经有 i-1 个再要有新的一个的期望为 n/(n-i+1)
        a, b = ans
        a *= n
        x = a // b
        a %= b
        if a == 0:
            ac.st(x)
            return
        # 加和化简
        g = math.gcd(a, b)
        ans = [a // g, b // g]
        a, b = ans
        ac.st(len(str(x)) * " " + str(a))
        ac.st(str(x) + "-" * len(str(b)))
        ac.st(len(str(x)) * " " + str(b))
        return

    @staticmethod
    def lg_p4316(ac=FastIO()):
        # 模板：期望 DP 反向建图与拓扑排序
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

        # 反向拓扑排序与状态转移计算
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
        # 模板：经典反向建图期望树形 DP 与有理数取模
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            dct[y].append(x)
            degree[x] += 1
        # 当前节点为起点的路径总长度
        length_sum = [0] * n
        # 当前节点为起点的路径总数
        path_cnt = [0] * n
        mod = 998244353
        stack = deque([i for i in range(n) if not degree[i]])
        for i in stack:
            path_cnt[i] = 1
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                degree[j] -= 1
                # 路径总长度增加
                length_sum[j] += path_cnt[i] + length_sum[i]
                # 路径条数增加
                path_cnt[j] += path_cnt[i]
                if not degree[j]:
                    # 新增一条以当前点开始和结束的路径即起点与终点可以相同
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
        # 模板：经典概率DP
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