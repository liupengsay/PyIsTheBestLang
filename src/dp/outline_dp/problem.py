"""
Algorithm：outline_dp
Description：make the matrix_state_dp change to outline_dp with flatten matrix to linear

====================================LeetCode====================================
4（https://leetcode.cn/problems/broken-board-dominoes/）outline_dp|classical|hungarian
1349（https://leetcode.cn/problems/maximum-students-taking-exam/）outline_dp|classical
1659（https://leetcode.cn/problems/maximize-grid-happiness/）outline_dp|classical


=====================================LuoGu======================================
P2704（https://www.luogu.com.cn/problem/P2704）outline_dp|classical

===================================CodeForces===================================
xx（xxx）xxxxxxxxxxxxxxxxxxxx


=======================================Other====================================
1400（https://vjudge.net/problem/HDU-1400）outline_dp|classical

"""
from collections import defaultdict
from typing import List

from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1659(m: int, n: int, introverts: int, extroverts: int) -> int:
        """
        url: https://leetcode.cn/problems/maximize-grid-happiness/
        tag: outline_dp|classical
        """
        pre = {(0, introverts, extroverts): 0}
        mask = 3 ** (n - 1)

        for i in range(m):
            for j in range(n):
                cur = defaultdict(int)
                for s, intro, ext in pre:
                    cur[(s % mask) * 3, intro, ext] = max(cur[(s % mask) * 3, intro, ext], pre[(s, intro, ext)])
                    left = s % 3
                    up = s // mask
                    if ext:
                        val = 40
                        if left and j:
                            val += 20 + 50 * left - 80
                        if up and i:
                            val += 20 + 50 * up - 80
                        cur[((s % mask) * 3 + 2, intro, ext - 1)] = max(cur[((s % mask) * 3 + 2, intro, ext - 1)],
                                                                        pre[(s, intro, ext)] + val)
                    if intro:
                        val = 120
                        if left and j:
                            val += -30 + 50 * left - 80
                        if up and i:
                            val += -30 + 50 * up - 80
                        cur[((s % mask) * 3 + 1, intro - 1, ext)] = max(cur[((s % mask) * 3 + 1, intro - 1, ext)],
                                                                        pre[(s, intro, ext)] + val)
                pre = cur
        return max(pre.values())

    @staticmethod
    def lc_1349(seats: List[List[str]]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-students-taking-exam/
        tag: outline_dp|classical
        """
        m, n = len(seats), len(seats[0])
        pre = {0: 0}
        mask = (1 << (n + 1)) - 1
        for i in range(m):
            for j in range(n):
                cur = defaultdict(int)
                x = seats[i][j]
                for p in pre:
                    cur[(p << 1) & mask] = max(cur[(p << 1) & mask], pre[p])
                    if x == ".":
                        if j and p & 1:
                            continue
                        if i and j and p & (1 << n):
                            continue
                        if i and j + 1 < n and p & (1 << (n - 2)):
                            continue
                        cur[((p << 1) | 1) & mask] = max(cur[((p << 1) | 1) & mask], pre[p] + 1)
                pre = cur
        ans = max(pre.values())
        return ans

    @staticmethod
    def lc_4(n: int, m: int, broken: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/broken-board-dominoes/
        tag: outline_dp|classical|hungarian
        """
        pre = {0: 0}
        grid = [[0] * n for _ in range(m)]
        for i, j in broken:
            grid[i][j] = 1
        mask = (1 << n) - 1
        for i in range(m):
            for j in range(n):
                cur = defaultdict(int)
                for p in pre:
                    cur[(p << 1) & mask] = max(cur[(p << 1) & mask], pre[p])
                    if not grid[i][j]:
                        if j and not grid[i][j - 1] and not p & 1:
                            cur[((p << 1) | 3) & mask] = max(cur[((p << 1) | 3) & mask], pre[p] + 1)
                        if i and not grid[i - 1][j] and not p & (1 << (n - 1)):
                            cur[((p << 1) | 1) & mask] = max(cur[((p << 1) | 1) & mask], pre[p] + 1)
                pre = cur
        return max(pre.values())

    @staticmethod
    def lg_p2704(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2704
        tag: outline_dp|classical
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]
        pre = defaultdict(int)
        pre[0] = ans = 0
        mask = (1 << (2 * n)) - 1
        for i in range(m):
            for j in range(n):
                cur = defaultdict(int)
                for p in pre:
                    pp = (p & mask) << 1
                    cur[pp] = max(cur[pp], pre[p])
                    if grid[i][j] == "P":
                        pp = ((p & mask) << 1) | 1
                        if j - 1 >= 0 and pp & 2:
                            continue
                        if j - 2 >= 0 and pp & 4:
                            continue
                        if pp & (1 << (2 * n)):
                            continue
                        if pp & (1 << n):
                            continue
                        cur[pp] = max(cur[pp], pre[p] + 1)
                        ans = max(ans, cur[pp])
                pre = cur
        ac.st(ans)
        return
