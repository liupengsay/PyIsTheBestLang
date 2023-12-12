"""
Algorithm：outline_dp
Description：make the matrix_state_dp change to outline_dp with flatten matrix to linear

====================================LeetCode====================================
1659（https://leetcode.com/problems/maximize-grid-happiness/）outline_dp|classical
1349（https://leetcode.com/problems/maximum-students-taking-exam/）outline_dp|classical
04（https://leetcode.com/problems/broken-board-dominoes/）outline_dp|classical

=====================================LuoGu======================================
xx（xxx）xxxxxxxxxxxxxxxxxxxx

===================================CodeForces===================================
xx（xxx）xxxxxxxxxxxxxxxxxxxx


=======================================Other====================================
1400（https://vjudge.net/problem/HDU-1400）outline_dp|classical

"""
from functools import lru_cache
from itertools import chain
from typing import List


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1659_1(m: int, n: int, introverts: int, extroverts: int) -> int:
        """
        url: https://leetcode.com/problems/maximize-grid-happiness/
        tag: outline_dp|classical
        """
        # memory_searchdfs|轮廓线 DP
        def dfs(pos, state, intro, ext):
            # 当前网格位置，前 n 个格子压缩状态，剩余内向人数，剩余外向人数
            res = dp[pos][state][intro][ext]
            if res != -1:
                return res
            if not intro and not ext:
                dp[pos][state][intro][ext] = 0
                return 0
            if pos == m * n:
                dp[pos][state][intro][ext] = 0
                return 0
            res = dfs(pos + 1, 3 * (state % s), intro, ext)
            i, j = pos // n, pos % n
            if intro:
                cur = dfs(pos + 1, 3 * (state % s) + 1, intro - 1, ext) + 120
                # 左边
                if j:
                    cur += cross[state % 3][1]
                # 上边
                if i:
                    cur += cross[state // s][1]
                res = res if res > cur else cur
            if ext:
                cur = dfs(pos + 1, 3 * (state % s) + 2, intro, ext - 1) + 40
                # 左边
                if j:
                    cur += cross[state % 3][2]
                # 上边
                if i:
                    cur += cross[state // s][2]
                res = res if res > cur else cur
            dp[pos][state][intro][ext] = res
            return res

        s = 3 ** (n - 1)
        cross = [[0, 0, 0], [0, -60, -10], [40, -10, 40]]
        # 手写memory_search内存优化
        dp = [[[[-1] * (extroverts + 1) for _ in range(introverts + 1)] for _ in range(s * 3)] for _ in
              range(m * n + 1)]
        return dfs(0, 0, introverts, extroverts)

    @staticmethod
    def lc_1659_2(m: int, n: int, introverts: int, extroverts: int) -> int:
        """
        url: https://leetcode.com/problems/maximize-grid-happiness/
        tag: outline_dp|classical
        """
        # 迭代轮廓线 DP
        s = 3 ** (n - 1)
        cross = [[0, 0, 0], [0, -60, -10], [40, -10, 40]]
        dp = [[[[0] * (extroverts + 1) for _ in range(introverts + 1)] for _ in range(s * 3)] for _ in range(m * n + 1)]
        for pos in range(m * n - 1, -1, -1):
            # 还可以滚动数组优化
            i, j = pos // n, pos % n
            for state in range(s * 3):
                for intro in range(introverts + 1):
                    for ext in range(extroverts + 1):
                        if intro == ext == 0:
                            continue
                        res = dp[pos + 1][3 * (state % s)][intro][ext]
                        if intro:
                            cur = dp[pos + 1][3 * (state % s) + 1][intro - 1][ext] + 120
                            # 左边
                            if j:
                                cur += cross[state % 3][1]
                            # 上边
                            if i:
                                cur += cross[state // s][1]
                            res = res if res > cur else cur
                        if ext:
                            cur = dp[pos + 1][3 * (state % s) + 2][intro][ext - 1] + 40
                            # 左边
                            if j:
                                cur += cross[state % 3][2]
                            # 上边
                            if i:
                                cur += cross[state // s][2]
                            res = res if res > cur else cur
                        dp[pos][state][intro][ext] = res
        return dp[0][0][introverts][extroverts]

    @staticmethod
    def lc_1659_3(m: int, n: int, introverts: int, extroverts: int) -> int:
        """
        url: https://leetcode.com/problems/maximize-grid-happiness/
        tag: outline_dp|classical
        """

        # memory_searchdfs|轮廓线 DP
        @lru_cache(None)
        def dfs(i, state, intro, ext):
            if i == m * n:
                return 0
            up = state // w if i // n else 0  # 轮廓线边界状态
            left = state % 3 if i % n else 0  # 轮廓线边界状态
            res = dfs(i + 1, (state - up * w) * 3, intro, ext)
            if intro:
                cur = dfs(i + 1, (state - up * w) * 3 + 1, intro - 1, ext) + 120
                cur += cross[1][up] + cross[1][left]
                if cur > res:
                    res = cur
            if ext:
                cur = dfs(i + 1, (state - up * w) * 3 + 2, intro, ext - 1) + 40
                cur += cross[2][up] + cross[2][left]
                if cur > res:
                    res = cur
            return res

        w = 3 ** (n - 1)
        cross = [[0, 0, 0], [0, -60, -10], [0, -10, 40]]
        return dfs(0, 0, introverts, extroverts)

    @staticmethod
    def lc_1349_1(seats: List[List[str]]) -> int:
        """
        url: https://leetcode.com/problems/maximum-students-taking-exam/
        tag: outline_dp|classical
        """
        # memory_searchdfs|轮廓线 DP

        def dfs(pos, state):
            res = dp[pos][state]
            if res != -1:
                return res
            if pos == m * n:
                res = 0
            else:
                i, j = pos // n, pos % n
                res = dfs(pos + 1, 2 * (state % s))
                if seats[i][j] != "#":
                    left = 1 if not j or not state % 2 else 0
                    left_up = 1 if not (i >= 1 and j >= 1) or not state & s else 0
                    right_up = 1 if not (i >= 1 and j <= n - 2) or not state & (1 << (n - 2)) else 0
                    if left_up and left and right_up:
                        cur = 1 + dfs(pos + 1, 2 * (state % s) + 1)
                        res = res if res > cur else cur
            dp[pos][state] = res
            return res

        m, n = len(seats), len(seats[0])
        s = 2 ** n
        dp = [[-1] * (1 << (n + 1)) for _ in range(m * n + 1)]
        return dfs(0, 0)

    @staticmethod
    def lc_1349_2(seats: List[List[str]]) -> int:
        """
        url: https://leetcode.com/problems/maximum-students-taking-exam/
        tag: outline_dp|classical
        """
        # 滚动数组迭代轮廓线 DP
        m, n = len(seats), len(seats[0])
        s = 2 ** n
        dp = [[-1] * (2 * s) for _ in range(2)]
        pre = 0
        for pos in range(m * n - 1, -1, -1):
            nex = 1 - pre
            i, j = pos // n, pos % n
            for state in range(2 * s):
                # state 表示前 n + 1 个网格的状态
                res = dp[pre][2 * (state % s)]
                if seats[i][j] != "#":
                    left = 1 if not j or not state % 2 else 0
                    left_up = 1 if not (i >= 1 and j >= 1) or not state & s else 0
                    right_up = 1 if not (i >= 1 and j <= n - 2) or not state & (1 << (n - 2)) else 0
                    if left_up and left and right_up:
                        cur = 1 + dp[pre][2 * (state % s) + 1]
                        res = res if res > cur else cur
                dp[nex][state] = res
            pre = nex
        return dp[pre][0]

    @staticmethod
    def lc_1349_3(seats: List[List[str]]) -> int:
        """
        url: https://leetcode.com/problems/maximum-students-taking-exam/
        tag: outline_dp|classical
        """

        # outline_dp|classical转成一维数组后更好写
        @lru_cache(None)
        def dfs(i, state):  # 一共四种初始转移方式即 i 为 0 或者 m*n 而 state 为 0 或者 (1<<(m*n))-1
            if i == m * n:
                return 0
            nex_state = (state - (state & 1)) // 2
            res = dfs(i + 1, nex_state)
            left = not (state & (1 << 0) and i // n and i % n)
            left_up = not (state & (1 << 2) and i // n and i % n < n - 1)
            right_up = not (state & (1 << n) and i % n)
            if lst[i] == "." and left and left_up and right_up:
                cur = 1 + dfs(i + 1, nex_state | (1 << n))  # fill_table
                if cur > res:
                    res = cur
            return res

        m, n = len(seats), len(seats[0])
        lst = chain.from_iterable(seats)
        return dfs(0, 0)

    @staticmethod
    def lcp_4_1(n: int, m: int, broken: List[List[int]]) -> int:
        # memory_searchdfs|轮廓线 DP

        def dfs(pos, state):
            # 当前位置，前 m 个格子状态
            res = dp[pos][state]
            if res != -1:
                return res
            if pos == m * n:
                res = 0
            else:
                i, j = pos // m, pos % m
                res = dfs(pos + 1, 2 * (state % s) + grid[i][j])
                if not grid[i][j]:
                    # 上边
                    if not state & s and i:
                        cur = 1 + dfs(pos + 1, 2 * (state % s) + 1)
                        res = res if res > cur else cur
                    # 左边
                    if not state % 2 and j:
                        cur = 1 + dfs(pos + 1, 2 * ((state + 1) % s) + 1)
                        res = res if res > cur else cur
            dp[pos][state] = res
            return res

        s = 1 << (m - 1)
        grid = [[0] * m for _ in range(n)]
        for a, b in broken:
            grid[a][b] = 1
        dp = [[-1] * (1 << m) for _ in range(m * n + 1)]
        return dfs(0, 0)

    @staticmethod
    def lcp_4_2(n: int, m: int, broken: List[List[int]]) -> int:
        # 滚动数组优化迭代轮廓线 DP

        s = 1 << (m - 1)
        grid = [[0] * m for _ in range(n)]
        for a, b in broken:
            grid[a][b] = 1

        dp = [[0] * (2 * s) for _ in range(2)]
        pre = 0
        for pos in range(m * n - 1, -1, -1):
            nex = 1 - pre
            i, j = pos // m, pos % m
            for state in range(2 * s):
                res = dp[pre][2 * (state % s) + grid[i][j]]
                if not grid[i][j]:
                    # 上边
                    if not state & s and i:
                        cur = 1 + dp[pre][2 * (state % s) + 1]
                        res = res if res > cur else cur
                    # 左边
                    if not state % 2 and j:
                        cur = 1 + dp[pre][2 * ((state + 1) % s) + 1]
                        res = res if res > cur else cur
                dp[nex][state] = res
            pre = nex
        return dp[pre][0]