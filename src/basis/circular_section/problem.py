"""
Algorithm：circular_section|permutation_circle
Description：implemention|hash|list|index|circular_section

====================================LeetCode====================================
957（https://leetcode.cn/problems/prison-cells-after-n-days/）circular_section
418（https://leetcode.cn/problems/sentence-screen-fitting/）circular_section
466（https://leetcode.cn/problems/count-the-repetitions/）circular_section
1806（https://leetcode.cn/problems/minimum-number-of-operations-to-reinitialize-a-permutation/description/）circular_section

=====================================LuoGu======================================
P1965（https://www.luogu.com.cn/problem/P1965）circular_section
P1532（https://www.luogu.com.cn/problem/P1532）circular_section
P2203（https://www.luogu.com.cn/problem/P2203）circular_section
P5550（https://www.luogu.com.cn/problem/P5550）circular_section|matrix_fast_power|dp
P7318（https://www.luogu.com.cn/problem/P7318）circular_section
P7681（https://www.luogu.com.cn/problem/P7681）prefix_sum|circular_section
P1468（https://www.luogu.com.cn/problem/P1468）state_compression|circular_section
P6148（https://www.luogu.com.cn/problem/P6148）circular_section|implemention

===================================CodeForces===================================
1342C（https://codeforces.com/problemset/problem/1342/C）circular_section|counter
1875B（https://codeforces.com/contest/1875/problem/B）circle_section
1760F（https://codeforces.com/contest/1760/problem/F）circle_section|brute_force


"""
import math
from typing import List

from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_957(cells: List[int], n: int) -> List[int]:
        """
        url: https://leetcode.cn/problems/prison-cells-after-n-days/
        tag: circular_section
        """
        m = len(cells)
        dct = dict()
        state = []
        day = 0
        while day < n:
            busy = set([i for i in range(1, m - 1)
                        if cells[i - 1] == cells[i + 1]])
            cells = [1 if i in busy else 0 for i in range(m)]
            day += 1
            state.append(cells[:])
            if tuple(cells) in dct:
                break
            dct[tuple(cells)] = day

        i = dct[tuple(cells)]
        j = day
        if j == n:
            k = n
        else:
            k = i + (n - i) % (j - i)
        return state[k - 1]

    @staticmethod
    def lc_1806(n: int) -> int:
        """
        url: https://leetcode.cn/problems/minimum-number-of-operations-to-reinitialize-a-permutation/description/
        tag: circular_section|permutation_circle|classical|multiplication_method
        """
        ans = 0
        visit = [0] * n
        for i in range(n):
            if visit[i]:
                continue
            cur = 0
            x = i
            while not visit[x]:
                visit[x] = 1
                cur += 1
                if x % 2 == 0:
                    x //= 2
                else:
                    x = n // 2 + (x - 1) // 2
            ans = math.lcm(ans, cur) if ans else cur
        return ans

    @staticmethod
    def lg_p1468(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1468
        tag: state_compression|circular_section|classical
        """
        n = ac.read_int()
        op1 = (1 << n) - 1
        op2 = sum(1 << i for i in range(0, n, 2))
        op3 = sum(1 << i for i in range(1, n, 2))
        op4 = sum(1 << i for i in range(0, n, 3))
        stack = [[(1 << n) - 1]]
        ans = []
        while stack:
            path = stack.pop()
            for op in [op1, op2, op3, op4]:
                if path[-1] ^ op in path:
                    ans.append(path[:])
                else:
                    stack.append(path + [path[-1] ^ op])

        c = ac.read_int()
        light = sum(1 << (i - 1) for i in ac.read_list_ints()[:-1])
        down = sum(1 << (i - 1) for i in ac.read_list_ints()[:-1])
        res = set()
        for p in ans:
            m = len(p)
            state = p[c % m]
            if state & light == light and state & down == 0:
                r = bin(state)[2:]
                res.add("0" * (n - len(r)) + r)
        if not res:
            ac.st("IMPOSSIBLE")
            return
        res = sorted([r[::-1] for r in res])
        for r in res:
            ac.st(r)
        return

    @staticmethod
    def lg_p6148_1(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6148
        tag: circular_section|implemention|permutation_circle|classical|multiplication_method|classical
        """
        n, m, k = ac.read_list_ints()
        nums = [ac.read_list_ints_minus_one() for _ in range(m)]
        nex = [-1] * n
        for i in range(n):
            x = i
            for a, b in nums:
                if a <= x <= b:
                    x = a + b - x
            nex[i] = x

        ans = [0] * n
        for i in range(n):
            if ans[i]:
                continue
            lst = [i]
            while nex[lst[-1]] != lst[0]:
                lst.append(nex[lst[-1]])
            m = len(lst)
            for j in range(m):
                ans[lst[(j + k) % m]] = lst[j] + 1
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lg_p6148_2(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6148
        tag: circular_section|implemention|permutation_circle|classical|multiplication_method|classical
        """
        n, m, k = ac.read_list_ints()
        nums = [ac.read_list_ints_minus_one() for _ in range(m)]
        nex = [-1] * n
        for i in range(n):
            x = i
            for a, b in nums:
                if a <= x <= b:
                    x = a + b - x
            nex[x] = i

        ans = list(range(n))
        while k:
            if k & 1:
                ans = [nex[i] for i in ans]
            k >>= 1
            nex = [nex[i] for i in nex]

        for i in ans:
            ac.st(i + 1)
        return
