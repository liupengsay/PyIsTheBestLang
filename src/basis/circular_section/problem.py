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
1237D（https://codeforces.com/problemset/problem/1237/D）two_pointers|implemention|circular_array
1372D（https://codeforces.com/problemset/problem/1372/D）circular_array|brain_teaser|brute_force|observation|circular_to_linear

===================================AtCoder===================================
ABC258E（https://atcoder.jp/contests/abc258/tasks/abc258_e）two_pointers|brute_force|circle_section|classical
ABC244D（https://atcoder.jp/contests/abc244/tasks/abc244_d）dfs|back_trace|brute_force|circular_section
ABC241E（https://atcoder.jp/contests/abc241/tasks/abc241_e）circular_section|brute_force_valid|classical
ABC214C（https://atcoder.jp/contests/abc214/tasks/abc214_c）circular_section|brain_teaser
ABC179E（https://atcoder.jp/contests/abc179/tasks/abc179_e）circular_section|data_range|classical
ABC175D（https://atcoder.jp/contests/abc175/tasks/abc175_d）permutation_circle|brute_force|prefix_sum

"""
import math
from collections import deque
from itertools import combinations
from typing import List

from src.util.fast_io import FastIO


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

    @staticmethod
    def abc_258e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc258/tasks/abc258_e
        tag: two_pointers|brute_force|circle_section|classical
        """
        n, q, x = ac.read_list_ints()
        w = ac.read_list_ints()
        nex = [-1] * n
        t = sum(w)
        mid = n * (x // t)
        x %= t
        pre = j = 0
        for i in range(n):
            if x == 0:
                nex[i] = i
            else:
                while pre < x:
                    pre += w[j % n]
                    j += 1
                nex[i] = j % n
            pre -= w[i]

        dct = dict()
        xx = 0
        lst = []
        while xx not in dct:
            dct[xx] = len(lst)
            lst.append(xx)
            xx = nex[xx]

        ind = dct[xx]
        length = len(dct)
        for _ in range(q):
            k = ac.read_int() - 1
            if x == 0:
                ac.st(mid)
                continue
            if k < len(lst):
                xx = lst[k]
            else:
                circle = length - ind
                k -= length
                j = k % circle
                xx = lst[ind + j]
            if xx < nex[xx]:
                ac.st(nex[xx] - xx + mid)
            else:
                ac.st(n + nex[xx] - xx + mid)
        return

    @staticmethod
    def abc_244d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc244/tasks/abc244_d
        tag: dfs|back_trace|brute_force|circular_section
        """
        s = ac.read_list_strs()
        t = ac.read_list_strs()
        m = 10 ** 18
        ans = [False]

        def dfs(cur):
            if ans[0]:
                return
            for item in combinations([0, 1, 2], 2):
                i, j = item
                tmp = cur[:]
                tmp[i], tmp[j] = tmp[j], tmp[i]
                if tmp in pre:
                    ind = pre.index(tmp)
                    length = len(pre)
                    circle = length - ind
                    tm = m - length
                    j = tm % circle
                    if pre[ind + j] == t:
                        ans[0] = True
                        return
                else:
                    pre.append(tmp)
                    dfs(tmp[:])
                    pre.pop()

        pre = [s]
        dfs(s[:])
        ac.st("Yes" if ans[0] else "No")
        return

    @staticmethod
    def abc_241e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc241/tasks/abc241_e
        tag: circular_section|brute_force_valid|classical
        """
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        lst = [0]
        dct = dict()
        x = 0
        index = [-1]
        while True:
            x += nums[x % n]
            if x % n in dct:
                break
            lst.append(x)
            index.append(x % n)
            dct[x % n] = len(lst) - 1

        tm = k
        length = len(lst)
        # the first pos of circle section
        ind = dct[x % n]
        # current lst is enough
        if tm < length:
            ac.st(lst[tm])
            return
        tm = k - 1
        # compute by circle section
        circle = length - ind
        tm -= length - 1
        j = tm % circle
        circle_sum = sum(nums[x % n] for x in lst[ind:])
        res = lst[-1] + nums[lst[-1] % n] + sum(nums[x % n] for x in lst[ind:ind + j]) + circle_sum * (tm // circle)
        ac.st(res)
        return

    @staticmethod
    def cf_1237d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1237/D
        tag: two_pointers|implemention|circular_array
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        ans = []
        j = 0
        stack = deque()

        for i in range(n):
            while j < 3 * n and (not stack or nums[j % n] * 2 >= nums[stack[0] % n]):
                while stack and nums[stack[-1] % n] <= nums[j % n]:
                    stack.pop()
                stack.append(j)
                j += 1
            ans.append(j - i if j - i < 2 * n + (n - i) else -1)
            if stack[0] == i:
                stack.popleft()
        ac.lst(ans)
        return

    @staticmethod
    def cf_1372d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1372/D
        tag: circular_array|brain_teaser|brute_force|observation|circular_to_linear
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        nums += nums
        odd = ac.accumulate([nums[i] * (i % 2) for i in range(2 * n)])
        even = ac.accumulate([nums[i] * (1 - i % 2) for i in range(2 * n)])
        ans = 0
        for i in range(n):
            ans = max(ans, max(odd[i + n] - odd[i], even[i + n] - even[i]))
        ac.st(ans)
        return

    @staticmethod
    def abc_179e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc179/tasks/abc179_e
        tag: circular_section|data_range|classical
        """
        n, x, m = ac.read_list_ints()

        lst = [x % m]
        x %= m
        pre = {lst[0]: 0}
        for i in range(2, n + 1):
            x *= x
            x %= m
            if x in pre:
                ind = pre[x]
                ans = sum(lst)
                rest = n - (i - 1)
                c_sum = sum(lst[ind:])
                circle = i - 1 - ind
                ans += c_sum * (rest // circle)
                ans += sum(lst[ind:ind + rest % circle])
                ac.st(ans)
                return
            lst.append(x)
            pre[lst[-1]] = len(lst) - 1
        ac.st(sum(lst))
        return

    @staticmethod
    def abc_175d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc175/tasks/abc175_d
        tag: permutation_circle|brute_force|prefix_sum
        """
        n, k = ac.read_list_ints()
        p = ac.read_list_ints_minus_one()
        c = ac.read_list_ints()

        visit = [0] * n
        ans = -math.inf
        for i in range(n):
            if not visit[i]:
                lst = [i]
                visit[i] = 1
                gain = []
                while not visit[p[lst[-1]]]:
                    lst.append(p[lst[-1]])
                    visit[lst[-1]] = 1
                    gain.append(c[lst[-1]])
                gain.append(c[i])
                pre = ac.accumulate(gain + gain)
                m = len(gain)
                for j1 in range(m):
                    for j2 in range(j1, j1 + m):
                        if j2 - j1 + 1 > k:
                            break
                        ans = max(ans, pre[j2 + 1] - pre[j1] + max(0, pre[m] * ((k - (j2 - j1 + 1)) // m)))
        ac.st(ans)
        return
