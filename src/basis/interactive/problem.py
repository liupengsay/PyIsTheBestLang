"""
Algorithm：binary_search|interactive|game
Description：

====================================LeetCode====================================
xx（xxx）xxxxxxxxxxxxxxxxxxxx

=====================================LuoGu======================================
xx（xxx）xxxxxxxxxxxxxxxxxxxx

===================================CodeForces===================================
1479A（https://codeforces.com/problemset/problem/1479/A）interactive
1486C2（https://codeforces.com/problemset/problem/1486/C2）interactive
1503B（https://codeforces.com/problemset/problem/1503/B）interactive
1624F（https://codeforces.com/contest/1624/problem/F）binary_search|interactive
1713D（https://codeforces.com/contest/1713/problem/D）binary_search|interactive
1846F（https://codeforces.com/problemset/problem/1846/F）interactive
1697D（https://codeforces.com/contest/1697/problem/D）strictly_binary_search|interactive|find_int_right_strictly
1729E（https://codeforces.com/problemset/problem/1729/E）interactive
1762D（https://codeforces.com/problemset/problem/1762/D）interactive
1903E（https://codeforces.com/problemset/problem/1903/E）interactive
1918E（https://codeforces.com/contest/1918/problem/E）interactive|binary_search|quick_sort
1807E（https://codeforces.com/contest/1807/problem/E）interactive|binary_search
1520F2（https://codeforces.com/contest/1520/problem/F2）segment_tree|interactive
1624F（https://codeforces.com/contest/1624/problem/F）interactive|strictly_binary_search
1846F（https://codeforces.com/contest/1846/problem/F）interactive
1934C（https://codeforces.com/contest/1934/problem/C）interactive|brain_teaser
1937C（https://codeforces.com/contest/1937/problem/C）interactive|brain_teaser
1973D（https://codeforces.com/contest/1973/problem/D）interactive|brain_teaser

===================================AtCoder===================================
ABC313D（https://atcoder.jp/contests/abc313/tasks/abc313_d）interactive|brain_teaser
ABC305F（https://atcoder.jp/contests/abc305/tasks/abc305_f）interactive|brain_teaser|spanning_tree|dfs|back_trace
ABC282F（https://atcoder.jp/contests/abc282/tasks/abc282_f）brain_teaser|tree_array|interactive|classical
ABC269E（https://atcoder.jp/contests/abc269/tasks/abc269_e）binary_search_strictly|interactive|classical


"""
import bisect
import random
import sys
from collections import deque

from src.basis.binary_search.template import BinarySearch
from src.data_structure.segment_tree.template import RangeAddPointGet
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1713d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1713/problem/D
        tag: binary_search|interactive
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = list(range(1, 2 ** n + 1))
            while len(nums) >= 4:
                n = len(nums)
                ans = []
                for i in range(0, n, 4):
                    a, b, c, d = nums[i], nums[i + 1], nums[i + 2], nums[i + 3]
                    x = ac.inter_ask(["?", a, c])
                    if x == 1:
                        y = ac.inter_ask(["?", a, d])
                        if y == 1:
                            ans.append(a)
                        else:
                            ans.append(d)
                    elif x == 2:
                        y = ac.inter_ask(["?", b, c])
                        if y == 1:
                            ans.append(b)
                        else:
                            ans.append(c)
                    else:
                        y = ac.inter_ask(["?", b, d])
                        if y == 1:
                            ans.append(b)
                        else:
                            ans.append(d)

                nums = ans[:]
            if len(nums) == 1:
                ac.inter_out(["!", nums[0]])
            else:
                a, b = nums[:]
                x = ac.inter_ask(["?", a, b])
                ac.inter_out(["!", a if x == 1 else b])
        return

    @staticmethod
    def cf_1697d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1697/problem/D
        tag: strictly_binary_search|interactive
        """
        n = ac.read_int()
        ans = [""] * n
        dct = dict()

        ac.lst(["?", 1, 1])
        sys.stdout.flush()
        w = ac.read_str()
        ans[0] = w
        dct[w] = 0

        pre = 1
        for i in range(1, n):
            ac.lst(["?", 2, 1, i + 1])
            sys.stdout.flush()
            cur = ac.read_int()
            if cur > pre:
                ac.lst(["?", 1, i + 1])
                sys.stdout.flush()
                w = ac.read_str()
                ans[i] = w
                dct[w] = i
                pre = cur
            else:
                lst = sorted(dct.values())
                m = len(lst)
                low, high = 0, m - 1
                while low < high:
                    mid = low + (high - low + 1) // 2
                    target = len(set(ans[lst[mid]:i]))
                    ac.lst(["?", 2, lst[mid] + 1, i + 1])
                    sys.stdout.flush()
                    cur = ac.read_int()
                    if cur == target:
                        low = mid
                    else:
                        high = mid - 1
                ans[i] = ans[lst[high]]
                dct[ans[i]] = i
        ac.lst(["!", "".join(ans)])
        sys.stdout.flush()
        return

    @staticmethod
    def cf_1918e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1918/problem/E
        tag: binary_search|interactive|quick_sort
        """

        ac.flush = True

        def query(x):
            ac.lst(["?", x + 1])
            cur = ac.read_str()
            if cur == ">":
                return 1
            elif cur == "<":
                return -1
            return 0

        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = [0] * n
            stack = [[1, n, list(range(n))]]
            while stack:
                left, right, ind = stack.pop()
                mid = ind[random.randint(0, len(ind) - 1)]
                while query(mid):
                    continue

                smaller = []
                bigger = []
                for i in ind:
                    if i == mid:
                        continue
                    if query(i) == 1:
                        bigger.append(i)
                    else:
                        smaller.append(i)
                    query(mid)
                nums[mid] = left + len(smaller)
                if left < nums[mid]:
                    stack.append([left, nums[mid] - 1, smaller])
                if nums[mid] < right:
                    stack.append([nums[mid] + 1, right, bigger])
            ac.lst(["!"] + nums)
        return

    @staticmethod
    def cf_1520f2(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1520/problem/F2
        tag: segment_tree|interactive
        """
        n, t = ac.read_list_ints()
        tree = RangeAddPointGet(n)
        tree.build([-2 * t - 1] * n)
        for _ in range(t):
            k = ac.read_int()

            def check(x):
                res = tree.point_get(x - 1)
                if res < 0:
                    ac.lst(["?", 1, x])
                    cur = ac.read_int()
                    tree.range_add(x - 1, x - 1, cur - res)
                    res = cur
                return x - res >= k

            ans = BinarySearch().find_int_left(1, n, check)
            tree.range_add(ans - 1, n - 1, 1)
            ac.lst(["!", ans])
        return

    @staticmethod
    def cf_1937c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1937/problem/C
        tag: interactive|brain_teaser
        """
        ac.flush = True

        def ask(tmp):
            ac.lst(tmp)
            return ac.read_str()

        for _ in range(ac.read_int()):
            n = ac.read_int()
            zero = 0
            for i in range(1, n):
                x = ask(["?", zero, zero, i, i])
                if x == "<":
                    zero = i

            nex = 1 if zero == 0 else 0
            lst = [nex]
            for i in range(n):
                if i == nex or i == zero:
                    continue
                x = ask(["?", zero, lst[-1], zero, i])
                if x == "<":
                    lst = [i]
                elif x == "=":
                    lst.append(i)
            nex = lst[0]
            for i in lst[1:]:
                x = ask(["?", nex, nex, i, i])
                if x == ">":
                    nex = i
            ac.lst(["!", zero, nex])
        return

    @staticmethod
    def cf_1934c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1934/problem/C
        tag: interactive|brain_teaser
        """
        ac.flush = True

        def ask(lst):
            ac.lst(lst)
            return ac.read_int()

        for _ in range(ac.read_int()):
            m, n = ac.read_list_ints()
            a = ask(["?", 1, 1])
            b = ask(["?", m, 1])
            y = (a + 2 + b + 1 - m) // 2
            x = a + 2 - y
            if 1 <= x <= m and 1 <= y <= n and ask(["?", x, y]) == 0:
                ac.lst(["!", x, y])
                continue
            c = ask(["?", m, n])
            y = (n + b + 1 - c) // 2
            x = m + n - c - y
            ac.lst(["!", x, y])
        return

    @staticmethod
    def abc_305f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc305/tasks/abc305_f
        tag: interactive|brain_teaser|spanning_tree|dfs|back_trace
        """
        ac.flush = True
        n, m = ac.read_list_ints()
        parent = [-1] * (n + 1)
        dct = [deque() for _ in range(n + 1)]
        lst = ac.read_list_ints()
        dct[1] = deque(lst[1:])
        visit = [0] * (n + 1)
        visit[1] = 1
        stack = [1]
        while stack:
            x = stack[-1]
            while dct[x] and visit[dct[x][0]]:
                dct[x].popleft()
            if dct[x]:
                y = dct[x].popleft()
                parent[y] = x
                stack.append(y)
                visit[y] = 1
                ac.st(y)
                lst = ac.read_list_strs()
                if lst[0] == "OK":
                    return
                dct[y] = deque([int(w) for w in lst[1:]])
            else:
                x = stack.pop()
                y = parent[x]
                stack.append(y)
                ac.st(y)
                ac.read_list_strs()
        return

    @staticmethod
    def abc_282f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc282/tasks/abc282_f
        tag: brain_teaser|tree_array|interactive|classical
        """
        ac.flush = True
        n = ac.read_int()
        start = [[] for _ in range(n)]
        end = [[] for _ in range(n)]
        lst = []
        for i in range(n):
            lst.append((i, i))
            start[i].append(i)
            end[i].append(i)
            cnt = 1
            while i + cnt < n:
                lst.append((i, i + cnt))
                start[i].append(i + cnt)
                end[i + cnt].append(i)
                cnt *= 2
        dct = {(ls[0], ls[1]): i for i, ls in enumerate(lst)}
        ac.st(len(lst))
        for ls in lst:
            ac.lst([ls[0] + 1, ls[1] + 1])

        for _ in range(ac.read_int()):
            ll, rr = ac.read_list_ints_minus_one()
            mid_ll = start[ll][bisect.bisect_right(start[ll], rr) - 1]
            mid_rr = end[rr][bisect.bisect_left(end[rr], ll)]
            a = dct[(ll, mid_ll)]
            b = dct[(mid_rr, rr)]
            ac.lst([a + 1, b + 1])
        return

    @staticmethod
    def abc_269e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc269/tasks/abc269_e
        tag: binary_search_strictly|interactive|classical
        """
        ac.flush = True
        n = ac.read_int()

        def check1(x):
            ac.lst(["?"] + [1, x + 1, 1, n])
            cnt = ac.read_int()
            return cnt < x + 1

        def check2(x):
            ac.lst(["?"] + [1, n, 1, x + 1])
            cnt = ac.read_int()
            return cnt < x + 1

        row = BinarySearch().find_int_left_strictly(0, n - 1, check1)
        col = BinarySearch().find_int_left_strictly(0, n - 1, check2)
        ac.lst(["!", row + 1, col + 1])
        return
