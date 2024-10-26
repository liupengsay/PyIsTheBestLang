"""
Algorithm：meet_in_middle
Description：data_range|brute_force 

====================================LeetCode====================================
1755（https://leetcode.cn/problems/closest-subsequence-sum/）meet_in_middle
2035（https://leetcode.cn/problems/partition-array-into-two-arrays-to-minimize-sum-difference/）meet_in_middle|sort|binary_search|two_pointers
956（https://leetcode.cn/problems/tallest-billboard/description/）meet_in_middle|bag_dp

=====================================LuoGu======================================
P5194（https://www.luogu.com.cn/problem/P5194）fibonacci|meet_in_middle|brute_force|binary_search

P5691（https://www.luogu.com.cn/problem/P5691）meet_in_middle|sorted_list|two_pointers|brute_force

=====================================CodeForces=====================================
1006F（https://codeforces.com/contest/1006/problem/F）prefix_sum|hash|counter|meet_in_middle
525E（https://codeforces.com/problemset/problem/525/E）meet_in_middle
888E（https://codeforces.com/problemset/problem/888/E）meet_in_middle|classical


=====================================AtCoder=====================================
ABC326F（https://atcoder.jp/contests/abc326/tasks/abc326_f）meet_in_middle|brain_teaser|classical
ABC271F（https://atcoder.jp/contests/abc271/tasks/abc271_f）meet_in_middle|brute_force|classical


=====================================AcWing=====================================
173（https://www.acwing.com/problem/content/173/）meet_in_middle


"""

import bisect
import math
import random
from collections import defaultdict, Counter
from itertools import combinations
from typing import List

from src.util.fast_io import FastIO



class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_956_1(rods: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/tallest-billboard/description/
        tag: meet_in_middle|dp
        """

        def check(lst):
            cur = {(0, 0)}
            for num in lst:
                cur |= {(a + num, b) for a, b in cur} | {(a, b + num) for a, b in cur}
            dct = defaultdict(int)
            for a, b in cur:
                dct[a - b] = max(dct[a - b], a)
            return dct

        m = len(rods)
        pre = check(rods[:m // 2])
        post = check(rods[m // 2:])
        ans = 0
        for k in pre:
            if -k in post:
                ans = max(ans, pre[k] + post[-k])
        return ans

    @staticmethod
    def lc_956_2(rods: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/tallest-billboard/description/
        tag: meet_in_middle|dp
        """

        pre = defaultdict(int)
        pre[0] = 0
        for num in rods:
            cur = pre.copy()
            for p in pre:
                cur[p + num] = max(cur[p + num], pre[p] + num)
                cur[p - num] = max(cur[p - num], pre[p])
            pre = cur
        return pre[0]

    @staticmethod
    def lc_2035(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/partition-array-into-two-arrays-to-minimize-sum-difference/
        tag: meet_in_middle|sort|binary_search|two_pointers|brute_force|group_by_length
        """

        def check(lst):
            m = len(lst)
            total = sum(lst)
            res = [set() for _ in range(m + 1)]
            res[0].add(0)
            res[m].add(total)
            for x in range(1, m // 2 + 1):
                for item in combinations(lst, x):
                    cur = sum(item)
                    res[k].add(cur)
                    res[m - k].add(total - cur)
            return res

        def find(left, right):
            a, b = len(left), len(right)
            res = math.inf
            i = 0
            j = b - 1
            while i < a and j >= 0:
                cur = abs(target - left[i] - right[j])
                res = res if res < cur else cur
                if left[i] + right[j] == target:
                    return 0
                if left[i] + right[j] > target:
                    j -= 1
                elif left[i] + right[j] < target:
                    i += 1
            return res

        n = len(nums) // 2
        pre = check(nums[:n])
        post = check(nums[n:])
        ans = math.inf
        target = sum(nums) / 2
        for k in range(n + 1):
            ans = min(ans, 2 * find(sorted(list(pre[k])), sorted(list(post[n - k]))))
        return int(ans)

    @staticmethod
    def lg_p5194(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5194
        tag: fibonacci|meet_in_middle|brute_force|binary_search
        """

        def check(lst):
            cur = {0}
            for x in lst:
                cur |= {p + x for p in cur if p + x <= c}
            return sorted(cur)

        n, c = ac.read_list_ints()
        val = [ac.read_int() for _ in range(n)]
        val = [x for x in val if x <= c]
        n = len(val)
        res1 = check(val[:n // 2])
        res2 = check(val[n // 2:])
        ans = 0
        n = len(res1)
        j = n - 1
        for num in res2:
            while num + res1[j] > c:
                j -= 1
            ans = max(ans, num + res1[j])
        ac.st(ans)
        return

    @staticmethod
    def cf_1006f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1006/problem/F
        tag: prefix_sum|hash|counter|meet_in_middle|classical
        """
        ac.get_random_seed()
        m, n, k = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        pre = [[dict() for _ in range(n)] for _ in range(m)]
        stack = [(0, 0, grid[0][0])]
        half = (m + n - 2) // 2
        while stack:
            x, y, val = stack.pop()
            if x + y == half:
                pre[x][y][val ^ ac.random_seed] = pre[x][y].get(val ^ ac.random_seed, 0) + 1
                continue

            if x + 1 < m:
                stack.append((x + 1, y, val ^ grid[x + 1][y]))
            if y + 1 < n:
                stack.append((x, y + 1, val ^ grid[x][y + 1]))

        ans = 0
        stack = [(m - 1, n - 1, k ^ grid[m - 1][n - 1])]
        while stack:
            x, y, val = stack.pop()
            if x + y == half:
                val ^= grid[x][y]
                ans += pre[x][y].get(val ^ ac.random_seed, 0)
                continue

            if x - 1 >= 0:
                stack.append((x - 1, y, val ^ grid[x - 1][y]))
            if y - 1 >= 0:
                stack.append((x, y - 1, val ^ grid[x][y - 1]))
        ac.st(ans)
        return

    @staticmethod
    def ac_173(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/173/
        tag: meet_in_middle
        """

        def check(tmp):
            cur = {0}
            for x in tmp:
                cur |= {p + x for p in cur if p + x <= w}
            return cur

        w, n = ac.read_list_ints()
        lst = [ac.read_int() for _ in range(n)]
        lst.sort()
        pre = sorted(list(check(lst[:n // 2])))
        post = sorted(list(check(lst[n // 2:])))
        if len(pre) > len(post):
            pre, post = post, pre
        ans = 0
        n = len(post)
        j = n - 1
        for num in pre:
            while num + post[j] > w:
                j -= 1
            ans = max(ans, num + post[j])
        ac.st(ans)
        return

    @staticmethod
    def lg_p5691(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5691
        tag: meet_in_middle|sorted_list|two_pointers|brute_force
        """

        def check(lst):
            dct = defaultdict(int)
            dct[0] = 1
            for k, p in lst:
                cur = defaultdict(int)
                for pp in dct:
                    for x in range(1, m + 1):
                        cur[pp + k * x ** p] += dct[pp]
                dct = cur
            return dct

        n = ac.read_int()
        m = ac.read_int()
        pos = [ac.read_list_ints() for _ in range(n)]
        pre = check(pos[:n // 2])
        post = check(pos[n // 2:])
        ans = sum(pre[x] * post[-x] for x in pre)
        ac.st(ans)
        return

    @staticmethod
    def abc_326f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc326/tasks/abc326_f
        tag: meet_in_middle|brain_teaser|classical
        """
        n, x, y = ac.read_list_ints()
        a = ac.read_list_ints()

        def check(lst):
            k = len(lst)
            res = {0: 0}
            for j in range(k):
                cur = dict()
                cur.update({num + a[lst[j]]: res[num] | (1 << lst[j]) for num in res})
                cur.update({num - a[lst[j]]: res[num] for num in res})
                res = cur
            return res

        axis = list(range(1, n, 2))
        m = len(axis)
        pre = axis[:m // 2]
        post = axis[m // 2:]
        res1 = check(pre)
        res2 = check(post)
        state = 0
        for xx in res1:
            if x - xx in res2:
                state |= res1[xx] | res2[x - xx]
                break
        else:
            ac.no()
            return

        axis = list(range(0, n, 2))
        m = len(axis)
        pre = axis[:m // 2]
        post = axis[m // 2:]
        res1 = check(pre)
        res2 = check(post)
        for yy in res1:
            if y - yy in res2:
                state |= res1[yy] | res2[y - yy]
                break
        else:
            ac.no()
            return

        ac.yes()
        ans = ""
        pre = 1
        for i in range(n):
            if i % 2 == 0:
                if (state >> i) & 1 == pre:
                    ans += "L"
                else:
                    ans += "R"
            else:
                if (state >> i) & 1 == pre:
                    ans += "R"
                else:
                    ans += "L"
            pre = (state >> i) & 1
        ac.st(ans)
        return

    @staticmethod
    def abc_271f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc271/tasks/abc271_f
        tag: meet_in_middle|brute_force|classical
        """
        n = ac.read_int()
        grid = [ac.read_list_ints() for _ in range(n)]
        left = [[Counter() for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j == 0:
                    left[i][j] = Counter([grid[i][j]])
                    continue
                num = grid[i][j]
                if i + j <= n - 1:
                    cur = Counter()
                    if i:
                        pre = left[i - 1][j]
                        for p in pre:
                            cur[p ^ num] += pre[p]
                        left[i - 1][j] = Counter()
                    if j:
                        pre = left[i][j - 1]
                        for p in pre:
                            cur[p ^ num] += pre[p]
                    left[i][j] = cur
                else:
                    break

        right = [[Counter() for _ in range(n)] for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if i == j == n - 1:
                    right[i][j] = Counter([grid[i][j]])
                    continue
                num = grid[i][j]
                if i + j >= n - 1:
                    cur = Counter()
                    if i + 1 < n:
                        pre = right[i + 1][j]
                        for p in pre:
                            cur[p ^ num] += pre[p]
                        right[i + 1][j] = Counter()
                    if j + 1 < n:
                        pre = right[i][j + 1]
                        for p in pre:
                            cur[p ^ num] += pre[p]
                    right[i][j] = cur
                else:
                    break
        ans = 0
        for i in range(n):
            pre = left[i][n - 1 - i]
            x = grid[i][n - 1 - i]
            nex = right[i][n - 1 - i]
            for p in pre:
                ans += nex[p ^ x] * pre[p]
        ac.st(ans)
        return

    @staticmethod
    def cf_888e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/888/E
        tag: meet_in_middle|classical
        """
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        random.shuffle(nums)
        pre = {0}
        for num in nums[:n // 2]:
            pre = pre | {(p + num) % m for p in pre} | {num % m}
        post = {0}
        for num in nums[n // 2:]:
            post = post | {(p + num) % m for p in post} | {num % m}
        pre = sorted(pre)
        ans = pre[-1]
        for num in post:
            ans = max(ans, (pre[-1] + num) % m)
            ans = max(ans, num)
            i = bisect.bisect_left(pre, m - num) - 1
            if i >= 0:
                ans = max(ans, (pre[i] + num) % m)
        ac.st(ans)
        return