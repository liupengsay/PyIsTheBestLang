"""
Algorithm：sort|bubble_sort|merge_sort(minimum_num_comparisons)|quick_sort(expected_fast)|custom_sort
Description：reverse_order_pair

====================================LeetCode====================================
164（https://leetcode.cn/problems/maximum-gap/）bucket_sort
179（https://leetcode.cn/problems/largest-number/）custom_sort|maximum
912（https://leetcode.cn/problems/sort-an-array/）quick_sort
1585（https://leetcode.cn/problems/check-if-string-is-transformable-with-substring-sort-operations/）bubble_sort|implemention
45（https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/）custom_sort|minimum
2412（https://leetcode.cn/problems/minimum-money-required-before-transactions/）custom_sort|greedy
1665（https://leetcode.cn/problems/minimum-initial-energy-to-finish-tasks/）custom_sort|greedy|sorting
100391（https://leetcode.cn/problems/minimum-amount-of-damage-dealt-to-bob/）greedy|custom_sort|classical

=====================================LuoGu======================================
P2310（https://www.luogu.com.cn/problem/P2310）sorting
P4378（https://www.luogu.com.cn/problem/P4378）brute_force|bubble_sort
P6243（https://www.luogu.com.cn/problem/P6243）greedy|custom_sort
P1774（https://www.luogu.com.cn/problem/P1774）merge_sort|reverse_order_pair
P1908（https://www.luogu.com.cn/problem/P1908）tree_array|reverse_order_pair
P1177（https://www.luogu.com.cn/problem/P1177）quick_sort

===================================CodeForces===================================
922D（https://codeforces.com/problemset/problem/922/D）greedy|custom_sort
1983D（https://codeforces.com/problemset/problem/1983/D）inverse_pair|observation|brain_teaser
632C（https://codeforces.com/problemset/problem/632/C）custom_sort|greedy|classical

====================================AtCoder=====================================
ABC042B（https://atcoder.jp/contests/abc042/tasks/abc042_b）custom_sort
ABC268F（https://atcoder.jp/contests/abc268/tasks/abc268_f）custom_sort|classical
ABC225F（https://atcoder.jp/contests/abc225/tasks/abc225_f）linear_dp|cmp_to_key|custom_sort|brain_teaser|reverse_order
ABC190F（https://atcoder.jp/contests/abc190/tasks/abc190_f）reverse_pair|various_sort|classical

=====================================AcWing=====================================
113（https://www.acwing.com/problem/content/description/115/）custom_sort

"""
import random
from functools import cmp_to_key
from typing import List

from src.basis.various_sort.template import VariousSort
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_912(nums: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/sort-an-array/
        tag: quick_sort
        """
        n = len(nums)
        stack = [(0, n - 1)]
        while stack:
            left, right = stack.pop()
            mid = nums[random.randint(left, right)]
            i, j = left, right
            while i <= j:
                while nums[i] < mid:
                    i += 1
                while nums[j] > mid:
                    j -= 1
                if i <= j:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
                    j -= 1
            if left < j:
                stack.append((left, j))
            if i < right:
                stack.append((i, right))
        return nums

    @staticmethod
    def ac_113(compare, n):
        """
        url: https://www.acwing.com/problem/content/description/115/
        tag: custom_sort
        """

        def compare_(x, y):
            if compare(x, y):
                return -1
            return 1

        nums = list(range(1, n + 1))
        nums.sort(key=cmp_to_key(compare_))
        return nums

    @staticmethod
    def abc_042b(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc042/tasks/abc042_b
        tag: custom_sort|classical
        """
        n, m = ac.read_list_ints()
        nums = [ac.read_str() for _ in range(n)]

        def compare(a, b):
            if a + b < b + a:
                return -1
            elif a + b > b + a:
                return 1
            return 0

        nums.sort(key=cmp_to_key(compare))
        ac.st("".join(nums))
        return

    @staticmethod
    def lc_179(nums: List[int]) -> str:
        """
        url: https://leetcode.cn/problems/largest-number/
        tag: custom_sort|maximum|classical
        """

        def compare(a, b):
            if int(a + b) > int(b + a):
                return -1
            elif int(a + b) < int(b + a):
                return 1
            return 0

        nums = [str(x) for x in nums]
        nums.sort(key=cmp_to_key(compare))
        return str(int("".join(nums)))

    @staticmethod
    def lg_1177(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1177
        tag: quick_sort
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        stack = [(0, n - 1)]
        while stack:
            left, right = stack.pop()
            mid = nums[random.randint(left, right)]
            i, j = left, right
            while i <= j:
                while nums[i] < mid:
                    i += 1
                while nums[j] > mid:
                    j -= 1
                if i <= j:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
                    j -= 1
            if left < j:
                stack.append((left, j))
            if i < right:
                stack.append((i, right))
        ac.lst(nums)
        return

    @staticmethod
    def lc_1665(tasks: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-initial-energy-to-finish-tasks/
        tag: custom_sort|greedy|classical
        """

        def compare(aa, bb):
            a1, m1 = aa
            a2, m2 = bb
            s12 = m1 if m1 > a1 + m2 else a1 + m2
            s21 = m2 if m2 > a2 + m1 else a2 + m1
            if s12 < s21:
                return -1
            elif s12 > s21:
                return 1
            return 0

        tasks.sort(key=cmp_to_key(compare))
        ans = cur = 0
        for a, m in tasks:
            if cur < m:
                ans += m - cur
                cur = m
            cur -= a
        return ans

    @staticmethod
    def lc_2412(transactions: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-money-required-before-transactions/
        tag: custom_sort|greedy|classical
        """

        def check(it):
            cos, cash = it[:]
            if cos > cash:
                return [-1, cash]
            return [1, -cos]

        transactions.sort(key=check)
        ans = cur = 0
        for a, b in transactions:
            if cur < a:
                ans += a - cur
                cur = a
            cur += b - a
        return ans

    @staticmethod
    def lg_p1908(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1908
        tag: tree_array|reverse_order_pair|P1774
        """
        ac.read_int()
        nums = ac.read_list_ints()
        ans = VariousSort().range_merge_to_disjoint_sort_inverse_pair(nums)
        ac.st(ans)
        return

    @staticmethod
    def abc_268f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc268/tasks/abc268_f
        tag: custom_sort|classical
        """

        def compare(i, j):
            cnt1 = cnt[i]
            cnt2 = cnt[j]
            s1 = cnt1[0] * cnt2[1]
            s2 = cnt2[0] * cnt1[1]
            if s1 > s2:
                return -1
            if s1 < s2:
                return 1
            return 0

        n = ac.read_int()
        words = [ac.read_str().replace("X", "0") for _ in range(n)]
        cnt = []
        score = []
        for word in words:
            cur = [0] * 2
            ss = cc = 0
            for w in word:
                if w == "0":
                    cur[0] += 1
                    cc += 1
                else:
                    cur[1] += int(w)
                    ss += cc * int(w)
            score.append(ss)
            cnt.append(cur[:])
        ind = list(range(n))
        ind.sort(key=cmp_to_key(compare))
        ans = post = 0
        for x in range(n - 1, -1, -1):
            ans += cnt[ind[x]][0] * post
            post += cnt[ind[x]][1]
        ac.st(ans + sum(score))
        return

    @staticmethod
    def abc_225f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc225/tasks/abc225_f
        tag: linear_dp|cmp_to_key|custom_sort|brain_teaser|reverse_order
        """
        n, k = ac.read_list_ints()

        def compare_(x, y):
            if x + y < y + x:
                return -1
            return 1

        words = [ac.read_str() for _ in range(n)]
        words.sort(key=cmp_to_key(compare_))

        dp = [chr(ord("z") + 1) for _ in range(k + 1)]
        dp[0] = ""
        for i in range(n - 1, -1, -1):
            word = words[i]
            ndp = dp[:]
            for j in range(1, k + 1):
                ndp[j] = min(ndp[j], word + dp[j - 1])
            dp = ndp[:]
        ac.st(dp[-1])
        return

    @staticmethod
    def lc_100391(power: int, damage: List[int], health: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-amount-of-damage-dealt-to-bob/
        tag: greedy|custom_sort|classical
        """
        n = len(damage)

        def compare(i, j):
            hi, di = health[i], damage[i]
            hj, dj = health[j], damage[j]
            s1 = dj * t[i]
            s2 = di * t[j]
            if s1 < s2:
                return -1
            if s1 > s2:
                return 1
            return 0

        t = [(h + power - 1) // power for h in health]
        ind = list(range(n))
        ind.sort(key=cmp_to_key(compare))

        ans = 0
        tot = sum(damage)
        for k in ind:
            ans += tot * ((health[k] + power - 1) // power)
            tot -= damage[k]
        return ans

    @staticmethod
    def cf_1983d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1983/D
        tag: inverse_pair|observation|brain_teaser
        """
        for _ in range(ac.read_int()):
            ac.read_int()
            a = ac.read_list_ints()
            b = ac.read_list_ints()
            if sorted(a) != sorted(b) or VariousSort().range_merge_to_disjoint_sort_inverse_pair(
                    a) % 2 != VariousSort().range_merge_to_disjoint_sort_inverse_pair(b) % 2:
                ac.no()
                continue
            ac.yes()
        return

    @staticmethod
    def cf_632c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/632/C
        tag: custom_sort|greedy|classical
        """
        n = ac.read_int()

        words = [ac.read_str() for _ in range(n)]

        def compare(wi, wj):
            if wi + wj < wj + wi:
                return -1
            return 1

        words.sort(key=cmp_to_key(compare))

        ans = "".join(words)
        ac.st(ans)
        return

    @staticmethod
    def abc_190f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc190/tasks/abc190_f
        tag: reverse_pair|various_sort|classical
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        ans = VariousSort().range_merge_to_disjoint_sort_inverse_pair(nums[:])
        ac.st(ans)
        for i in range(n - 1):
            x = nums[i]
            ans -= x
            ans += n - x - 1
            ac.st(ans)
        return
