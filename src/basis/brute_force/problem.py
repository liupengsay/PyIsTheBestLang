"""
Algorithm：brute_force|matrix_rotate|matrix_spiral|contribution_method
Description：brute force according to the data range

====================================LeetCode====================================
670（https://leetcode.cn/problems/maximum-swap/）greedy|brute_force
395（https://leetcode.cn/problems/longest-substring-with-at-least-k-repeating-characters/）brute_force|divide_and_conquer
1330（https://leetcode.cn/problems/reverse-subarray-to-maximize-array-value/）brute_force
2488（https://leetcode.cn/problems/count-subarrays-with-median-k/）median|brute_force|pre_consequence|post_consequence
2484（https://leetcode.cn/problems/count-palindromic-subsequences/）prefix_suffix|hash|brute_force|palindrome_substring
2322（https://leetcode.cn/problems/minimum-score-after-removals-on-a-tree/）brute_force|tree_dp|unionfind|xor_min
2321（https://leetcode.cn/problems/maximum-score-of-spliced-array/）brute_force
2306（https://leetcode.cn/problems/naming-a-company/）alphabet|brute_force
2272（https://leetcode.cn/problems/substring-with-largest-variance/）alphabet|brute_force
2183（https://leetcode.cn/problems/count-array-pairs-divisible-by-k/）gcd|brute_force
2151（https://leetcode.cn/problems/maximum-good-people-based-on-statements/）state_compression|brute_force
2147（https://leetcode.cn/problems/number-of-ways-to-divide-a-long-corridor/）brute_force|counter
2122（https://leetcode.cn/problems/recover-the-original-array/）brute_force
2468（https://leetcode.cn/problems/split-message-based-on-limit/）binary_search
2417（https://leetcode.cn/problems/closest-fair-integer/）digit|greedy|brute_force
2681（https://leetcode.cn/problems/power-of-heroes/）contribution_method|brute_force|counter
1625（https://leetcode.cn/problems/lexicographically-smallest-string-after-applying-operations/）brute_force|lexicographical_order
1819（https://leetcode.cn/problems/number-of-different-subsequences-gcds/）harmonic_progression|brute_force
1862（https://leetcode.cn/submissions/detail/371754298/）brute_force|harmonic_progression
2014（https://leetcode.cn/problems/longest-subsequence-repeated-k-times/）data_range|brute_force|greedy|permutations
2077（https://leetcode.cn/problems/paths-in-maze-that-lead-to-same-room/）bit_operation|brute_force
2081（https://leetcode.cn/problems/sum-of-k-mirror-numbers/）palindrome_num|base_n|brute_force
2170（https://leetcode.cn/problems/minimum-operations-to-make-the-array-alternating/）brute_force|secondary_maximum
1215（https://leetcode.cn/problems/stepping-numbers/）data_range|back_track|brute_force
2245（https://leetcode.cn/problems/maximum-trailing-zeros-in-a-cornered-path/）prefix_sum|brute_force
1878（https://leetcode.cn/problems/get-biggest-three-rhombus-sums-in-a-grid/）prefix_sum与|brute_force
2018（https://leetcode.cn/problems/check-if-word-can-be-placed-in-crossword/description/）brute_force
2591（https://leetcode.cn/problems/distribute-money-to-maximum-children/）brute_force
910（https://leetcode.cn/problems/smallest-range-ii/description/）brute_force|data_range
1131（https://leetcode.cn/problems/maximum-of-absolute-value-expression/description/）manhattan_distance|brute_force
1761（https://leetcode.cn/problems/minimum-degree-of-a-connected-trio-in-a-graph/）directed_graph|undirected_graph|brute_force
1178（https://leetcode.cn/problems/number-of-valid-words-for-each-puzzle/）hash|counter|brute_force|bit_operation
1638（https://leetcode.cn/problems/count-substrings-that-differ-by-one-character/description/）brute_force|dp|brute_force
2212（https://leetcode.cn/problems/maximum-points-in-an-archery-competition/）bit_operation|brute_force|back_track
2749（https://leetcode.cn/problems/minimum-operations-to-make-the-integer-zero/）brute_force|bit_operation
2094（https://leetcode.cn/problems/finding-3-digit-even-numbers/description/）brain_teaser|brute_force|data_range
842（https://leetcode.cn/problems/split-array-into-fibonacci-sequence/description/）brain_teaser|brute_force|back_track
2122（https://leetcode.cn/problems/recover-the-original-array/）brute_force|hash|implemention
1782（https://leetcode.cn/problems/count-pairs-of-nodes/description/）brute_force

=====================================LuoGu======================================
P1548（https://www.luogu.com.cn/problem/P1548）brute_force
P1632（https://www.luogu.com.cn/problem/P1632）brute_force
P2128（https://www.luogu.com.cn/problem/P2128）brute_force
P2191（https://www.luogu.com.cn/problem/P2191）reverse_thinking|matrix_rotate
P2699（https://www.luogu.com.cn/problem/P2699）classification_discussion|brute_force|implemention
P1371（https://www.luogu.com.cn/problem/P1371）prefix_suffix|brute_force|counter
P1369（https://www.luogu.com.cn/problem/P1369）matrix_dp|greedy|brute_force
P1158（https://www.luogu.com.cn/problem/P1158）sorting|brute_force|suffix_maximum
P8928（https://www.luogu.com.cn/problem/P8928）brute_force|counter
P8892（https://www.luogu.com.cn/problem/P8892）brute_force
P8799（https://www.luogu.com.cn/problem/P8799）brute_force
P3142（https://www.luogu.com.cn/problem/P3142）brute_force
P3143（https://www.luogu.com.cn/problem/P3143）brute_force|prefix_suffix
P3670（https://www.luogu.com.cn/problem/P3670）hash|brute_force|counter
P3799（https://www.luogu.com.cn/problem/P3799）brute_force|regular_triangle
P3910（https://www.luogu.com.cn/problem/P3910）factorization|brute_force
P4086（https://www.luogu.com.cn/problem/P4086）suffix|reverse_order|brute_force
P4596（https://www.luogu.com.cn/problem/P4596）brute_force
P4759（https://www.luogu.com.cn/problem/P4759）factorization|brute_force
P6267（https://www.luogu.com.cn/problem/P6267）factorization|brute_force
P5077（https://www.luogu.com.cn/problem/P5077）factorization|brute_force可
P4960（https://www.luogu.com.cn/problem/P4960）implemention|brute_force
P4994（https://www.luogu.com.cn/problem/P4994）implemention|pi(n)<=6n
P5190（https://www.luogu.com.cn/problem/P5190）counter|prefix_sum|harmonic_progression|O(nlogn)
P5614（https://www.luogu.com.cn/problem/P5614）brute_force
P6014（https://www.luogu.com.cn/problem/P6014）hash|mod|counter
P6067（https://www.luogu.com.cn/problem/P6067）sorting|prefix_suffix|brute_force
P6248（https://www.luogu.com.cn/problem/P6248）brute_force
P6355（https://www.luogu.com.cn/problem/P6355）brute_force|triangle|counter
P6365（https://www.luogu.com.cn/problem/P6365）inclusion_exclusion|brute_force|counter
P6439（https://www.luogu.com.cn/problem/P6439）brute_force
P6686（https://www.luogu.com.cn/problem/P6686）brute_force|triangle|counter
P2666（https://www.luogu.com.cn/problem/P2666）brute_force|counter
P2705（https://www.luogu.com.cn/problem/P2705）brute_force
P5690（https://www.luogu.com.cn/problem/P5690）brute_force
P7076（https://www.luogu.com.cn/problem/P7076）bit_operation|brute_force|counter
P7094（https://www.luogu.com.cn/problem/P7094）math|data_range|brute_force
P7273（https://www.luogu.com.cn/problem/P7273）brute_force|math|greedy
P7286（https://www.luogu.com.cn/problem/P7286）sorting|brute_force|counter
P7626（https://www.luogu.com.cn/problem/P7626）brute_force|matrix|diagonal
P7799（https://www.luogu.com.cn/problem/P7799）hash|brute_force|counter
P1018（https://www.luogu.com.cn/problem/P1018）brute_force
P1311（https://www.luogu.com.cn/problem/P1311）brute_force|counter
P2119（https://www.luogu.com.cn/problem/P2119）brute_force|prefix_suffix|counter
P2652（https://www.luogu.com.cn/problem/P2652）brute_force|two_pointers
P2994（https://www.luogu.com.cn/problem/P2994）brute_force
P3985（https://www.luogu.com.cn/problem/P3985）brute_force
P4181（https://www.luogu.com.cn/problem/P4181）greedy|brute_force|suffix_sum
P6149（https://www.luogu.com.cn/problem/P6149）brute_force|triangle|prefix_sum|binary_search
P6393（https://www.luogu.com.cn/problem/P6393）data_range|brute_force
P6767（https://www.luogu.com.cn/problem/P6767）brute_force
P8270（https://www.luogu.com.cn/problem/P8270）brain_teaser|brute_force
P8587（https://www.luogu.com.cn/problem/P8587）bucket_counter|brute_force
P8663（https://www.luogu.com.cn/problem/P8663）bucket_counter|brute_force
P8672（https://www.luogu.com.cn/problem/P8672）string|brute_force|permutation_circle|counter
P8712（https://www.luogu.com.cn/problem/P8712）brute_force
P8749（https://www.luogu.com.cn/problem/P8749）yanghui_triangle|brute_force
P8808（https://www.luogu.com.cn/problem/P8808）fibonacci|brute_force
P8809（https://www.luogu.com.cn/problem/P8809）brute_force|contribution_method|counter
P9076（https://www.luogu.com.cn/problem/P9076）factorization|brute_force
P9008（https://www.luogu.com.cn/problem/P9008）inclusion_exclusion|brute_force|counter
P9006（https://www.luogu.com.cn/problem/P9006）brute_force|mod|counter
P8948（https://www.luogu.com.cn/problem/P8948）brute_force
P8894（https://www.luogu.com.cn/problem/P8894）data_range|brute_force|prefix_suffix|counter
P8872（https://www.luogu.com.cn/problem/P8872）sorting|prefix_suffix|brute_force

===================================CodeForces===================================
1426F（https://codeforces.com/problemset/problem/1426/F）classification_discussion|brute_force|counter|fast_power
1400D（https://codeforces.com/problemset/problem/1400/D）brute_force|binary_search
1793D（https://codeforces.com/contest/1793/problem/D）brute_force|counter
584D（https://codeforces.com/problemset/problem/584/D）brute_force|prime|decompose_into_sum_of_prime_at_most_3
1311D（https://codeforces.com/problemset/problem/1311/D）greedy|brute_force
1181C（https://codeforces.com/problemset/problem/1181/C）column_wised|brute_force
484B（https://codeforces.com/problemset/problem/484/B）sorting|brute_force|binary_search
382C（https://codeforces.com/problemset/problem/382/C）classification_discussion
D - Remainder Reminder（https://atcoder.jp/contests/abc090/tasks/arc091_b）brute_force
D - Katana Thrower（https://atcoder.jp/contests/abc085/tasks/abc085_d）brute_force
988E（https://codeforces.com/contest/988/problem/E）brain_teaser|greedy|brute_force
1661B（https://codeforces.com/contest/1661/problem/B）brute_force

====================================AtCoder=====================================
D - Digit Sum（https://atcoder.jp/contests/abc044/tasks/arc060_b）base|classification_discussion|brute_force|factorization
D - Menagerie （https://atcoder.jp/contests/abc055/tasks/arc069_b）brain_teaser|brute_force
C - Sequence（https://atcoder.jp/contests/abc059/tasks/arc072_a）brute_force|prefix_sum|greedy
C - Chocolate Bar（https://atcoder.jp/contests/abc062/tasks/arc074_a）brute_force
C - Sugar Water（https://atcoder.jp/contests/abc074/tasks/arc083_a）brute_force|math

================================Acwing===================================
95（https://www.acwing.com/problem/content/description/97/）brute_force

"""
import bisect
import math
from collections import defaultdict, deque
from functools import reduce, lru_cache
from itertools import combinations, permutations
from math import inf
from operator import mul, or_
from typing import List

from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1311d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1311/D
        tag: greedy|brute_force
        """
        # 根据greedy策略 a=b=1 时显然满足条件，因此brute_force不会超过这个代价的范围就行
        for _ in range(ac.read_int()):
            a, b, c = ac.read_list_ints()
            ans = inf
            res = []
            for x in range(1, 2 * a + 1):
                for y in range(x, 2 * b + 1, x):
                    if y % x == 0:
                        for z in [(c // y) * y, (c // y) * y + y]:
                            cost = abs(a - x) + abs(b - y) + abs(c - z)
                            if cost < ans:
                                ans = cost
                                res = [x, y, z]
            ac.st(ans)
            ac.lst(res)
        return

    @staticmethod
    def cf_584d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/584/D
        tag: brute_force|prime|decompose_into_sum_of_prime_at_most_3
        """
        # 将 n 分解为最多三个质数的和
        def is_prime4(x):
            if x == 1:
                return False
            if (x == 2) or (x == 3):
                return True
            if (x % 6 != 1) and (x % 6 != 5):
                return False
            for ii in range(5, int(math.sqrt(x)) + 1, 6):
                if (x % ii == 0) or (x % (ii + 2) == 0):
                    return False
            return True

        # 将正整数分解为最多三个质数的和
        n = ac.read_int()
        assert 3 <= n < 10 ** 9

        if is_prime4(n):
            ac.st(1)
            ac.st(n)
            return

        for i in range(2, 10 ** 5):
            j = n - 3 - i
            if is_prime4(i) and is_prime4(j):
                ac.st(3)
                ac.lst([3, i, j])
                return
        return

    @staticmethod
    def lc_670(num: int) -> int:
        """
        url: https://leetcode.cn/problems/maximum-swap/
        tag: greedy|brute_force
        """
        # 在复杂度有限的情况下有限采用brute_force的方式而不是greedy

        def check():  # greedy
            lst = list(str(num))
            n = len(lst)
            post = list(range(n))
            # 从后往前遍历，对每个数位，记录其往后最大且最靠后的比它大的数位位置，再从前往后交换第一个有更大的靠后值得数位
            j = n - 1
            for i in range(n - 2, -1, -1):
                if lst[i] > lst[j]:
                    j = i
                if lst[j] > lst[i]:
                    post[i] = j

            for i in range(n):
                if post[i] != i:
                    lst[i], lst[post[i]] = lst[post[i]], lst[i]
                    return int("".join(lst))
            return int("".join(lst))

        def check2():  # brute_force
            lst = list(str(num))
            n = len(lst)
            ans = num
            for item in combinations(list(range(n)), 2):
                cur = lst[:]
                i, j = item
                cur[i], cur[j] = cur[j], cur[i]
                x = int("".join(cur))
                ans = ans if ans > x else x
            return ans

        check()
        return check2()

    @staticmethod
    def cf_484b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/484/B
        tag: sorting|brute_force|binary_search
        """
        # 查询数组中两两mod|运算的最大值（要求较小值作为mod|数）
        ac.read_int()
        nums = sorted(list(set(ac.read_list_ints())))
        n = len(nums)
        ceil = nums[-1]

        dp = [0] * (ceil + 1)
        i = 0
        for x in range(1, ceil + 1):
            dp[x] = dp[x - 1]
            while i < n and nums[i] <= x:
                dp[x] = nums[i]
                i += 1

        ans = 0
        for num in nums:
            x = 1
            while x * num <= ceil:
                x += 1
                for a in [x * num - 1]:
                    ans = ac.max(ans, dp[ac.min(a, ceil)] % num)
        ac.st(ans)
        return

    @staticmethod
    def cf_382c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/382/C
        tag: classification_discussion
        """

        # 2023年3月29日·灵茶试炼·classification_discussion
        n = ac.read_int()
        nums = sorted(ac.read_list_ints())

        # 只有一种情况有无穷多个
        if n == 1:
            ac.st(-1)
            return

        # sorting后相邻项差值最大值与最小值以及不同差值
        diff = [nums[i] - nums[i - 1] for i in range(1, n)]
        high = max(diff)
        low = min(diff)
        cnt = len(set(diff))

        # 1. 大于等于3个不同差值显然没有
        if cnt >= 3:
            ac.st(0)
            return
        elif cnt == 2:
            # 2. 有2个不同差值存在合理情况当且仅当 high=2*low 且 count(high)==1
            if high != 2 * low or diff.count(high) != 1:
                ac.st(0)
                return

            for i in range(1, n):
                if nums[i] - nums[i - 1] == high:
                    ac.st(1)
                    ac.st(nums[i - 1] + low)
                    return
        else:
            # 3.有1个差值时分为0与不为0，不为0分 n大于2 与等于2
            if low == high == 0:
                ac.st(1)
                ac.st(nums[0])
                return
            if n == 2:
                if low % 2 == 0:
                    ac.st(3)
                    ac.lst([nums[0] - low, nums[0] + low // 2, nums[1] + low])
                else:
                    ac.st(2)
                    ac.lst([nums[0] - low, nums[1] + low])
            else:
                ac.st(2)
                ac.lst([nums[0] - low, nums[-1] + low])
        return

    @staticmethod
    def abc_44d(ac=FastIO()):
        # 进制与分情况brute_force因子
        def check():
            lst = []
            num = n
            while num:
                lst.append(num % b)
                num //= b
            return sum(lst) == s

        n = ac.read_int()
        s = ac.read_int()
        if s > n:
            ac.st(-1)
        elif s == n:
            ac.st(n + 1)
        else:
            # (n-s) % (b-1) == 0
            ans = inf
            for x in range(1, n - s + 1):
                if x * x > n - s:
                    break
                if (n - s) % x == 0:
                    # brute_force b-1 的值为 n-s 的因子
                    y = (n - s) // x
                    b = x + 1
                    if check():
                        ans = b if ans > b else ans
                    b = y + 1
                    if check():
                        ans = b if ans > b else ans
            ac.st(-1 if ans == inf else ans)
        return

    @staticmethod
    def abc_59c(ac=FastIO()):
        # brute_forceprefix_sum的符号greedy增减
        n = ac.read_int()
        nums = ac.read_list_ints()
        ans1 = 0
        pre = 0
        for i in range(n):
            pre += nums[i]
            if i % 2 == 0:
                if pre <= 0:
                    ans1 += 1 - pre
                    pre = 1
            else:
                if pre >= 0:
                    ans1 += pre + 1
                    pre = -1
        ans2 = 0
        pre = 0
        for i in range(n):
            pre += nums[i]
            if i % 2 == 1:
                if pre <= 0:
                    ans2 += 1 - pre
                    pre = 1
            else:
                if pre >= 0:
                    ans2 += pre + 1
                    pre = -1
        ac.st(ac.min(ans1, ans2))
        return

    @staticmethod
    def abc_62c(ac=FastIO()):
        # brute_force切割方式
        m, n = ac.read_list_ints()

        def check1():
            nonlocal ans
            for x in range(1, m):
                lst = [x * n, (m - x) * (n // 2), (m - x) * (n // 2 + n % 2)]
                cur = max(lst) - min(lst)
                if cur < ans:
                    ans = cur
            return

        def check2():
            nonlocal ans
            for x in range(1, m - 1):
                lst = [x * n, ((m - x) // 2) * n, ((m - x) // 2 + (m - x) % 2) * n]
                cur = max(lst) - min(lst)
                if cur < ans:
                    ans = cur
            return

        ans = inf
        check1()
        check2()
        m, n = n, m
        check1()
        check2()
        ac.st(ans)
        return

    @staticmethod
    def abc_74c(ac=FastIO()):
        # brute_force系数利用公式边界
        res = 0
        a, b, c, d, e, f = ac.read_list_ints()
        ans = [100 * a, 0]
        for p in range(3001):
            if p * a * 100 > f:
                break
            for q in range(3001):
                if p * a * 100 + q * b * 100 > f:
                    break
                if p == q == 0:
                    continue
                ceil = (p * a + q * b) * e

                for x in range(3001):
                    if x * c > ceil:
                        break
                    y1 = (ceil - x * c) // d
                    y2 = (f - p * a * 100 - q * b * 100 - x * c) // d
                    y1 = y1 if y1 < y2 else y2
                    if y1 < 0:
                        continue
                    y = y1
                    percent = 100 * (x * c + y * d) / (p * a * 100 + q * b * 100 + x * c + y * d)
                    if percent > res:
                        res = percent
                        ans = [p * a * 100 + q * b * 100 + x * c + y * d, x * c + y * d]
        ac.lst(ans)
        return

    @staticmethod
    def ac_95(ac=FastIO()):
        # brute_force第一行状态
        n = ac.read_int()

        for _ in range(n):
            grid = [[int(w) for w in ac.read_str()] for _ in range(5)]
            ac.read_str()

            ans = -1
            for state in range(1 << 5):
                lst = [x for x in range(5) if state & (1 << x)]
                temp = [g[:] for g in grid]
                cur = len(lst)
                for x in lst:
                    i, j = 0, x
                    temp[i][j] = 1 - temp[i][j]
                    for a, b in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                        if 0 <= a < 5 and 0 <= b < 5:
                            temp[a][b] = 1 - temp[a][b]

                for r in range(1, 5):
                    for j in range(5):
                        if temp[r - 1][j] == 0:
                            i, j = r, j
                            temp[i][j] = 1 - temp[i][j]
                            cur += 1
                            for a, b in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                                if 0 <= a < 5 and 0 <= b < 5:
                                    temp[a][b] = 1 - temp[a][b]
                if all(all(x == 1 for x in g) for g in temp):
                    ans = ans if ans < cur and ans != -1 else cur
            ac.st(ans if ans <= 6 else -1)
        return

    @staticmethod
    def lg_p1018(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1018
        tag: brute_force
        """
        # brute_force乘号的位置
        n, k = ac.read_list_ints()
        nums = ac.read_list_str()

        ans = 0
        for item in combinations(list(range(1, n)), k):
            cur = nums[:]
            for i in item:
                cur[i] = "*" + cur[i]
            res = [int(w) for w in ("".join(cur)).split("*")]
            cur = reduce(mul, res)
            ans = ac.max(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1311(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1311
        tag: brute_force|counter
        """
        # 线性brute_forcecounter，每次重置避免重复counter
        n, k, p = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        cnt = [0] * k
        for i in range(n):
            cnt[nums[i][0]] += 1
        pre = [0] * k
        ans = 0
        for i in range(n):
            c = nums[i][0]
            pre[c] += 1
            if nums[i][1] <= p:
                for j in range(k):
                    if j != c:
                        ans += pre[j] * (cnt[j] - pre[j])
                    else:
                        ans += pre[j] - 1
                        ans += cnt[j] - pre[j]
                        ans += (pre[j] - 1) * (cnt[j] - pre[j])
                    cnt[j] -= pre[j]
                pre = [0] * k
        ac.st(ans)
        return

    @staticmethod
    def lg_p2119(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2119
        tag: brute_force|prefix_suffix|counter
        """

        # brute_force差值，并prefix_suffix个数
        n, m = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(m)]

        cnt = [0] * (n + 1)
        for num in nums:
            cnt[num] += 1

        aa = [0] * (n + 1)
        bb = [0] * (n + 1)
        cc = [0] * (n + 1)
        dd = [0] * (n + 1)

        # brute_forceb-a=x
        for x in range(1, n // 9 + 1):
            if 1 + 9 * x + 1 > n:
                break

            # 前缀abcounter
            pre_ab = [0] * (n + 1)
            for b in range(2 * x + 1, n + 1):
                pre_ab[b] = pre_ab[b - 1]
                pre_ab[b] += cnt[b] * cnt[b - 2 * x]

            # 作为cd
            for c in range(n - x, -1, -1):
                if c - 6 * x - 1 >= 1:
                    cc[c] += pre_ab[c - 6 * x - 1] * cnt[c + x]
                    dd[c + x] += pre_ab[c - 6 * x - 1] * cnt[c]
                else:
                    break

            # 后缀cd
            post_cd = [0] * (n + 2)
            for c in range(n - x, -1, -1):
                post_cd[c] = post_cd[c + 1]
                post_cd[c] += cnt[c] * cnt[c + x]

            # 作为abcounter
            for b in range(2 * x + 1, n + 1):
                if b + 6 * x + 1 <= n:
                    aa[b - 2 * x] += post_cd[b + 6 * x + 1] * cnt[b]
                    bb[b] += post_cd[b + 6 * x + 1] * cnt[b - 2 * x]
                else:
                    break

        for x in nums:
            ac.lst([aa[x], bb[x], cc[x], dd[x]])
        return

    @staticmethod
    def lg_p2652(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2652
        tag: brute_force|two_pointers
        """

        # brute_force花色与two_pointers长度
        n = ac.read_int()
        dct = defaultdict(set)
        for _ in range(n):
            a, b = ac.read_list_ints()
            dct[a].add(b)
        ans = n
        for a in dct:
            lst = sorted(list(dct[a]))
            m = len(lst)
            j = 0
            for i in range(m):
                while j < m and lst[j] - lst[i] <= n - 1:
                    j += 1
                ans = ac.min(ans, n - (j - i))
        ac.st(ans)
        return

    @staticmethod
    def lg_p2994(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2994
        tag: brute_force
        """

        # 按照座位brute_force分配人员
        def dis():
            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        n, m = ac.read_list_ints()
        cow = [ac.read_list_ints() for _ in range(n)]
        pos = [ac.read_list_ints() for _ in range(n)]
        visit = [0] * n
        for j in range(m):
            ceil = inf
            ind = 0
            x1, y1 = pos[j]
            for i in range(n):
                if visit[i]:
                    continue
                x2, y2 = cow[i]
                cur = dis()
                if cur < ceil:
                    ceil = cur
                    ind = i
            if ceil < inf:
                visit[ind] = 1
        ans = [i + 1 for i in range(n) if not visit[i]]
        if not ans:
            ac.st(0)
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lg_p4181(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4181
        tag: greedy|brute_force|suffix_sum
        """
        # greedybrute_force与后缀和
        n, m, r = ac.read_list_ints()
        cow = [ac.read_int() for _ in range(n)]
        cow.sort()
        nums1 = [ac.read_list_ints()[::-1] for _ in range(m)]
        nums1.sort(key=lambda it: -it[0])
        nums2 = [ac.read_int() for _ in range(r)]
        nums2.sort(reverse=True)
        # preprocess后缀和
        ind = 0
        post = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            cur = 0
            while ind < m and cow[i]:
                if nums1[ind][1] == 0:
                    ind += 1
                    continue
                x = ac.min(nums1[ind][1], cow[i])
                cow[i] -= x
                nums1[ind][1] -= x
                cur += nums1[ind][0] * x
            post[i] = post[i + 1] + cur
        # brute_force
        ans = post[0]
        pre = 0
        for i in range(ac.min(r, n)):
            pre += nums2[i]
            ans = ac.max(ans, pre + post[i + 1])
        ac.st(ans)
        return

    @staticmethod
    def lg_p6149(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6149
        tag: brute_force|triangle|prefix_sum|binary_search
        """
        # brute_force三角形的直角点prefix_sum与binary_search距离和
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        dct_x = defaultdict(list)
        dct_y = defaultdict(list)
        for x, y in nums:
            dct_x[x].append(y)
            dct_y[y].append(x)
        pre_x = defaultdict(list)
        for x in dct_x:
            dct_x[x].sort()
            pre_x[x] = ac.accumulate(dct_x[x])
        pre_y = defaultdict(list)
        for y in dct_y:
            dct_y[y].sort()
            pre_y[y] = ac.accumulate(dct_y[y])

        ans = 0
        mod = 10 ** 9 + 7
        for x, y in nums:
            # binary_search找到中间点 xi 两侧距离
            xi = bisect.bisect_left(dct_y[y], x)
            left_x = (xi + 1) * x - pre_y[y][xi + 1]
            right_x = pre_y[y][-1] - pre_y[y][xi + 1] - (len(dct_y[y]) - xi - 1) * x

            yi = bisect.bisect_left(dct_x[x], y)
            left_y = (yi + 1) * y - pre_x[x][yi + 1]
            right_y = pre_x[x][-1] - pre_x[x][yi + 1] - (len(dct_x[x]) - yi - 1) * y
            ans += (left_x + right_x) * (left_y + right_y)
            ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def lg_p6393(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6393
        tag: data_range|brute_force
        """
        # 利用data_range范围brute_force
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        dct = dict()
        for i in range(n):
            a, b = nums[i]
            if b not in dct:
                dct[b] = dict()
            if a not in dct[b]:
                dct[b][a] = deque()
            dct[b][a].append(i)
        for i in range(n):
            a, b = nums[i]
            ind = -2
            for bb in dct:
                if (b * b) % bb == 0:
                    # 寻找符合条件的最小值
                    aa = a + b * b // bb + b
                    if aa in dct[bb]:
                        while dct[bb][aa] and dct[bb][aa][0] <= i:
                            dct[bb][aa].popleft()
                        if dct[bb][aa]:
                            j = dct[bb][aa][0]
                            if ind == -2 or j < ind:
                                ind = j
                        else:
                            del dct[bb][aa]
                            if not dct[bb]:
                                del dct[bb]
            ac.st(ind + 1)
        return

    @staticmethod
    def lc_2591(money: int, children: int) -> int:
        """
        url: https://leetcode.cn/problems/distribute-money-to-maximum-children/
        tag: brute_force
        """
        # brute_force考虑边界条件
        ans = -1
        for x in range(children + 1):
            if x * 8 > money:
                break
            rest_money = money - x * 8
            rest_people = children - x
            if rest_money < rest_people:
                continue
            if not rest_people and rest_money:
                continue
            if rest_people == 1 and rest_money == 4:
                continue
            ans = x
        return ans

    @staticmethod
    def lc_2681(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/power-of-heroes/
        tag: contribution_method|brute_force|counter
        """
        # 按照contribution_methodbrute_forcecounter
        mod = 10 ** 9 + 7
        nums.sort()
        ans = pre = 0
        for num in nums:
            ans += num * num * pre
            ans += num * num * num
            pre %= mod
            ans %= mod
            pre *= 2
            pre += num
        return ans

    @staticmethod
    def lg_p6767(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6767
        tag: brute_force
        """
        # greedybrute_force性价比较低的数量
        n, a, b, c, d = ac.read_list_ints()
        if b * c > a * d:
            a, b, c, d = c, d, a, b

        ans = inf
        for x in range(10 ** 5 + 1):
            cur = x * d + b * ac.max(math.ceil((n - x * c) / a), 0)
            ans = ac.min(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p8270(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8270
        tag: brain_teaser|brute_force
        """
        # brain_teaserbrute_force，转换为两两字母比较
        s = ac.read_str()
        t = ac.read_str()
        lst = sorted(list("abcdefghijklmnopqr"))
        m = len(lst)
        pre = set()
        for i in range(m):
            for j in range(i, m):
                cur = {lst[i], lst[j]}
                ss = ""
                tt = ""
                for w in s:
                    if w in cur:
                        ss += w
                for w in t:
                    if w in cur:
                        tt += w
                if ss == tt:
                    pre.add(lst[i] + lst[j])
                    pre.add(lst[j] + lst[i])
        ans = ""
        for _ in range(ac.read_int()):
            st = ac.read_str()
            m = len(st)
            flag = True
            for i in range(m):
                for j in range(i, m):
                    if st[i] + st[j] not in pre:
                        flag = False
                        break
                if not flag:
                    break
            ans += "Y" if flag else "N"
        ac.st(ans)
        return

    @staticmethod
    def lg_p8672(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8672
        tag: string|brute_force|permutation_circle|counter
        """
        # 字符串brute_force与permutation_circle|counter
        s = ac.read_str()
        n = len(s)
        dct = dict()
        dct["B"] = s.count("B")
        dct["A"] = s.count("A")
        dct["T"] = s.count("T")
        ans = inf
        for item in permutations("BAT", 3):
            t = ""
            for w in item:
                t += dct[w] * w
            cnt = defaultdict(int)
            for i in range(n):
                if s[i] != t[i]:
                    cnt[s[i] + t[i]] += 1
            cur = 0
            for w in item:
                for p in item:
                    if w != p:
                        x = ac.min(cnt[w + p], cnt[p + w])
                        cur += x
                        cnt[w + p] -= x
                        cnt[p + w] -= x
            rest = sum(cnt.values())
            cur += rest * 2 // 3
            ans = ac.min(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p9076(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P9076
        tag: factorization|brute_force
        """
        # 根据数字的因数brute_force
        n = ac.read_int()
        ans = 0
        pre = set()
        for a in range(1, int(n ** 0.5) + 1):
            if n % a == 0:
                for bc in [n // a - 1, a - 1]:
                    if bc in pre:
                        continue
                    pre.add(bc)
                    for x in range(2, bc + 1):
                        if bc % x == 0:
                            y = bc // x - 1
                            if y > 1:
                                ans += 1
                        if bc // x <= 2:
                            break
        ac.st(ans)
        return

    @staticmethod
    def lg_p9008(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P9008
        tag: inclusion_exclusion|brute_force|counter
        """
        # 朋友敌人陌生人容斥brute_forcecounter
        n, p, q = ac.read_list_ints()
        friend = defaultdict(set)
        for _ in range(p):
            u, v = ac.read_list_ints()
            friend[u].add(v)
            friend[v].add(u)
        ans = n * (n - 1) // 2
        rem = set()
        for _ in range(q):
            u, v = ac.read_list_ints()
            rem.add((u, v) if u < v else (v, u))
            for x in friend[u]:
                if x not in friend[v]:
                    rem.add((x, v) if x < v else (v, x))
            for y in friend[v]:
                if y not in friend[u]:
                    rem.add((y, u) if y < u else (u, y))
        ac.st(ans - len(rem))
        return

    @staticmethod
    def lg_p9006(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P9006
        tag: brute_force|mod|counter
        """
        # brute_forcemod|counter
        mod = 100000007
        n, k = ac.read_list_ints()
        num = 9 * 10 ** (n - 1)
        x = num // k
        x %= mod
        ans = [x] * k
        for y in range(10 ** (n - 1) + x * k, 10 ** (n - 1) + x * k + num % k):
            ans[y % k] += 1
        ac.lst([x % mod for x in ans])
        return

    @staticmethod
    def lg_p8948(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8948
        tag: brute_force
        """
        # preprocess和brute_force所有情况
        dct = dict()
        dct[2000] = [400, 600]
        for i in range(401):
            for j in range(601):
                x = (3 * i + 2 * j) * 10 / 12
                x = int(x) + int(x - int(x) >= 0.5)
                if 10 <= x <= 1990:
                    dct[x] = [i, j]
        for _ in range(ac.read_int()):
            ac.lst(dct[ac.read_int()])
        return

    @staticmethod
    def lg_p8894(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8894
        tag: data_range|brute_force|prefix_suffix|counter
        """
        # 按照区间范围值brute_forceprefix_suffixcounter
        n = ac.read_int()
        mod = 998244353
        nums = [ac.read_list_ints() for _ in range(n)]
        ceil = max(q for _, q in nums)
        low = min(p for p, _ in nums)
        ans = 0
        for s in range(low, ceil + 1):
            pre = [0] * (n + 1)
            pre[0] = 1
            for i in range(n):
                p, q = nums[i]
                if p > s:
                    pre[i + 1] = 0
                    break
                else:
                    pre[i + 1] = pre[i] * (ac.min(s, q) - p + 1) % mod

            post = [0] * (n + 1)
            post[n] = 1
            for i in range(n - 1, -1, -1):
                p, q = nums[i]
                if p >= s:
                    post[i] = 0
                    break
                else:
                    post[i] = post[i + 1] * (ac.min(q, s - 1) - p + 1) % mod
            for i in range(n):
                p, q = nums[i]
                if p <= s <= q:
                    ans += pre[i] * post[i + 1] * s
                    ans %= mod
                if pre[i + 1] == 0:
                    break
        ac.st(ans)
        return

    @staticmethod
    def lg_p8872(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8872
        tag: sorting|prefix_suffix|brute_force
        """
        # sorting后prefix_suffix移动次数brute_force
        n, m = ac.read_list_ints()
        nums = sorted(ac.read_list_ints())
        ans = inf
        for i in range(n):
            if i > m:
                break
            right = (m - i) // 2
            if right >= n - i - 1:
                ac.st(0)
                return
            cur = nums[-right - 1] - nums[i]
            ans = ac.min(ans, cur)

        for i in range(n - 1, -1, -1):
            if n - i - 1 > m:
                break
            left = (m - n + i + 1) // 2
            if left >= i:
                ac.st(0)
                return
            cur = nums[i] - nums[left]
            ans = ac.min(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lc_2018(board: List[List[str]], word: str) -> bool:
        """
        url: https://leetcode.cn/problems/check-if-word-can-be-placed-in-crossword/description/
        tag: brute_force
        """
        # brute_force空挡位置与矩阵行列取数
        k = len(word)

        def check(cur):
            if len(cur) != len(word):
                return False
            return all(cur[i] == " " or cur[i] == word[i] for i in range(k))

        def compute(lst):
            length = len(lst)
            pre = 0
            for i in range(length):
                if lst[i] == "#":
                    if check([lst[x] for x in range(pre, i)]):
                        return True
                    pre = i + 1
            if check([lst[x] for x in range(pre, length)]):
                return True
            return False

        for tmp in board:
            if compute(tmp[:]) or compute(tmp[::-1]):
                return True

        for tmp in zip(*board):
            if compute(tmp[:]) or compute(tmp[::-1]):
                return True
        return False

    @staticmethod
    def lc_2170(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-operations-to-make-the-array-alternating/
        tag: brute_force|secondary_maximum
        """
        # brute_force，运用最大值与次大值技巧
        odd = defaultdict(int)
        even = defaultdict(int)
        n = len(nums)
        odd_cnt = 0
        even_cnt = 0
        for i in range(n):
            if i % 2 == 0:
                even[nums[i]] += 1
                even_cnt += 1
            else:
                odd[nums[i]] += 1
                odd_cnt += 1

        # 最大值与次大值
        a = b = 0
        for num in even:
            if even[num] >= a:
                a, b = even[num], a
            elif even[num] >= b:
                b = even[num]

        # brute_force奇数位置的数
        ans = odd_cnt + even_cnt - a
        for num in odd:
            cur = odd_cnt - odd[num]
            if even[num] == a:
                x = b
            else:
                x = a
            cur += even_cnt - x
            if cur < ans:
                ans = cur

        return ans

    @staticmethod
    def lc_910(nums: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/smallest-range-ii/description/
        tag: brute_force|data_range
        """
        # brute_force操作的范围，最大值与最小值
        nums.sort()
        ans = nums[-1] - nums[0]
        n = len(nums)
        for i in range(n - 1):
            a, b = nums[n - 1] - k, nums[i] + k
            a = a if a > b else b
            c, d = nums[0] + k, nums[i + 1] - k
            c = c if c < d else d
            if a - c < ans:
                ans = a - c
        return ans

    @staticmethod
    def lc_1178(words: List[str], puzzles: List[str]) -> List[int]:
        """
        url: https://leetcode.cn/problems/number-of-valid-words-for-each-puzzle/
        tag: hash|counter|brute_force|bit_operation
        """
        # classicalhashcounterbrute_force，bit_operation
        dct = defaultdict(int)
        for word in words:
            cur = set(word)
            lst = [ord(w) - ord("a") for w in cur]
            state = reduce(or_, [1 << x for x in lst])
            if len(cur) <= 7:
                dct[state] += 1
        ans = []
        for word in puzzles:
            lst = [ord(w) - ord("a") for w in word]
            n = len(lst)
            cur = 0
            for i in range(1 << (n - 1)):
                i *= 2
                i += 1
                s = sum(1 << lst[j] for j in range(n) if i & (1 << j))
                cur += dct[s]
            ans.append(cur)
        return ans

    @staticmethod
    def lc_1215(low: int, high: int) -> List[int]:
        """
        url: https://leetcode.cn/problems/stepping-numbers/
        tag: data_range|back_track|brute_force
        """

        # data_range|back_trackbrute_force所有满足条件的数

        def dfs():
            nonlocal num, ceil
            if num > ceil:
                return
            ans.append(num)
            last = num % 10
            for x in [last - 1, last + 1]:
                if 0 <= x <= 9:
                    num = num * 10 + x
                    dfs()
                    num //= 10
            return

        ceil = 2 * 10 ** 9
        ans = [0]
        for i in range(1, 10):
            num = i
            dfs()
        ans.sort()
        i, j = bisect.bisect_left(ans, low), bisect.bisect_right(ans, high)
        return ans[i:j]

    @staticmethod
    def lc_1131(arr1: List[int], arr2: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-of-absolute-value-expression/description/
        tag: manhattan_distance|brute_force
        """
        # manhattan_distance，brute_force可能的符号组合
        n = len(arr1)
        ans = 0
        for x in [1, -1]:
            for y in [1, -1]:
                for z in [1, -1]:
                    a1 = max(x * arr1[i] + y * arr2[i] + z * i for i in range(n))
                    a2 = min(x * arr1[i] + y * arr2[i] + z * i for i in range(n))
                    if a1 - a2 > ans:
                        ans = a1 - a2
        return ans

    @staticmethod
    def lc_1638_1(s: str, t: str) -> int:
        """
        url: https://leetcode.cn/problems/count-substrings-that-differ-by-one-character/description/
        tag: brute_force|dp|brute_force
        """
        # brute_force子字符串对开头位置也可DPbrute_force
        m, n = len(s), len(t)
        ans = 0
        for i in range(m):
            for j in range(n):
                cur = int(s[i] != t[j])
                x, y = i, j
                while cur <= 1 and x < m and y < n:
                    ans += cur == 1
                    x += 1
                    y += 1
                    if x == m or y == n:
                        break
                    cur += int(s[x] != t[y])
        return ans

    @staticmethod
    def lc_1638_2(s: str, t: str) -> int:
        """
        url: https://leetcode.cn/problems/count-substrings-that-differ-by-one-character/description/
        tag: brute_force|dp|brute_force
        """
        # brute_force子字符串对开头位置也可DPbrute_force
        m = len(s)
        n = len(t)
        cnt = [[0] * (n + 1) for _ in range(m + 1)]
        same = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if s[i] == t[j]:
                    same[i + 1][j + 1] = same[i][j] + 1  # 以i,j为结尾的最长连续子串长度
                    cnt[i + 1][j + 1] = cnt[i][j]  # 以i,j为结尾的子串对数
                else:
                    same[i + 1][j + 1] = 0  # 转移可以对角线方向转移则只需要O(1)空间
                    cnt[i + 1][j + 1] = same[i][j] + 1
        return sum(sum(d) for d in cnt)

    @staticmethod
    def lc_1761(n: int, edges: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-degree-of-a-connected-trio-in-a-graph/
        tag: directed_graph|undirected_graph|brute_force
        """
        # 无向图转为有向图brute_force
        edges = [[i - 1, j - 1] for i, j in edges]
        degree = [0] * n
        dct = [set() for _ in range(n)]
        directed = [set() for _ in range(n)]
        for i, j in edges:
            dct[i].add(j)
            degree[i] += 1
            degree[j] += 1
            dct[j].add(i)
        for i, j in edges:
            if degree[i] < degree[j] or (degree[i] == degree[j] and i < j):
                directed[i].add(j)
            else:
                directed[j].add(i)
        ans = inf
        for i in range(n):
            for j in directed[i]:
                for k in directed[j]:
                    if k in dct[i]:
                        x = degree[i] + degree[j] + degree[k] - 6
                        if x < ans:
                            ans = x
        return ans if ans < inf else -1

    @staticmethod
    def lc_1878(grid: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/get-biggest-three-rhombus-sums-in-a-grid/
        tag: prefix_sum与|brute_force
        """
        # 两个方向上的prefix_sum与边长brute_force

        m, n = len(grid), len(grid[0])

        @lru_cache(None)
        def left_up(p, q):

            if p < 0 or q < 0:
                return 0
            res = grid[p][q]
            if p and q:
                res += left_up(p - 1, q - 1)

            return res

        @lru_cache(None)
        def right_up(p, q):
            if p < 0 or q < 0:
                return 0
            res = grid[p][q]
            if p and q + 1 < n:
                res += right_up(p - 1, q + 1)

            return res

        ans = set()
        k = max(m, n)
        for i in range(m):
            for j in range(n):
                ans.add(grid[i][j])

                for x in range(1, k + 1):
                    up_point = [i - x, j]
                    down_point = [i + x, j]
                    left_point = [i, j - x]
                    right_point = [i, j + x]
                    if not all(0 <= a < m and 0 <= b < n for a, b in [up_point, down_point, left_point, right_point]):
                        break
                    cur = left_up(right_point[0], right_point[1]) - left_up(up_point[0], up_point[1])
                    cur += left_up(down_point[0], down_point[1]) - left_up(left_point[0], left_point[1])

                    cur += right_up(left_point[0], left_point[1]) - right_up(up_point[0], up_point[1])
                    cur += right_up(down_point[0], down_point[1]) - right_up(right_point[0], right_point[1])
                    cur -= grid[down_point[0]][down_point[1]]
                    cur += grid[up_point[0]][up_point[1]]
                    ans.add(cur)
        ans = list(ans)
        ans.sort(reverse=True)
        return ans[:3]

    @staticmethod
    def lc_2212(x: int, y: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/maximum-points-in-an-archery-competition/
        tag: bit_operation|brute_force|back_track
        """
        # bit_operationbrute_force或者back_track
        n = len(y)
        ans = [0] * n
        ans[0] = x
        res = 0
        for i in range(1 << n):
            lst = [0] * n
            cur = 0
            for j in range(n):
                if i & (1 << j):
                    lst[j] = y[j] + 1
                    cur += j
            s = sum(lst)
            if s <= x:
                lst[0] += x - s
                if cur > res:
                    res = cur
                    ans = lst[:]
        return ans

    @staticmethod
    def lc_2245(grid: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-trailing-zeros-in-a-cornered-path/
        tag: prefix_sum|brute_force
        """
        # 四个方向的prefix_sum与两两组合brute_force

        def check(num, f):
            res = 0
            while num % f == 0:
                res += 1
                num //= f
            return res

        m, n = len(grid), len(grid[0])

        cnt = [[[check(grid[i][j], 2), check(grid[i][j], 5)] for j in range(n)] for i in range(m)]

        @lru_cache(None)
        def up(i, j):
            cur = cnt[i][j][:]
            if i:
                nex = up(i - 1, j)
                cur[0] += nex[0]
                cur[1] += nex[1]
            return cur

        @lru_cache(None)
        def down(i, j):
            cur = cnt[i][j][:]
            if i + 1 < m:
                nex = down(i + 1, j)
                cur[0] += nex[0]
                cur[1] += nex[1]
            return cur

        @lru_cache(None)
        def left(i, j):
            cur = cnt[i][j][:]
            if j:
                nex = left(i, j - 1)
                cur[0] += nex[0]
                cur[1] += nex[1]
            return cur

        @lru_cache(None)
        def right(i, j):
            cur = cnt[i][j][:]
            if j + 1 < n:
                nex = right(i, j + 1)
                cur[0] += nex[0]
                cur[1] += nex[1]
            return cur

        ans = 0
        for i in range(m):
            for j in range(n):
                lst = [up(i, j), down(i, j), left(i, j), right(i, j)]
                for ls in lst:
                    x = ls[0] if ls[0] < ls[1] else ls[1]
                    if x > ans:
                        ans = x
                tmp = cnt[i][j]
                for item in combinations(lst, 2):
                    ls1, ls2 = item
                    x = ls1[0] + ls2[0] - tmp[0] if ls1[0] + ls2[0] - tmp[0] < ls1[1] + ls2[1] - tmp[1] \
                        else ls1[1] + ls2[1] - tmp[1]
                    if x > ans:
                        ans = x
        return ans