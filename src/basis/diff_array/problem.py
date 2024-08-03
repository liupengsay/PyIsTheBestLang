"""

Algorithm：diff_array|prefix_sum|suffix_sum|prefix_max_consequence_sum|suffix_max_consequence_sum|diff_matrix|discretization_diff_array|md_diff_array|matrix_prefix_sum
Description：prefix_sum|prefix_sum_of_prefix_sum|suffix_sum

====================================LeetCode====================================
152（https://leetcode.cn/problems/maximum-product-subarray/）prefix_mul|maximum_sub_consequence_product
598（https://leetcode.cn/problems/range-addition-ii/）diff_matrix
2281（https://leetcode.cn/problems/sum-of-total-strength-of-wizards/）brute_force|prefix_sum_of_prefix_sum
2251（https://leetcode.cn/problems/number-of-flowers-in-full-bloom/）discretization_diff_array
2132（https://leetcode.cn/problems/stamping-the-grid/）prefix_sum|brute_force|diff_matrix|implemention
1229（https://leetcode.cn/problems/meeting-scheduler/）discretization_diff_array
6292（https://leetcode.cn/problems/increment-submatrices-by-one/）diff_matrix|prefix_sum
2565（https://leetcode.cn/problems/subsequence-with-the-minimum-score/）prefix_suffix|pointer|brute_force
644（https://leetcode.cn/problems/maximum-average-subarray-ii/）prefix_sum|binary_search|average
1292（https://leetcode.cn/problems/maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold/）O(mn)|brute_force
1674（https://leetcode.cn/problems/minimum-moves-to-make-array-complementary/）diff_array|action_scope|counter
1714（https://leetcode.cn/problems/sum-of-special-evenly-spaced-elements-in-array/）prefix_sum
1738（https://leetcode.cn/problems/find-kth-largest-xor-coordinate-value/）matrix_prefix_xor_sum
1895（https://leetcode.cn/problems/largest-magic-square/）matrix_prefix_sum|brute_force
1943（https://leetcode.cn/problems/describe-the-painting/）discretization_diff_array
2021（https://leetcode.cn/problems/brightest-position-on-street/）discretization_diff_array
837（https://leetcode.cn/problems/new-21-game/description/）diff_array|implemention|probability
891（https://leetcode.cn/problems/sum-of-subsequence-widths/description/）prefix_suffix|brute_force|counter
1191（https://leetcode.cn/problems/k-concatenation-maximum-sum/description/）prefix_suffix|max_sub_consequence_sum
1074（https://leetcode.cn/problems/number-of-submatrices-that-sum-to-target/description/）matrix_prefix_sum|brute_force
1139（https://leetcode.cn/problems/largest-1-bordered-square/）matrix_prefix_sum|counter|brute_force
2281（https://leetcode.cn/problems/sum-of-total-strength-of-wizards/description/）monotonic_stack|counter|prefix_sum_of_prefix_sum
995（https://leetcode.cn/problems/minimum-number-of-k-consecutive-bit-flips/description/）greedy|diff_array|implemention
986（https://leetcode.cn/problems/interval-list-intersections/description/）discretization_diff_array|two_pointers
1744（https://leetcode.cn/problems/can-you-eat-your-favorite-candy-on-your-favorite-day/description/）prefix_sum|greedy|implemention
1703（https://leetcode.cn/problems/minimum-adjacent-swaps-for-k-consecutive-ones/）prefix_sum|median|greedy|1520E
2167（https://leetcode.cn/problems/minimum-time-to-remove-all-cars-containing-illegal-goods/）math|prefix_sum|brute_force
2983（https://leetcode.cn/problems/palindrome-rearrangement-queries/）brain_teaser|prefix_sum|brute_force|range_intersection
3017（https://leetcode.com/problems/count-the-number-of-houses-at-a-certain-distance-ii/description/）diff_array|classical
100311（https://leetcode.cn/contest/weekly-contest-400/problems/count-days-without-meetings/）discretization_diff_array

=====================================LuoGu======================================
list?user=739032&status=12&page=15（https://www.luogu.com.cn/record/list?user=739032&status=12&page=15）suffix_sum
P2367（https://www.luogu.com.cn/problem/P2367）diff_array
P2280（https://www.luogu.com.cn/problem/P2280）matrix_prefix_sum
P3138（https://www.luogu.com.cn/problem/P3138）matrix_prefix_sum
P3406（https://www.luogu.com.cn/problem/P3406）diff_array|greedy
P3655（https://www.luogu.com.cn/problem/P3655）diff_array|implemention
P5542（https://www.luogu.com.cn/problem/P5542）diff_matrix
P5686（https://www.luogu.com.cn/problem/P5686）prefix_sum_of_prefix_sum
P6180（https://www.luogu.com.cn/problem/P6180）prefix_sum|counter
P6481（https://www.luogu.com.cn/problem/P6481）prefix|range_update
P2956（https://www.luogu.com.cn/problem/P2956）diff_matrix|prefix_sum
P3397（https://www.luogu.com.cn/problem/P3397）diff_matrix|prefix_sum
P1869（https://www.luogu.com.cn/problem/P1869）prefix_sum|comb|odd_even
P7667（https://www.luogu.com.cn/problem/P7667）math|sort|prefix_sum
P2671（https://www.luogu.com.cn/problem/P2671）prefix_or_sum|counter|brute_force|odd_even
P1719（https://www.luogu.com.cn/problem/P1719）max_sub_matrix_sum|brute_force|prefix_sum
P2882（https://www.luogu.com.cn/problem/P2882）greedy|brute_force|diff_array
P4552（https://www.luogu.com.cn/problem/P4552）diff_array|brain_teaser|classical
P1627（https://www.luogu.com.cn/problem/P1627）prefix_suffix|median|counter
P1895（https://www.luogu.com.cn/problem/P1895）prefix_sum|counter|binary_search
P1982（https://www.luogu.com.cn/problem/P1982）maximum_prefix_sub_consequence_sum|prefix_max
P2070（https://www.luogu.com.cn/problem/P2070）hash|discretization_diff_array|counter
P2190（https://www.luogu.com.cn/problem/P2190）diff_array|circular_array
P2352（https://www.luogu.com.cn/problem/P2352）discretization_diff_array
P2363（https://www.luogu.com.cn/problem/P2363）matrix_prefix_sum|brute_force
P2706（https://www.luogu.com.cn/problem/P2706）max_sub_matrix_sum
P2879（https://www.luogu.com.cn/problem/P2879）diff_array|greedy
P3028（https://www.luogu.com.cn/problem/P3028）discretization_diff_array|range_cover
P4030（https://www.luogu.com.cn/problem/P4030）brain_teaser|matrix_prefix_sum
P4440（https://www.luogu.com.cn/problem/P4440）prefix_sum|counter
P4623（https://www.luogu.com.cn/problem/P4623）discretization_diff_array|counter
P6032（https://www.luogu.com.cn/problem/P6032）prefix_suffix|counter
P6278（https://www.luogu.com.cn/problem/P6278）reverse_order_pair|action_scope|diff_array|prefix_sum
P6537（https://www.luogu.com.cn/problem/P6537）prefix_sum|brute_force
P6877（https://www.luogu.com.cn/problem/P6877）sort|greedy|prefix_suffix|dp|brute_force
P6878（https://www.luogu.com.cn/problem/P6878）prefix_suffix|brute_force
P8081（https://www.luogu.com.cn/problem/P8081）diff_array|counter|action_scope
P8033（https://www.luogu.com.cn/problem/P8033）matrix_prefix_sum|counter
P7992（https://www.luogu.com.cn/problem/P7992）bucket_counter|action_scope|diff_array|counter
P7948（https://www.luogu.com.cn/problem/P7948）sort|prefix_suffix|pointer
P8343（https://www.luogu.com.cn/problem/P8343）sub_matrix_prefix_sum|brute_force|two_pointers
P8551（https://www.luogu.com.cn/problem/P8551）diff_array
P8666（https://www.luogu.com.cn/problem/P8666）binary_search|md_diff_array|implemention
P8715（https://www.luogu.com.cn/problem/P8715）prefix_suffix|counter
P8783（https://www.luogu.com.cn/problem/P8783）O(n^3)|two_pointers|brute_force|counter|sub_matrix
P6070（https://www.luogu.com.cn/problem/P6070）diff_array|matrix_diff_array|flatten
P3016（https://www.luogu.com.cn/problem/P3016）prefix_sum|triangle|left_up_sum|inclusion_exclusion


===================================CodeForces===================================
33C（https://codeforces.com/problemset/problem/33/C）prefix_suffix|brute_force
797C（https://codeforces.com/problemset/problem/797/C）suffix_min|lexicographical_order|implemention
75D（https://codeforces.com/problemset/problem/75/D）max_sub_consequence_sum|compress_arrays
1355C（https://codeforces.com/problemset/problem/1355/C）action_scope|diff_array|triangle
1795C（https://codeforces.com/problemset/problem/1795/C）prefix_sum|binary_search|diff_array|counter|implemention
1343D（https://codeforces.com/problemset/problem/1343/D）brute_force|diff_array|counter
1722E（https://codeforces.com/problemset/problem/1722/E）data_range|matrix_prefix_sum
1772D（https://codeforces.com/contest/1772/problem/D）discretization_diff_array|action_scope|counter
1015E2（https://codeforces.com/contest/1015/problem/E2）brute_force|dp|diff_array
1234E（https://codeforces.com/contest/1234/problem/E）brute_force|diff_array|classical|action_scope
1985H1（https://codeforces.com/contest/1985/problem/H1）union_find|contribution_method|diff_matrix|brain_teaser
1985H2（https://codeforces.com/contest/1985/problem/H2）union_find|contribution_method|diff_matrix|brain_teaser
245H（https://codeforces.com/problemset/problem/245/H）interval_dp|prefix_sum

====================================AtCoder=====================================
ABC106D（https://atcoder.jp/contests/abc106/tasks/abc106_d）prefix_sum|dp|counter
ABC338D（https://atcoder.jp/contests/abc338/tasks/abc338_d）diff_array|action_scope|contribution_method
ABC331D（https://atcoder.jp/contests/abc331/tasks/abc331_d）prefix_sum_matrix|circular_section
ABC309C（https://atcoder.jp/contests/abc309/tasks/abc309_c）discretization_diff_array
ABC288D（https://atcoder.jp/contests/abc288/tasks/abc288_d）diff_array|brain_teaser|classical
ABC347E（https://atcoder.jp/contests/abc347/tasks/abc347_e）diff_array|implemention|prefix
ABC347F（https://atcoder.jp/contests/abc347/tasks/abc347_f）diff_array|matrix_prefix_sum|matrix_rotate|brute_force|implemention
ABC274F（https://atcoder.jp/contests/abc274/tasks/abc274_f）brute_force|brain_teaser|discretization_diff_array|classical
ABC269F（https://atcoder.jp/contests/abc269/tasks/abc269_f）diff_array|inclusion_exclusion|prefix_sum|math|classical
ABC268E（https://atcoder.jp/contests/abc268/tasks/abc268_e）brute_force|diff_array|action_scope|brain_teaser|classical
ABC263D（https://atcoder.jp/contests/abc263/tasks/abc263_d）prefix_sum|brute_force
ABC265D（https://atcoder.jp/contests/abc265/tasks/abc265_d）prefix_sum|brute_force
ABC260E（https://atcoder.jp/contests/abc260/tasks/abc260_e）diff_array|action_scope|two_pointer|hash|classical
ABC221D（https://atcoder.jp/contests/abc221/tasks/abc221_d）discretization_diff_array
ABC210D（https://atcoder.jp/contests/abc210/tasks/abc210_d）prefix_max|matrix_prefix|classical
ABC360D（https://atcoder.jp/contests/abc360/tasks/abc360_d）diff_array|implemention|contribution_method

=====================================AcWing=====================================
99（https://www.acwing.com/problem/content/description/101/）matrix_prefix_sum
100（https://www.acwing.com/problem/content/102/）diff_array|classical
101（https://www.acwing.com/problem/content/103/）diff_array|greedy
102（https://www.acwing.com/problem/content/104/）prefix_sum|binary_search|brute_force|average
121（https://www.acwing.com/problem/content/description/123/）discretization_diff_array|prefix_sum|two_pointers|binary_search
126（https://www.acwing.com/problem/content/128/）max_sub_matrix_sum
3993（https://www.acwing.com/problem/content/description/3996/）suffix_sum|data_range|brain_teaser

"""
import bisect
import math
from collections import defaultdict
from itertools import accumulate
from typing import List

from src.basis.binary_search.template import BinarySearch
from src.basis.diff_array.template import DiffMatrix, PreFixSumMatrix, PreFixXorMatrix
from src.graph.union_find.template import UnionFindGeneral
from src.utils.fast_io import FastIO
from src.utils.fast_io import inf


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p3397(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3397
        tag: diff_matrix|prefix_sum
        """
        n, m = ac.read_list_ints()
        shifts = []
        for _ in range(m):
            x1, y1, x2, y2 = ac.read_list_ints()
            shifts.append([x1, x2, y1, y2, 1])
        ans = DiffMatrix().get_diff_matrix(n, n, shifts)
        for a in ans:
            ac.lst(a)
        return

    @staticmethod
    def lg_p4552(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4552
        tag: diff_array|brain_teaser|classical
        """
        n = ac.read_int()
        pre = -1
        pos = 0
        neg = 0
        for _ in range(n):
            num = ac.read_int()
            if pre != -1:
                if pre > num:
                    neg += pre - num
                else:
                    pos += num - pre
            pre = num
        ac.st(max(pos, neg))
        ac.st(abs(pos - neg) + 1)
        return

    @staticmethod
    def lg_p1719(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1719
        tag: max_sub_matrix_sum|brute_force|prefix_sum|classical
        """
        n = ac.read_int()
        total = []
        while len(total) < n * n:
            total.extend(ac.read_list_ints())
        grid = []
        for i in range(n):
            grid.append(total[i * n: (i + 1) * n])
        del total

        ans = float("-inf")
        for i in range(n):
            lst = [0] * n
            for j in range(i, n):
                lst = [lst[k] + grid[j][k] for k in range(n)]
                floor = pre = 0
                for num in lst:
                    pre += num
                    ans = ac.max(ans, pre - floor)
                    floor = ac.min(floor, pre)
        ac.st(ans)
        return

    @staticmethod
    def cf_1722e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1722/E
        tag: data_range|matrix_prefix_sum|classical|can_be_discretization_hard_version
        """
        for _ in range(ac.read_int()):
            k, q = ac.read_list_ints()
            rec = [ac.read_list_ints() for _ in range(k)]
            qur = [ac.read_list_ints() for _ in range(q)]
            m = n = 1001
            dp = [[0] * n for _ in range(m)]
            for a, b in rec:
                dp[a][b] += a * b
            pre = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m):
                for j in range(n):
                    pre[i + 1][j + 1] = pre[i + 1][j] + \
                                        pre[i][j + 1] - pre[i][j] + dp[i][j]

            for hs, ws, hb, wb in qur:
                hb -= 1
                wb -= 1
                hs += 1
                ws += 1
                ans = pre[hb + 1][wb + 1] - pre[hs][wb + 1] - \
                      pre[hb + 1][ws] + pre[hs][ws]
                ac.st(ans)
        return

    @staticmethod
    def lg_p2671(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2671
        tag: prefix_or_sum|counter|brute_force|odd_even
        """
        n, m = ac.read_list_ints()
        number = ac.read_list_ints()
        colors = ac.read_list_ints()
        mod = 10007

        ans = 0
        pre_sum = [[0, 0] for _ in range(m + 1)]
        pre_cnt = [[0, 0] for _ in range(m + 1)]
        for i in range(n):
            num, color = number[i], colors[i]
            k = i % 2
            z_ax = (i + 1) * pre_sum[color][k]
            z_az = (i + 1) * num * pre_cnt[color][k]
            ans += z_ax + z_az
            pre_sum[color][k] += num
            pre_cnt[color][k] += 1
            ans %= mod

        pre_sum = [[0, 0] for _ in range(m + 1)]
        pre_cnt = [[0, 0] for _ in range(m + 1)]
        for i in range(n - 1, -1, -1):
            num, color = number[i], colors[i]
            k = i % 2
            x_az = (i + 1) * pre_sum[color][k]
            x_ax = (i + 1) * num * pre_cnt[color][k]
            ans += x_ax + x_az
            pre_sum[color][k] += num
            pre_cnt[color][k] += 1
            ans %= mod

        ac.st(ans)
        return

    @staticmethod
    def cf_1795c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1795/C
        tag: prefix_sum|binary_search|diff_array|counter|implemention
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            a = ac.read_list_ints()
            b = ac.read_list_ints()
            pre = [0] * (n + 1)
            for i in range(n):
                pre[i + 1] = pre[i] + b[i]

            ans = [0] * n
            diff = [0] * n
            for i in range(n):
                j = bisect.bisect_left(pre, pre[i] + a[i])
                if j == n + 1 or pre[j] > pre[i] + a[i]:
                    j -= 1
                diff[i] += 1
                if j < n:
                    diff[j] -= 1
                if pre[j] - pre[i] < a[i]:
                    if j < n:
                        ans[j] += a[i] - (pre[j] - pre[i])
            for i in range(1, n):
                diff[i] += diff[i - 1]

            for i in range(n):
                ans[i] += b[i] * diff[i]
            ac.lst(ans)
        return

    @staticmethod
    def lc_995(nums: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/minimum-number-of-k-consecutive-bit-flips/description/
        tag: greedy|diff_array|implemention
        """
        n = len(nums)
        ans = 0
        diff = [0] * (n + 1)
        for i in range(n - k + 1):
            diff[i] += diff[i - 1] if i else 0
            nums[i] += diff[i]
            nums[i] %= 2
            if nums[i] == 0:
                nums[i] = 1
                diff[i] += 1
                diff[i + k] -= 1
                ans += 1
        for i in range(n - k + 1, n):
            diff[i] += diff[i - 1] if i else 0
            nums[i] += diff[i]
            nums[i] %= 2
        return ans if all(x == 1 for x in nums) else -1

    @staticmethod
    def lc_1074(matrix: List[List[int]], target: int) -> int:
        """
        url: https://leetcode.cn/problems/number-of-submatrices-that-sum-to-target/description/
        tag: matrix_prefix_sum|brute_force|classical
        """
        m, n = len(matrix), len(matrix[0])
        pre = PreFixSumMatrix(matrix)
        ans = 0
        for i in range(m):
            for j in range(i, m):
                dct = defaultdict(int)
                dct[0] = 1
                for k in range(n):
                    cur = pre.query(i, 0, j, k)
                    ans += dct[cur - target]
                    dct[cur] += 1
        return ans

    @staticmethod
    def lc_1191(arr: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/k-concatenation-maximum-sum/description/
        tag: prefix_suffix|max_sub_consequence_sum
        """
        mod = 10 ** 9 + 7
        n = len(arr)
        s = sum(arr)
        pre = [0] * n
        x = 0
        for i in range(n):
            x = x if x > 0 else 0
            x += arr[i]
            pre[i] = x

        post = [0] * n
        x = 0
        for i in range(n - 1, -1, -1):
            x = x if x > 0 else 0
            x += arr[i]
            post[i] = x
        ans = max(0, max(pre))
        if k > 1:
            if pre[-1] + post[0] > ans:
                ans = pre[-1] + post[0]
            if pre[-1] + post[0] + s * (k - 2) > ans:
                ans = pre[-1] + post[0] + s * (k - 2)
        return ans % mod

    @staticmethod
    def cf_1355c(a, b, c, d):
        """
        url: https://codeforces.com/problemset/problem/1355/C
        tag: action_scope|diff_array|triangle|classical|brain_teaser|brute_force
        """
        diff = [0] * (b + c + 1)
        for x in range(a, b + 1):
            diff[x + b] += 1
            if x + c + 1 <= b + c:
                diff[x + c + 1] -= 1
        for i in range(1, b + c + 1):
            diff[i] += diff[i - 1]

        for i in range(1, b + c + 1):
            diff[i] += diff[i - 1]

        ans = 0
        for z in range(c, d + 1):
            ans += diff[-1] - diff[min(z, b + c)]
        return ans

    @staticmethod
    def lc_2281(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/sum-of-total-strength-of-wizards/description/
        tag: monotonic_stack|counter|prefix_sum_of_prefix_sum|classical|brain_teaser
        """
        n = len(nums)
        post = [n - 1] * n
        pre = [0] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] > nums[i]:
                post[stack.pop()] = i - 1
            if stack:
                pre[i] = stack[-1] + 1
            stack.append(i)
        mod = 10 ** 9 + 7
        s = list(accumulate(nums, initial=0))
        ss = list(accumulate(s, initial=0))
        ans = 0
        for i in range(n):
            left = pre[i]
            right = post[i]
            ans += nums[i] * ((i - left + 1) * (ss[right + 2] - ss[i + 1]) - (right - i + 1) * (ss[i + 1] - ss[left]))
            ans %= mod
        return ans

    @staticmethod
    def lc_2565(s: str, t: str) -> int:
        """
        url: https://leetcode.cn/problems/subsequence-with-the-minimum-score/
        tag: prefix_suffix|pointer|brute_force|reverse_thinking
        """
        m, n = len(s), len(t)
        pre = [0] * (m + 1)
        ind = 0
        for i in range(m):
            if ind < n and s[i] == t[ind]:
                ind += 1
            pre[i + 1] = ind
        if ind == n:
            return 0

        post = [0] * (m + 1)
        ind = 0
        for i in range(m - 1, -1, -1):
            if ind < n and s[i] == t[-ind - 1]:
                ind += 1
            post[i] = ind

        ans = min(n - (post[i] + pre[i]) for i in range(m + 1))
        return ans

    @staticmethod
    def lg_p2882(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2882
        tag: greedy|brute_force|diff_array|classical
        """
        n = ac.read_int()
        lst = [int(ac.read_str() == "F") for _ in range(n)]
        ans = [inf, 0]
        for k in range(1, n + 1):
            diff = [0] * n
            m = 0
            for i in range(n - k + 1):
                if i:
                    diff[i] += diff[i - 1]
                x = diff[i] + lst[i]
                if x % 2:
                    continue
                else:
                    m += 1
                    diff[i] += 1
                    if i + k < n:
                        diff[i + k] -= 1
            for i in range(n - k + 1, n):
                diff[i] += diff[i - 1]
                if (diff[i] + lst[i]) % 2 == 0:
                    break
            else:
                if [m, k] < ans:
                    ans = [m, k]
        ac.lst(ans[::-1])
        return

    @staticmethod
    def cf_1772d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1772/problem/D
        tag: discretization_diff_array|action_scope|counter|classical
        """
        ceil = 10 ** 9
        for _ in range(ac.read_int()):
            n = ac.read_int()
            diff = defaultdict(int)
            nums = ac.read_list_ints()
            for i in range(1, n):
                a, b = nums[i - 1], nums[i]
                if a == b:
                    diff[0] += 1
                    diff[ceil + 1] -= 1
                elif a < b:
                    mid = a + (b - a) // 2
                    diff[0] += 1
                    diff[mid + 1] -= 1
                else:
                    mid = b - (b - a) // 2
                    diff[mid] += 1
                    diff[ceil + 1] -= 1

            axis = sorted(list(diff.keys()))
            m = len(axis)
            for i in range(m):
                if i:
                    diff[axis[i]] += diff[axis[i - 1]]
                if diff[axis[i]] == n - 1:
                    ac.st(axis[i])
                    break
            else:
                ac.st(-1)
        return

    @staticmethod
    def ac_99(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/101/
        tag: matrix_prefix_sum
        """
        n, m = ac.read_list_ints()

        lst = [ac.read_list_ints() for _ in range(n)]
        length = max(max(ls[:-1]) for ls in lst) + 1
        grid = [[0] * length for _ in range(length)]
        for x, y, v in lst:
            grid[x][y] += v
        ans = 0
        dp = [[0] * (length + 1) for _ in range(length + 1)]
        for i in range(length):
            for j in range(length):
                dp[i + 1][j + 1] = dp[i][j + 1] + \
                                   dp[i + 1][j] - dp[i][j] + grid[i][j]
                a, b = max(i - m + 1, 0), max(j - m + 1, 0)
                cur = dp[i + 1][j + 1] - dp[i + 1][b] - dp[a][j + 1] + dp[a][b]
                ans = ans if ans > cur else cur
        ac.st(ans)
        return

    @staticmethod
    def ac_102(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/104/
        tag: prefix_sum|binary_search|brute_force|average
        """
        n, f = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(n)]

        def check(x):
            y = 0
            pre = [0] * (n + 1)
            for i in range(n):
                y += nums[i] * 1000 - x
                pre[i + 1] = pre[i] if pre[i] < y else y
                if i >= f - 1 and y - pre[i - f + 1] >= 0:
                    return True
            return False

        ans = BinarySearch().find_int_right(0, max(nums) * 1000, check)
        ac.st(ans)
        return

    @staticmethod
    def ac_121(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/123/
        tag: discretization_diff_array|prefix_sum|two_pointers|binary_search|classical
        """
        c, b = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(b)]
        lst_x = sorted(list(set([x for x, _ in nums])))
        lst_y = sorted(list(set([x for _, x in nums])))
        m = len(lst_x)
        n = len(lst_y)
        ind_x = {num: i for i, num in enumerate(lst_x)}
        ind_y = {num: i for i, num in enumerate(lst_y)}
        grid = [[0] * (n + 1) for _ in range(m + 1)]
        for x, y in nums:
            grid[ind_x[x] + 1][ind_y[y] + 1] += 1
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                grid[i][j] = grid[i - 1][j] + grid[i][j - 1] - grid[i - 1][j - 1] + grid[i][j]

        def check(xx):
            up = 0
            for ii in range(m):
                while up < m and lst_x[up] - lst_x[ii] <= xx - 1:
                    up += 1
                right = 0
                for jj in range(n):
                    while right < n and lst_y[right] - lst_y[jj] <= xx - 1:
                        right += 1
                    cur = grid[up][right] - grid[up][jj] - grid[ii][right] + grid[ii][jj]
                    if cur >= c:
                        return True

            return False

        ans = BinarySearch().find_int_left(0, 10000, check)
        return

    @staticmethod
    def ac_126(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/128/
        tag: max_sub_matrix_sum|brute_force
        """
        n = ac.read_int()
        nums = []
        while len(nums) < n * n:
            nums.extend(ac.read_list_ints())
        grid = [nums[i:i + n] for i in range(0, n * n, n)]
        del nums
        ans = grid[0][0]
        for i in range(n):
            pre = [0] * n
            for k in range(i, n):
                pre = [pre[j] + grid[k][j] for j in range(n)]
                floor = 0
                x = 0
                for j in range(n):
                    x += pre[j]
                    ans = ac.max(ans, x - floor)
                    floor = ac.min(floor, x)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1627(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1627
        tag: prefix_suffix|median|counter|classical
        """
        n, b = ac.read_list_ints()
        nums = ac.read_list_ints()
        i = nums.index(b)

        pre = defaultdict(int)
        cnt = ans = 0
        for j in range(i - 1, -1, -1):
            num = nums[j]
            cnt += 1 if num > b else -1
            pre[cnt] += 1
            if cnt == 0:
                ans += 1

        cnt = 0
        for j in range(i + 1, n):
            num = nums[j]
            cnt += 1 if num > b else -1
            ans += pre[-cnt]
            ans += 1 if not cnt else 0
        ans += 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p1895(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1895
        tag: prefix_sum|counter|binary_search
        """
        n = 10 ** 5
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] + len(str(i))

        pre = [0] * (n + 1)
        for i in range(1, n + 1):
            pre[i] = pre[i - 1] + dp[i]

        def check(x):
            ii = bisect.bisect_left(pre, x)
            rest = x - pre[ii - 1]
            j = bisect.bisect_left(dp, rest)
            d = rest - dp[j - 1]
            return str(j)[d - 1]

        for _ in range(ac.read_int()):
            ac.st(check(ac.read_int()))
        return

    @staticmethod
    def lg_p1982(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1982
        tag: maximum_prefix_sub_consequence_sum|prefix_max
        """
        n, p = ac.read_list_ints()
        nums = ac.read_list_ints()
        pre = 0
        for i in range(n):
            pre = pre if pre > 0 else 0
            pre += nums[i]
            nums[i] = pre
            if i:
                nums[i] = ac.max(nums[i], nums[i - 1])

        final = nums[0]
        pre = nums[0] * 2
        for i in range(1, n):
            final = ac.max(final, pre)
            pre = ac.max(pre, pre + nums[i])
        pos = 1 if final > 0 else -1
        ac.st(pos * (abs(final) % p))
        return

    @staticmethod
    def lg_p2070(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2070
        tag: hash|discretization_diff_array|counter
        """
        n = ac.read_int()
        pos = 0
        diff = defaultdict(int)
        for _ in range(n):
            dis, op = ac.read_list_strs()
            dis = int(dis)
            if op == "L":
                diff[pos - dis] += 1
                diff[pos] -= 1
                pos -= dis
            else:
                diff[pos] += 1
                diff[pos + dis] -= 1
                pos += dis

        axis = sorted(diff.keys())
        m = len(axis)
        ans = 0
        for i in range(1, m):
            diff[axis[i]] += diff[axis[i - 1]]
            if diff[axis[i - 1]] >= 2:
                ans += axis[i] - axis[i - 1]
        ac.st(ans)
        return

    @staticmethod
    def lg_p2190(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2190
        tag: diff_array|circular_array
        """
        n, m = ac.read_list_ints()
        diff = [0] * n
        for _ in range(m):
            x, y, z = ac.read_list_ints()
            x -= 1
            y -= 1
            if x <= y:
                diff[x] += z
                diff[y] -= z
            else:
                diff[x] += z
                if y > 0:
                    diff[0] += z
                    diff[y] -= z
        for i in range(1, n):
            diff[i] += diff[i - 1]
        ac.st(math.ceil(max(diff) / 36))
        return

    @staticmethod
    def lg_p2352(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2352
        tag: discretization_diff_array
        """
        diff = defaultdict(int)
        for _ in range(ac.read_int()):
            a, b = ac.read_list_ints()
            diff[a] += 1
            diff[b] += 0
            diff[b + 1] -= 1
        axis = sorted(list(diff.keys()))
        m = len(axis)
        ans = diff[axis[0]] * axis[0]
        for i in range(1, m):
            diff[axis[i]] += diff[axis[i - 1]]
            ans = ac.max(ans, diff[axis[i]] * axis[i])
        ac.st(ans)
        return

    @staticmethod
    def lg_p2363(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2363
        tag: matrix_prefix_sum|brute_force|classical
        """
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        pre = PreFixSumMatrix(nums)
        ans = 0
        for i in range(n):
            for j in range(n):
                dct = defaultdict(int)
                for x in range(i + 1):
                    for y in range(j + 1):
                        dct[pre.query(x, y, i, j)] += 1
                for p in range(i + 1, n):
                    for q in range(j + 1, n):
                        ans += dct[pre.query(i + 1, j + 1, p, q)]

                dct = defaultdict(int)
                for x in range(i + 1):
                    for y in range(j, n):
                        dct[pre.query(x, j, i, y)] += 1
                for p in range(i + 1, n):
                    for q in range(j):
                        ans += dct[pre.query(i + 1, q, p, j - 1)]
        ac.st(ans)
        return

    @staticmethod
    def lg_p2706(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2706
        tag: max_sub_matrix_sum|brute_force|classical|monotonic_stack|matrix_prefix_sum
        """
        m, n = ac.read_list_ints()
        grid = []
        while len(grid) < m * n:
            grid.extend(ac.read_list_ints())
        grid = [grid[i:i + n] for i in range(0, m * n, n)]
        pre = PreFixSumMatrix(grid)
        ans = 0
        height = [0] * n
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    height[j] += 1
                else:
                    height[j] = 0

            left = [0] * n
            right = [n - 1] * n
            stack = []
            for j in range(n):
                while stack and height[stack[-1]] > height[j]:
                    right[stack.pop()] = j - 1
                if stack:
                    left[j] = stack[-1] + 1
                stack.append(j)

            for j in range(n):
                if height[j]:
                    cur = pre.query(i - height[j] + 1, left[j], i, right[j])
                    ans = ans if ans > cur else cur
        ac.st(ans)
        return

    @staticmethod
    def lg_p2879(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2879
        tag: diff_array|greedy
        """
        n, _, h, r = ac.read_list_ints()
        diff = [0] * n
        pre = set()
        for _ in range(r):
            a, b = ac.read_list_ints_minus_one()
            if a > b:
                a, b = b, a
            if (a, b) in pre:
                continue
            pre.add((a, b))
            diff[a + 1] -= 1
            diff[b] += 1
        for i in range(1, n):
            diff[i] += diff[i - 1]
        gap = h - max(diff)
        for d in diff:
            ac.st(d + gap)
        return

    @staticmethod
    def lg_p3028(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3028
        tag: discretization_diff_array|range_cover|reverse_thinking
        """
        n = ac.read_int()
        diff = defaultdict(int)
        for _ in range(n):
            a, b = ac.read_list_ints()
            if a > b:
                a, b = b, a
            diff[a] += 1
            diff[b + 1] -= 1
            diff[b] += 0
        axis = sorted(list(diff.keys()))
        ans = diff[axis[0]]
        m = len(axis)
        for i in range(1, m):
            diff[axis[i]] += diff[axis[i - 1]]
            ans = ac.max(ans, diff[axis[i]])
        ac.st(ans)
        return

    @staticmethod
    def lg_p4030(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4030
        tag: brain_teaser|matrix_prefix_sum|classical
        """
        m, n, t = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        mat = [[0] * n for _ in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                a, b = grid[i - 1][j - 1], grid[i - 1][j]
                c, d = grid[i][j - 1], grid[i][j]
                if a + d == b + c:
                    mat[i][j] = 1
        pm = PreFixSumMatrix(mat)
        for i in range(t):
            x, y, k = ac.read_list_ints()
            if k == 1:
                ac.st("Y")
                continue
            x -= 1
            y -= 1
            if pm.query(x + 1, y + 1, x + k - 1, y + k - 1) == (k - 1) * (k - 1):
                ac.st("Y")
            else:
                ac.st("N")
        return

    @staticmethod
    def lg_p4440(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4440
        tag: prefix_sum|counter|alphabet|date_range
        """
        s = ac.read_str()
        pre = []
        cnt = [0] * 26
        pre.append(cnt[:])
        for w in s:
            cnt[ord(w) - ord("a")] += 1
            pre.append(cnt[:])
        for _ in range(ac.read_int()):
            a, b, c, d = ac.read_list_ints_minus_one()
            if d - c != b - a:
                ac.st("NE")
                continue
            if all(pre[b + 1][j] - pre[a][j] == pre[d + 1][j] - pre[c][j] for j in range(26)):
                ac.st("DA")
            else:
                ac.st("NE")
        return

    @staticmethod
    def lg_p4623(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4623
        tag: discretization_diff_array|counter|triangle
        """
        n = ac.read_int()
        m = 10 ** 6 + 1
        diff_x = [0] * m
        diff_y = [0] * m
        for _ in range(n):
            x1, y1, x2, y2, x3, y3 = ac.read_list_ints()
            low_x = min(x1, x2, x3)
            high_x = max(x1, x2, x3)
            low_y = min(y1, y2, y3)
            high_y = max(y1, y2, y3)
            diff_x[low_x + 1] += 1
            diff_x[high_x] -= 1
            diff_y[low_y + 1] += 1
            diff_y[high_y] -= 1

        for i in range(1, m):
            diff_x[i] += diff_x[i - 1]
        for i in range(1, m):
            diff_y[i] += diff_y[i - 1]

        for _ in range(ac.read_int()):
            op, _, num = ac.read_list_strs()
            num = int(num)
            if op == "x":
                ac.st(diff_x[num])
            else:
                ac.st(diff_y[num])
        return

    @staticmethod
    def lg_p6032(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6032
        tag: prefix_suffix|counter|classical
        """
        n, k, p = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        post = [0] * k
        for cc, _ in nums:
            post[cc] += 1

        pre = dict()
        ans = ss = 0
        for i in range(n):
            cc, pp = nums[i]
            if pp <= p:
                ans += ss + post[cc] - 1
                ss = 0
                pre = dict()
                post[cc] -= 1
                continue

            ss -= pre.get(cc, 0)
            pre[cc] = pre.get(cc, 0) + 1
            post[cc] -= 1
            ss += post[cc]
        ac.st(ans)
        return

    @staticmethod
    def lg_p6278(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6278
        tag: reverse_order_pair|action_scope|diff_array|prefix_sum
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        diff = [0] * (n + 1)
        pre = []
        for num in nums:
            diff[num] += len(pre) - bisect.bisect_right(pre, num)
            bisect.insort_left(pre, num)
        diff = ac.accumulate(diff)
        for i in range(n):
            ac.st(diff[i])
        return

    @staticmethod
    def lg_p6537(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6537
        tag: prefix_sum|brute_force
        """
        n = ac.read_int()
        grid = [ac.read_list_ints() for _ in range(n)]
        pre = PreFixSumMatrix(grid)
        ans = 0
        for i in range(n):
            for j in range(n):
                dct = dict()
                for x in range(i + 1):
                    for y in range(j + 1):
                        val = pre.query(x, y, i, j)
                        dct[val] = dct.get(val, 0) + 1
                for x in range(i + 1, n):
                    for y in range(j + 1, n):
                        val = pre.query(i + 1, j + 1, x, y)
                        ans += dct.get(val, 0)
                dct = defaultdict(int)
                for x in range(i + 1):
                    for y in range(j, n):
                        val = pre.query(x, j, i, y)
                        dct[val] = dct.get(val, 0) + 1
                for x in range(i + 1, n):
                    for y in range(j):
                        val = pre.query(i + 1, y, x, j - 1)
                        ans += dct.get(val, 0)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6877(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6877
        tag: sort|greedy|prefix_suffix|dp|brute_force
        """
        n = ac.read_int()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        aa = a[:]

        a.sort()
        b.sort()
        pre = [0] * (n + 2)
        for i in range(n):
            pre[i + 1] = ac.max(pre[i], a[i] - b[i])

        post = [0] * (n + 2)
        for i in range(n - 1, -1, -1):
            post[i] = ac.max(post[i + 1], a[i + 1] - b[i])

        ans = dict()
        for i in range(n + 1):
            ans[a[i]] = ac.max(pre[i], post[i])
        ac.lst([ans[x] for x in aa])
        return

    @staticmethod
    def lg_p6878(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6878
        tag: two_pointers|brute_force
        """
        n, k = ac.read_list_ints()
        s = ac.read_str()
        ans = inf
        ind1 = [i for i in range(n) if s[i] == "J"]
        ind2 = [i for i in range(n) if s[i] == "O"]
        ind3 = [i for i in range(n) if s[i] == "I"]
        m = len(ind2)
        left = 0
        right = 0

        for mid in range(m - k + 1):
            while left + 1 < len(ind1) and ind1[left + 1] < ind2[mid]:
                left += 1
            while right < len(ind3) and ind3[right] <= ind2[mid + k - 1]:
                right += 1

            if k - 1 <= left < len(ind1) and right < len(ind3) and ind1[left] < ind2[mid] < ind3[
                right] and right + k - 1 < len(ind3):
                cur = ind3[right + k - 1] - ind1[left - k + 1] + 1 - 3 * k
                if cur < ans:
                    ans = cur
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def lg_p8081(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8081
        tag: diff_array|counter|action_scope
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        ice = []
        cnt = 0
        for i in range(n):
            if nums[i] < 0:
                cnt += 1
            else:
                if cnt:
                    ice.append((i - cnt, i - 1))
                cnt = 0
        if cnt:
            ice.append((n - cnt, n - 1))
        if not ice:
            ac.st(0)
            return

        diff = [0] * n
        ceil = 0
        for x, y in ice:
            t = y - x + 1
            ceil = ac.max(ceil, t)
            low = ac.max(0, x - 2 * t)
            if low <= x - 1:
                diff[low] += 1
                diff[x] -= 1

        diff = ac.accumulate(diff)
        diff = ac.accumulate([int(x == 0) for x in diff[1:]])
        ans = n - diff[-1]
        another = 0
        for x, y in ice:
            t = y - x + 1
            if x - 2 * t - 1 >= 0 and t == ceil:
                another = ac.max(another, diff[x - 2 * t] - diff[ac.max(x - 3 * t, 0)])
        ac.st(ans + another)
        return

    @staticmethod
    def lg_p8033(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8033
        tag: matrix_prefix_sum|counter|specific_plan
        """
        m, n, k = ac.read_list_ints()
        grid = [list(ac.read_str()) for _ in range(m)]
        mat = [[int(w == "*") for w in lst] for lst in grid]
        pre = PreFixSumMatrix(mat)
        val = 0
        ans = []

        for i in range(k - 1, m):
            for j in range(k - 1, n):
                cur = pre.query(i - k + 2, j - k + 2, i - 1, j - 1)
                if cur > val:
                    val = cur
                    ans = [i, j]
        i, j = ans
        x1, y1, x2, y2 = i - k + 1, j - k + 1, i, j
        grid[x1][y1] = grid[x1][y2] = "+"
        grid[x2][y1] = grid[x2][y2] = "+"
        for i in [x1, x2]:
            for j in range(y1 + 1, y2):
                grid[i][j] = "-"
        for j in [y1, y2]:
            for i in range(x1 + 1, x2):
                grid[i][j] = "|"
        ac.st(val)
        for g in grid:
            ac.st("".join(g))
        return

    @staticmethod
    def lg_p7992(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P7992
        tag: bucket_counter|action_scope|diff_array|counter|data_range|inclusion_exclusion
        """
        n, m = ac.read_list_ints()
        a = [0] * (m + 1)
        b = [0] * (m + 1)
        diff = [0] * (2 * m + 2)
        for _ in range(n):
            x, y = ac.read_list_ints()
            a[x] += 1
            b[y] += 1
        for i in range(m + 1):
            for j in range(m + 1):
                diff[i + j] += a[i] * a[j]
                diff[i + j + 1] -= b[i] * b[j]
        for i in range(2 * m + 1):
            if i:
                diff[i] += diff[i - 1]
            ac.st(diff[i])
        return

    @staticmethod
    def lg_p7948(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P7948
        tag: sort|prefix_suffix|pointer|classical|offline_query
        """
        for _ in range(ac.read_int()):
            n, q = ac.read_list_ints()
            a = ac.read_list_ints()
            b = ac.read_list_ints()
            a.sort(reverse=True)
            pre = [0] * n
            s = 0
            for i in range(n):
                s += a[i]
                pre[i] = (i + 1) * a[i] - s

            ind = list(range(q))
            ind.sort(key=lambda it: -b[it])
            ans = [-1] * q
            j = n - 1
            for i in ind:
                k = b[i]
                while j >= 0 and pre[j] < - k * (j + 1):
                    j -= 1
                ans[i] = j + 1
            ac.lst(ans)
        return

    @staticmethod
    def lg_p8343(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8343
        tag: sub_matrix_prefix_sum|brute_force|two_pointers|math
        """
        m, n, a, b = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        if a > b:
            a, b = b, a
        pre = PreFixSumMatrix(grid)
        ans = inf
        for i in range(m):
            for k in range(i, m):
                lst = [0]
                ind_a = ind_b = 0
                for j in range(n):
                    cur = pre.query(i, 0, k, j)
                    lst.append(cur)
                    while ind_a + 1 < j + 1 and cur - lst[ind_a] >= a:
                        ans = ac.min(ans, abs(cur - lst[ind_a] - a) + abs(cur - lst[ind_a] - b))
                        ind_a += 1

                    while ind_b + 1 < j + 1 and cur - lst[ind_b] <= b:
                        ans = ac.min(ans, abs(cur - lst[ind_b] - a) + abs(cur - lst[ind_b] - b))
                        ind_b += 1

                    ans = ac.min(ans, abs(cur - lst[ind_a] - a) + abs(cur - lst[ind_a] - b))
                    ans = ac.min(ans, abs(cur - lst[ind_b] - a) + abs(cur - lst[ind_b] - b))
                    if ans == b - a:
                        ac.st(ans)
                        return
        ac.st(ans)
        return

    @staticmethod
    def lg_p8551(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8551
        tag: diff_array|brain_teaser|pointer|classical|brute_force
        """
        n = ac.read_int()
        m = 3 * 10 ** 5 + 1
        diff = [0] * (m + 2)
        point = [0] * (m + 2)
        for _ in range(n):
            a, b = ac.read_list_ints()
            diff[a] += 1
            diff[b + 1] -= 1
            point[a - 1] = 1
            point[b] = 1
        ans = 0
        pre = inf
        for i in range(1, m + 2):
            diff[i] += diff[i - 1]
            if point[i]:
                ans = ac.max(ans, diff[i] * (i - pre))
                pre = i + 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p8666(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8666
        tag: binary_search|md_diff_array|implemention|classical|matrix_flatten|inclusion_exclusion|performance
        """
        aa, bb, cc, m = ac.read_list_ints()
        nums = ac.read_list_ints()

        def tuple_to_pos(i, j, k, b, c):
            pos = (i * b + j) * c + k
            return pos

        lst = [ac.read_list_ints() for _ in range(m)]
        diff = [0] * (cc + 2) * (bb + 2) * (aa + 2)

        def check(x):
            for i in range(len(diff)):
                diff[i] = 0
            for i1, i2, j1, j2, k1, k2, h in lst[:x]:
                diff[tuple_to_pos(i1, j1, k1, bb + 2, cc + 2)] += h

                diff[tuple_to_pos(i2 + 1, j1, k1, bb + 2, cc + 2)] -= h
                diff[tuple_to_pos(i1, j2 + 1, k1, bb + 2, cc + 2)] -= h
                diff[tuple_to_pos(i1, j1, k2 + 1, bb + 2, cc + 2)] -= h

                diff[tuple_to_pos(i2 + 1, j2 + 1, k1, bb + 2, cc + 2)] += h
                diff[tuple_to_pos(i1, j2 + 1, k2 + 1, bb + 2, cc + 2)] += h
                diff[tuple_to_pos(i2 + 1, j1, k2 + 1, bb + 2, cc + 2)] += h

                diff[tuple_to_pos(i2 + 1, j2 + 1, k2 + 1, bb + 2, cc + 2)] -= h

            for i1 in range(aa):
                for j1 in range(bb):
                    for k1 in range(cc):
                        cur = diff[tuple_to_pos(i1 + 1, j1 + 1, k1 + 1, bb + 2, cc + 2)]

                        cur += diff[tuple_to_pos(i1, j1 + 1, k1 + 1, bb + 2, cc + 2)]
                        cur += diff[tuple_to_pos(i1 + 1, j1, k1 + 1, bb + 2, cc + 2)]
                        cur += diff[tuple_to_pos(i1 + 1, j1 + 1, k1, bb + 2, cc + 2)]

                        cur -= diff[tuple_to_pos(i1, j1, k1 + 1, bb + 2, cc + 2)]
                        cur -= diff[tuple_to_pos(i1, j1 + 1, k1, bb + 2, cc + 2)]
                        cur -= diff[tuple_to_pos(i1 + 1, j1, k1, bb + 2, cc + 2)]

                        cur += diff[tuple_to_pos(i1, j1, k1, bb + 2, cc + 2)]

                        diff[tuple_to_pos(i1 + 1, j1 + 1, k1 + 1, bb + 2, cc + 2)] = cur
                        if cur > nums[tuple_to_pos(i1, j1, k1, bb, cc)]:
                            return True

            return False

        ans = BinarySearch().find_int_left(1, m, check)
        ac.st(ans)
        return

    @staticmethod
    def lc_891(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/sum-of-subsequence-widths/description/
        tag: prefix_suffix|brute_force|counter|contribution_method
        """
        mod = 10 ** 9 + 7
        dp = [1]
        for i in range(10 ** 5):
            dp.append((dp[-1] * 2) % mod)
        n = len(nums)
        nums.sort()
        ans = 0
        for i in range(n):
            ans += nums[i] * dp[i]
            ans -= nums[i] * dp[n - 1 - i]
            ans %= mod
        return ans

    @staticmethod
    def lc_1292(mat: List[List[int]], threshold: int) -> int:
        """
        url: https://leetcode.cn/problems/maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold/
        tag: O(mn)|brute_force|binary_search|classical
        """
        m, n = len(mat), len(mat[0])
        ans = 0
        pre = PreFixSumMatrix(mat)
        for i in range(m):
            for j in range(n):
                r = n - j if n - j < m - i else m - i
                for d in range(ans + 1, r + 1):
                    cur = pre.query(i, j, i + d - 1, j + d - 1)
                    if cur > threshold:
                        break
                    ans = d
        return ans

    @staticmethod
    def lc_1674(nums: List[int], limit: int) -> int:
        """
        url: https://leetcode.cn/problems/minimum-moves-to-make-array-complementary/
        tag: diff_array|action_scope|counter|contribution_method|classical
        """
        n = len(nums)
        diff = [0] * (2 * limit + 2)
        for i in range(n // 2):
            x, y = nums[i], nums[n - 1 - i]
            low_1 = 1 + x if x < y else 1 + y
            high_1 = limit + x if x > y else limit + y
            if low_1 <= high_1:
                diff[low_1] += 1
                diff[high_1 + 1] -= 1
            diff[x + y] -= 1
            diff[x + y + 1] += 1

            if 2 <= low_1 - 1:
                diff[2] += 2
                diff[low_1] -= 2
            if high_1 + 1 <= 2 * limit:
                diff[high_1 + 1] += 2
                diff[2 * limit + 1] -= 2
        for i in range(1, 2 * limit + 2):
            diff[i] += diff[i - 1]
        return min(diff[2:2 * limit + 1])

    @staticmethod
    def lc_1738(matrix: List[List[int]], k: int) -> int:
        """
        url: https://leetcode.cn/problems/find-kth-largest-xor-coordinate-value/
        tag: matrix_prefix_xor_sum|classical
        """
        mat = PreFixXorMatrix(matrix)
        m, n = len(matrix), len(matrix[0])
        lst = []
        for i in range(m):
            lst.extend(mat.pre[i + 1][1:])
        lst.sort()
        return lst[-k]

    @staticmethod
    def lc_2132(grid: List[List[int]], h: int, w: int) -> bool:
        """
        url: https://leetcode.cn/problems/stamping-the-grid/
        tag: prefix_sum|brute_force|diff_matrix|implemention|classical|prefix_sum_prefix_sum
        """
        m, n = len(grid), len(grid[0])
        pre = PreFixSumMatrix(grid)
        dp = [[0] * n for _ in range(m)]
        for i in range(m - h + 1):
            for j in range(n - w + 1):
                cur = pre.query(i, j, i + h - 1, j + w - 1)
                if cur == 0:
                    dp[i][j] = 1

        pre = PreFixSumMatrix(dp)
        for i in range(m):
            for j in range(n):
                if not grid[i][j]:
                    x = i - h + 1 if i - h + 1 > 0 else 0
                    y = j - w + 1 if j - w + 1 > 0 else 0
                    cur = pre.query(x, y, i, j)
                    if not cur:
                        return False
        return True

    @staticmethod
    def ac_3993(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/3996/
        tag: suffix_sum|data_range|brain_teaser|classical|hard
        """
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        low = min(nums)
        high = max(nums)
        if low == high:
            ac.st(0)
            return

        cnt = [0] * (high - low + 1)
        for num in nums:
            cnt[num - low] += 1
        ans = post_cnt = post_sum = 0
        for i in range(high - low, 0, -1):
            post_cnt += cnt[i]
            post_sum += post_cnt
            if post_sum > k:
                ans += 1
                post_sum = post_cnt
        ans += post_sum > 0
        ac.st(ans)
        return

    @staticmethod
    def lc_837(n: int, k: int, max_pts: int) -> float:
        """
        url: https://leetcode.cn/problems/new-21-game/description/
        tag: diff_array|implemention|probability|refresh_table|classical
        """
        s = k + max_pts
        dp = [0] * (s + 1)
        dp[0] = 1
        diff = [0] * (s + 1)
        for i in range(s + 1):
            diff[i] += diff[i - 1]
            dp[i] += diff[i]
            if i < k:
                diff[i + 1] += dp[i]
                diff[i + max_pts + 1] -= dp[i]
        return sum(dp[k:n + 1]) / max_pts

    @staticmethod
    def lg_p6070(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6070
        tag: diff_array|matrix_diff_array|flatten
        """
        n, m, k = ac.read_list_ints()
        grid = [0] * (n * n)
        for _ in range(m):
            x, y, z = ac.read_list_ints_minus_one()
            grid[x * n + y] = z + 1
        diff = [0] * ((n + 2) * (n + 2))

        ans = 0
        for i in range(n):
            for j in range(n):
                val = diff[(i + 1) * (n + 2) + j] + diff[i * (n + 2) + j + 1] - diff[i * (n + 2) + j]
                diff[(i + 1) * (n + 2) + j + 1] += val
                d = diff[(i + 1) * (n + 2) + j + 1] + grid[i * n + j]
                if d:
                    if i + k + 1 > n + 1 or j + k + 1 > n + 1:
                        ac.st(-1)
                        return
                    diff[(i + 1) * (n + 2) + j + 1] -= d
                    diff[(i + 1) * (n + 2) + j + k + 1] += d
                    diff[(i + k + 1) * (n + 2) + j + 1] += d
                    diff[(i + k + 1) * (n + 2) + j + k + 1] -= d
                    ans += abs(d)
        ac.st(ans)
        return

    @staticmethod
    def lc_2983(s: str, queries: List[List[int]]) -> List[bool]:
        """
        url: https://leetcode.cn/problems/palindrome-rearrangement-queries/
        tag: brain_teaser|prefix_sum|brute_force|range_intersection
        """
        lst = [ord(w) - ord("a") for w in s]
        n = len(lst)
        lst1, lst2 = lst[:n // 2], lst[n // 2:][::-1]
        pre1 = []
        pre2 = []
        for i in range(26):
            pre1.append(list(accumulate([int(w == i) for w in lst1], initial=0)))
            pre2.append(list(accumulate([int(w == i) for w in lst2], initial=0)))

        right = [0] * (n // 2 + 1)
        right[n // 2] = 1
        for i in range(n // 2 - 1, -1, -1):
            if lst1[i] == lst2[i]:
                right[i] = 1
            else:
                break

        left = [0] * (n // 2 + 1)
        left[0] = 1
        for i in range(n // 2):
            if lst1[i] == lst2[i]:
                left[i + 1] = 1
            else:
                break

        ans = []
        for a, b, c, d in queries:
            c, d = n - 1 - d, n - 1 - c
            cc = min(a, c)
            dd = max(b, d)
            if not left[cc] or not right[dd + 1]:
                ans.append(False)
                continue
            if a <= c <= d <= b or c <= a <= b <= d:
                c, d = cc, dd
                cur1 = [pre1[j][d + 1] - pre1[j][c] for j in range(26)]
                cur2 = [pre2[j][d + 1] - pre2[j][c] for j in range(26)]
                if cur1 != cur2:
                    ans.append(False)
                else:
                    ans.append(True)
            elif b < c or d < a:
                cur1 = [pre1[j][b + 1] - pre1[j][a] for j in range(26)]
                cur2 = [pre2[j][b + 1] - pre2[j][a] for j in range(26)]
                if cur1 != cur2:
                    ans.append(False)
                    continue
                cur1 = [pre1[j][d + 1] - pre1[j][c] for j in range(26)]
                cur2 = [pre2[j][d + 1] - pre2[j][c] for j in range(26)]
                if cur1 != cur2:
                    ans.append(False)
                    continue
                if b < c:
                    c, d = b + 1, c - 1
                else:
                    c, d = d + 1, a - 1
                cur1 = [pre1[j][d + 1] - pre1[j][c] for j in range(26)]
                cur2 = [pre2[j][d + 1] - pre2[j][c] for j in range(26)]
                if cur1 != cur2:
                    ans.append(False)
                else:
                    ans.append(True)
            elif b <= d:
                cur1 = [pre1[j][b + 1] - pre1[j][a] for j in range(26)]
                cur2 = [pre2[j][d + 1] - pre2[j][c] for j in range(26)]
                x, y = a, c - 1
                cur = [pre2[j][y + 1] - pre2[j][x] for j in range(26)]
                cur1 = [cur1[j] - cur[j] for j in range(26)]

                x, y = b + 1, d
                cur = [pre1[j][y + 1] - pre1[j][x] for j in range(26)]
                cur2 = [cur2[j] - cur[j] for j in range(26)]

                if cur1 != cur2 or any(x < 0 for x in cur1 + cur2):
                    ans.append(False)
                else:
                    ans.append(True)
            else:
                cur1 = [pre1[j][b + 1] - pre1[j][a] for j in range(26)]
                cur2 = [pre2[j][d + 1] - pre2[j][c] for j in range(26)]
                x, y = d + 1, b
                cur = [pre2[j][y + 1] - pre2[j][x] for j in range(26)]
                cur1 = [cur1[j] - cur[j] for j in range(26)]

                x, y = c, a - 1
                cur = [pre1[j][y + 1] - pre1[j][x] for j in range(26)]
                cur2 = [cur2[j] - cur[j] for j in range(26)]

                if cur1 != cur2 or any(x < 0 for x in cur1 + cur2):
                    ans.append(False)
                else:
                    ans.append(True)
        return ans

    @staticmethod
    def lc_3017(n: int, x: int, y: int) -> List[int]:
        """
        url: https://leetcode.com/problems/count-the-number-of-houses-at-a-certain-distance-ii/description/
        tag: diff_array|classical
        """

        def update():
            if 0 <= low <= high:
                cnt[low] += 1
                if high + 1 <= n:
                    cnt[high + 1] -= 1
            return

        cnt = [0] * (n + 1)
        x -= 1
        y -= 1
        if x > y:
            x, y = y, x
        if abs(x - y) <= 1:
            for i in range(n):
                low = 0
                high = i
                update()

                low = 0
                high = n - 1 - i
                update()
            for i in range(1, n + 1):
                cnt[i] += cnt[i - 1]
            return cnt[1:]

        for i in range(x + 1):
            # 左边
            if i + 1 <= x:
                low = i + 1 - i
                high = x - i
                update()  # [i+1, x]

            if y <= n - 1:
                low = x - i + 1 + y - y  # [y, n-1]
                high = x - i + 1 + n - 1 - y
                update()

            j = (x + y + 1) // 2
            # [x+1, j]
            if x + 1 <= j < y:
                low = x + 1 - i
                high = j - i
                update()

            # [j+1, y-1]
            if j + 1 <= y - 1:
                low = x - i + 1 + y - (y - 1)
                high = x - i + 1 + y - (j + 1)
                update()

        # 右边
        for i in range(y, n):
            # [i+1, n-1]
            low = i + 1 - i
            high = n - 1 - i
            update()

        # 中间
        for i in range(x + 1, y):

            # [y, n-1]
            if y <= n - 1:
                right = min(y - i, i - x + 1)
                low = right + y - y
                high = right + n - 1 - y
                update()

            j = min((2 * i + y - x + 1) // 2, y - 1)
            # [i+1, j]
            if i + 1 <= j < y:
                low = 1
                high = j - i
                update()
            # [j+1, y-1]
            if j + 1 <= y - 1:
                low = i - x + 1 + y - (y - 1)
                high = i - x + 1 + y - (j + 1)
                update()

        for i in range(1, n + 1):
            cnt[i] += cnt[i - 1]
        return [x * 2 for x in cnt[1:]]

    @staticmethod
    def abc_331d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc331/tasks/abc331_d
        tag: prefix_sum_matrix|circular_section
        """
        n, q = ac.read_list_ints()
        grid = []
        for i in range(n):
            grid.append([int(w == "B") for w in ac.read_str()])
        pre = PreFixSumMatrix(grid)

        def check(x, y):
            aa = x // n
            bb = y // n
            res = aa * bb * pre.query(0, 0, n - 1, n - 1)
            if x % n:
                block = pre.query(0, 0, (x % n) - 1, n - 1)
                res += block * bb
                if y % n:
                    res += pre.query(0, 0, (x % n) - 1, (y % n) - 1)
            if y % n:
                block = pre.query(0, 0, n - 1, (y % n) - 1)
                res += block * aa
            return res

        for _ in range(q):
            a, b, c, d = ac.read_list_ints()
            ans = check(c + 1, d + 1) - check(c + 1, b) - check(a, d + 1) + check(a, b)
            ac.st(ans)
        return

    @staticmethod
    def abc_288d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc288/tasks/abc288_d
        tag: diff_array|brain_teaser|classical
        """
        n, k = ac.read_list_ints()
        a = [0] + ac.read_list_ints()
        diff = [0] * (n + 1)
        for i in range(1, n + 1):
            diff[i] = a[i] - a[i - 1]

        pre = []
        for i in range(k):
            lst = diff[:]
            for j in range(n + 1):
                if j % k != i:
                    lst[j] = 0
            pre.append(ac.accumulate(lst))

        for _ in range(ac.read_int()):
            ll, rr = ac.read_list_ints()
            for j in range(k):
                v = pre[j][rr + 1] - pre[j][ll]
                if j == ll % k:
                    v += a[ll - 1]
                if j == (rr + 1) % k:
                    continue
                if v:
                    ac.no()
                    break
            else:
                ac.yes()
        return

    @staticmethod
    def abc_347f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc347/tasks/abc347_f
        tag: diff_array|matrix_prefix_sum|matrix_rotate|brute_force|implemention
        """
        n, m = ac.read_list_ints()
        matrix = [ac.read_list_ints() for _ in range(n)]
        ans = -inf
        for _ in range(4):
            n = len(matrix)
            for i in range(n // 2):
                for j in range((n + 1) // 2):
                    a, b, c, d = matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1], matrix[i][j]
                    matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1] = a, b, c, d

            pre = PreFixSumMatrix(matrix)
            left = [-inf] * (n + 1) * (n + 1)
            for i in range(n):
                for j in range(n):
                    if i - m + 1 >= 0 and j - m + 1 >= 0:
                        left[(i + 1) * (n + 1) + j + 1] = max(left[i * (n + 1) + j + 1], left[(i + 1) * (n + 1) + j],
                                                              left[i * (n + 1) + j],
                                                              pre.query(i - m + 1, j - m + 1, i, j))
                    else:
                        left[(i + 1) * (n + 1) + j + 1] = max(left[i * (n + 1) + j + 1], left[(i + 1) * (n + 1) + j],
                                                              left[i * (n + 1) + j])

            right = [-inf] * (n + 1) * (n + 1)
            for i in range(n):
                for j in range(n - 1, -1, -1):
                    if i - m + 1 >= 0 and j + m - 1 < n:
                        right[(i + 1) * (n + 1) + j] = max(right[(i + 1) * (n + 1) + j + 1], right[i * (n + 1) + j],
                                                           right[i * (n + 1) + j + 1],
                                                           pre.query(i - m + 1, j, i, j + m - 1))
                    else:
                        right[(i + 1) * (n + 1) + j] = max(right[(i + 1) * (n + 1) + j + 1], right[i * (n + 1) + j],
                                                           right[i * (n + 1) + j + 1])

            down = [-inf] * (n + 1)
            for i in range(n - 1, -1, -1):
                if i + m - 1 < n:
                    down[i] = max(down[i + 1], max(pre.query(i, j, i + m - 1, j + m - 1) for j in range(n - m + 1)))

            up = [-inf] * (n + 1)
            for i in range(n):
                if i - m + 1 >= 0:
                    up[i + 1] = max(up[i], max(pre.query(i - m + 1, j, i, j + m - 1) for j in range(n - m + 1)))

            for i in range(n):
                for j in range(n):
                    cur = down[i + 1] + left[(i + 1) * (n + 1) + j + 1] + right[(i + 1) * (n + 1) + j + 1]
                    ans = max(ans, cur)
            for i in range(n):
                if i - m + 1 >= 0:
                    cur = max(pre.query(i - m + 1, j, i, j + m - 1) for j in range(n - m + 1)) + up[i - m + 1] + down[
                        i + 1]
                    ans = max(ans, cur)

        ac.st(ans)
        return

    @staticmethod
    def abc_274f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc274/tasks/abc274_f
        tag: brute_force|brain_teaser|discretization_diff_array|classical
        """
        n, a = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        ans = 0
        for i in range(n):
            w, x, v = nums[i]
            dct = defaultdict(int)
            cur = 0
            lst = []
            for j, (ww, xx, vv) in enumerate(nums):
                if vv == v:
                    if x <= xx <= x + a:
                        cur += ww
                    continue
                ceil = (x + a - xx) / (vv - v)
                floor = (x - xx) / (vv - v)
                if vv - v < 0:
                    floor, ceil = ceil, floor
                floor = max(0, floor)
                if floor <= ceil:
                    lst.append([floor, ceil, ww])

            nodes = [inf]
            for ls in lst:
                nodes.extend(ls[:-1])
            nodes = sorted(set(nodes))
            ind = {num: i for i, num in enumerate(nodes)}
            for aa, bb, ww in lst:
                dct[ind[aa]] += ww
                dct[ind[bb] + 1] -= ww
            m = len(nodes)
            for j in range(1, m):
                dct[j] += dct[j - 1]
            ans = max(ans, cur + max(dct.values()) if dct else cur)
        ac.st(ans)
        return

    @staticmethod
    def abc_269f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc269/tasks/abc269_f
        tag: diff_array|inclusion_exclusion|prefix_sum|math|classical
        """
        n, m = ac.read_list_ints()
        mod = 998244353

        def compute(start, diff, cnt):
            return (start + start + diff * (cnt - 1)) * cnt // 2

        def check(x, y):

            if not x or not y:
                return 0

            odd_start = (y + y % 2) * ((y + 1) // 2) // 2
            odd_diff = ((y + 1) // 2) * 2 * m
            odd_cnt = (x + 1) // 2

            even_start = m * (y // 2) + y * (y + 1) // 2 - odd_start
            even_diff = (y // 2) * 2 * m
            even_cnt = x // 2
            res = compute(odd_start, odd_diff, odd_cnt) + compute(even_start, even_diff, even_cnt)
            return res % mod

        for _ in range(ac.read_int()):
            a, b, c, d = ac.read_list_ints()
            ans = check(b, d) - check(b, c - 1) - check(a - 1, d) + check(a - 1, c - 1)
            ac.st(ans % mod)
        return

    @staticmethod
    def abc_268e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc268/tasks/abc268_e
        tag: brute_force|diff_array|action_scope|brain_teaser|classical
        """
        n = ac.read_int()
        tmp = ac.read_list_ints()
        diff = [0] * n

        def range_add(a, b, c):
            diff[a] += c
            if b + 1 < n:
                diff[b + 1] -= c
            return

        mid = n // 2
        for i in range(n):
            x = tmp[i]
            if x >= i:
                d = x - i
            else:
                d = n + x - i
            if d > mid:
                range_add(0, 0, n - d)
                if n % 2:
                    if 1 <= d - mid - 1:
                        range_add(1, d - mid - 1, 1)
                else:
                    range_add(1, d - mid, 1)
                range_add(d - mid + 1, d, -1)
                if d + 1 < n:
                    range_add(d + 1, n - 1, 1)
            elif d == mid:
                range_add(0, 0, mid)
                range_add(1, d, -1)
                range_add(d + 1, n - 1, 1)
            elif d + mid < n - 1:
                range_add(0, 0, d)
                range_add(1, d, -1)
                range_add(d + 1, d + mid, 1)
                if n % 2:
                    if d + mid + 2 < n:
                        range_add(d + mid + 2, n - 1, -1)
                else:
                    range_add(d + mid + 1, n - 1, -1)
            else:
                range_add(d + 1, d + mid, 1)
                range_add(0, 0, d)
                range_add(1, d, -1)
        res = ac.accumulate(ac.accumulate(diff)[1:])[1:]
        ac.st(min(res))
        return

    @staticmethod
    def abc_268e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc268/tasks/abc268_e
        tag: brute_force|diff_array|action_scope|brain_teaser|classical
        """
        n = ac.read_int()
        tmp = ac.read_list_ints()
        diff = [0] * n

        def range_add(a, b, c):
            diff[a] += c
            if b + 1 < n:
                diff[b + 1] -= c
            return

        mid = n // 2
        for i in range(n):
            x = tmp[i]
            if x >= i:
                d = x - i
            else:
                d = n + x - i
            if d > mid:
                range_add(0, 0, n - d)
                if n % 2:
                    if 1 <= d - mid - 1:
                        range_add(1, d - mid - 1, 1)
                else:
                    range_add(1, d - mid, 1)
                range_add(d - mid + 1, d, -1)
                if d + 1 < n:
                    range_add(d + 1, n - 1, 1)
            elif d == mid:
                range_add(0, 0, mid)
                range_add(1, d, -1)
                range_add(d + 1, n - 1, 1)
            elif d + mid < n - 1:
                range_add(0, 0, d)
                range_add(1, d, -1)
                range_add(d + 1, d + mid, 1)
                if n % 2:
                    if d + mid + 2 < n:
                        range_add(d + mid + 2, n - 1, -1)
                else:
                    range_add(d + mid + 1, n - 1, -1)
            else:
                range_add(d + 1, d + mid, 1)
                range_add(0, 0, d)
                range_add(1, d, -1)
        res = ac.accumulate(ac.accumulate(diff)[1:])[1:]
        ac.st(min(res))
        return

    @staticmethod
    def abc_260e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc260/tasks/abc260_e
        tag: diff_array|action_scope|two_pointer|hash|classical
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(m)]
        for i in range(n):
            x, y = ac.read_list_ints_minus_one()
            dct[x].append(i)
            dct[y].append(i)
        cnt = [0] * n
        tot = 0
        ans = 0
        diff = [0] * (m + 2)
        j = 0
        for i in range(m):
            while j < m and tot < n:
                for x in dct[j]:
                    cnt[x] += 1
                    if cnt[x] == 1:
                        tot += 1
                j += 1
            if tot == n:
                low = j - i
                high = m - i
                diff[low] += 1
                diff[high + 1] -= 1
            for x in dct[i]:
                cnt[x] -= 1
                if not cnt[x]:
                    tot -= 1
        diff = ac.accumulate(diff[1:])
        ac.lst(diff[1:m + 1])
        return

    @staticmethod
    def abc_210d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc210/tasks/abc210_d
        tag: prefix_max|matrix_prefix|classical
        """
        m, n, c = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        ans = inf
        pre = [[inf] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                p = min(pre[i + 1][j], pre[i][j + 1])
                ans = min(ans, p + c * i + c * j + grid[i][j])
                pre[i + 1][j + 1] = min(p, grid[i][j] - c * i - c * j)

        pre = [[inf] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n - 1, -1, -1):
                p = min(pre[i][j], pre[i + 1][j + 1])
                ans = min(ans, p + c * i - c * j + grid[i][j])
                pre[i + 1][j] = min(p, grid[i][j] - c * i + c * j)
        ac.st(ans)
        return

    @staticmethod
    def cf_1985h1(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1985/problem/H1
        tag: union_find|contribution_method|diff_matrix|brain_teaser
        """
        for _ in range(ac.read_int()):
            m, n = ac.read_list_ints()
            grid = [ac.read_str() for _ in range(m)]
            uf = UnionFindGeneral(m * n)
            row = [0] * m
            col = [0] * n
            for i in range(m):
                for j in range(n):
                    if grid[i][j] == "#":
                        uf.size[i * n + j] = 1
                    else:
                        row[i] += 1
                        col[j] += 1
            for i in range(m):
                for j in range(n):
                    if grid[i][j] == "#":
                        if i + 1 < m and grid[i + 1][j] == "#":
                            uf.union(i * n + j, i * n + j + n)
                        if j + 1 < n and grid[i][j + 1] == "#":
                            uf.union(i * n + j, i * n + j + 1)
            group = uf.get_root_part()
            row_diff = [0] * (m + 1)
            col_diff = [0] * (n + 1)
            for g in group:
                if uf.size[g]:
                    x1 = max(min(x // n for x in group[g]) - 1, 0)
                    x2 = min(max(x // n for x in group[g]) + 1, m - 1)
                    y1 = max(min(x % n for x in group[g]) - 1, 0)
                    y2 = min(max(x % n for x in group[g]) + 1, n - 1)
                    row_diff[x1] += uf.size[g]
                    row_diff[x2 + 1] -= uf.size[g]
                    col_diff[y1] += uf.size[g]
                    col_diff[y2 + 1] -= uf.size[g]
            ans = 0
            for i in range(m):
                row_diff[i] += row_diff[i - 1] if i else 0
                ans = max(row[i] + row_diff[i], ans)
            for j in range(n):
                col_diff[j] += col_diff[j - 1] if j else 0
                ans = max(col[j] + col_diff[j], ans)
            ac.st(ans)
        return

    @staticmethod
    def cf_1985h2(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1985/problem/H2
        tag: union_find|contribution_method|diff_matrix|brain_teaser
        """
        for _ in range(ac.read_int()):
            m, n = ac.read_list_ints()
            grid = [ac.read_str() for _ in range(m)]
            uf = UnionFindGeneral(m * n)
            row = [0] * m
            col = [0] * n
            for i in range(m):
                for j in range(n):
                    if grid[i][j] == "#":
                        uf.size[i * n + j] = 1
                    else:
                        row[i] += 1
                        col[j] += 1
            for i in range(m):
                for j in range(n):
                    if grid[i][j] == "#":
                        if i + 1 < m and grid[i + 1][j] == "#":
                            uf.union(i * n + j, i * n + j + n)
                        if j + 1 < n and grid[i][j + 1] == "#":
                            uf.union(i * n + j, i * n + j + 1)
            group = uf.get_root_part()
            lst = []
            for g in group:
                if uf.size[g]:
                    x1 = max(min(x // n for x in group[g]) - 1, 0)
                    x2 = min(max(x // n for x in group[g]) + 1, m - 1)
                    y1 = max(min(x % n for x in group[g]) - 1, 0)
                    y2 = min(max(x % n for x in group[g]) + 1, n - 1)
                    lst.append((x1, x2, y1, y2, -uf.size[g]))
                    lst.append((x1, x2, 0, n - 1, uf.size[g]))
                    lst.append((0, m - 1, y1, y2, uf.size[g]))
            res = DiffMatrix().get_diff_matrix3(m, n, lst)
            ans = max(max(row[i] + col[j] - (grid[i][j] == ".") + res[i][j] for j in range(n)) for i in range(m))
            ac.st(ans)
        return

    @staticmethod
    def lg_p3016(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3016
        tag: prefix_sum|triangle|left_up_sum|inclusion_exclusion
        """
        n, k = ac.read_list_ints()
        grid = []
        for i in range(n):
            lst = ac.read_list_ints() + [0] * (n - 1 - i)
            grid.append(lst)
        pre = PreFixSumMatrix(grid)
        left_up = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n):
            cur = 0
            for j in range(i + 1):
                cur += grid[i][j]
                left_up[i + 1][j + 1] = left_up[i][j] + cur
        ans = -inf
        for x in range(k, 2 * k + 1):
            cnt = x * (x + 1) // 2
            for i in range(x - 1, n):
                for j in range(i - x + 2):
                    cur = left_up[i + 1][j + x] - left_up[i - x + 1][j]
                    cur -= pre.query(i - x + 1, 0, i, j - 1) if j else 0
                    ans = max(ans, int(cur / cnt))

        for x in range(k, 2 * k + 1):
            cnt = x * (x + 1) // 2
            for i in range(x - 1, n - x + 1):
                for j in range(x - 1, i + 1):
                    cur = pre.query(i, 0, i + x - 1, j)
                    cur -= left_up[i + x][j]
                    cur += left_up[i][j - x] if j >= x else 0
                    ans = max(ans, int(cur / cnt))
        ac.st(ans)
        return