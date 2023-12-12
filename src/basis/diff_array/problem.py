"""

Algorithm：diff_array|prefix_sum|suffix_sum|prefix_max_consequence_sum|suffix_max_consequence_sum|diff_matrix|discretization_diff_array|md_diff_array|matrix_prefix_sum
Description：prefix_sum|prefix_sum_of_prefix_sum|suffix_sum

====================================LeetCode====================================
152（https://leetcode.com/problems/maximum-product-subarray/）prefix_mul|maximum_sub_consequence_product
598（https://leetcode.com/problems/range-addition-ii/）diff_matrix
2281（https://leetcode.com/problems/sum-of-total-strength-of-wizards/）brute_force|prefix_sum_of_prefix_sum
2251（https://leetcode.com/problems/number-of-flowers-in-full-bloom/）discretization_diff_array
2132（https://leetcode.com/problems/stamping-the-grid/）prefix_sum|brute_force|diff_matrix|implemention
1229（https://leetcode.com/problems/meeting-scheduler/）discretization_diff_array
6292（https://leetcode.com/problems/increment-submatrices-by-one/）diff_matrix|prefix_sum
2565（https://leetcode.com/problems/subsequence-with-the-minimum-score/）prefix_suffix|pointer|brute_force
644（https://leetcode.com/problems/maximum-average-subarray-ii/）prefix_sum|binary_search|average
1292（https://leetcode.com/problems/maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold/）O(mn)|brute_force
1674（https://leetcode.com/problems/minimum-moves-to-make-array-complementary/）diff_array|action_scope|counter
1714（https://leetcode.com/problems/sum-of-special-evenly-spaced-elements-in-array/）prefix_sum
1738（https://leetcode.com/problems/find-kth-largest-xor-coordinate-value/）matrix_prefix_xor_sum
1895（https://leetcode.com/problems/largest-magic-square/）matrix_prefix_sum|brute_force
1943（https://leetcode.com/problems/describe-the-painting/）discretization_diff_array
2021（https://leetcode.com/problems/brightest-position-on-street/）discretization_diff_array
837（https://leetcode.com/problems/new-21-game/description/）diff_array|implemention|probability
891（https://leetcode.com/problems/sum-of-subsequence-widths/description/）prefix_suffix|brute_force|counter
1191（https://leetcode.com/problems/k-concatenation-maximum-sum/description/）prefix_suffix|max_sub_consequence_sum
1074（https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/description/）matrix_prefix_sum|brute_force
1139（https://leetcode.com/problems/largest-1-bordered-square/）matrix_prefix_sum|counter|brute_force
2281（https://leetcode.com/problems/sum-of-total-strength-of-wizards/description/）monotonic_stack|counter|prefix_sum_of_prefix_sum
995（https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/description/）greedy|diff_array|implemention
986（https://leetcode.com/problems/interval-list-intersections/description/）discretization_diff_array|two_pointers
1744（https://leetcode.com/problems/can-you-eat-your-favorite-candy-on-your-favorite-day/description/）prefix_sum|greedy|implemention
1703（https://leetcode.com/problems/minimum-adjacent-swaps-for-k-consecutive-ones/）prefix_sum|median|greedy|1520E
2167（https://leetcode.com/problems/minimum-time-to-remove-all-cars-containing-illegal-goods/）math|prefix_sum|brute_force

=====================================LuoGu======================================
8772（https://www.luogu.com.cn/record/list?user=739032&status=12&page=15）suffix_sum
2367（https://www.luogu.com.cn/problem/P2367）diff_array
2280（https://www.luogu.com.cn/problem/P2280）matrix_prefix_sum
3138（https://www.luogu.com.cn/problem/P3138）matrix_prefix_sum
3406（https://www.luogu.com.cn/problem/P3406）diff_array|greedy
3655（https://www.luogu.com.cn/problem/P3655）diff_array|implemention
5542（https://www.luogu.com.cn/problem/P5542）diff_matrix
5686（https://www.luogu.com.cn/problem/P5686）prefix_sum_of_prefix_sum
6180（https://www.luogu.com.cn/problem/P6180）prefix_sum|counter
6481（https://www.luogu.com.cn/problem/P6481）prefix|range_update
2956（https://www.luogu.com.cn/problem/P2956）diff_matrix|prefix_sum
3397（https://www.luogu.com.cn/problem/P3397）diff_matrix|prefix_sum
1869（https://www.luogu.com.cn/problem/P1869）prefix_sum|comb|odd_even
7667（https://www.luogu.com.cn/problem/P7667）math|sorting|prefix_sum
2671（https://www.luogu.com.cn/problem/P2671）prefix_or_sum|counter|brute_force|odd_even
1719（https://www.luogu.com.cn/problem/P1719）max_sub_matrix_sum|brute_force|prefix_sum
2882（https://www.luogu.com.cn/problem/P2882）greedy|brute_force|diff_array
4552（https://www.luogu.com.cn/problem/P4552）diff_array|brain_teaser|classical
1627（https://www.luogu.com.cn/problem/P1627）prefix_suffix|median|counter
1895（https://www.luogu.com.cn/problem/P1895）prefix_sum|counter|binary_search
1982（https://www.luogu.com.cn/problem/P1982）maximum_prefix_sub_consequence_sum|prefix_max
2070（https://www.luogu.com.cn/problem/P2070）hash|discretization_diff_array|counter
2190（https://www.luogu.com.cn/problem/P2190）diff_array|circular_array
2352（https://www.luogu.com.cn/problem/P2352）discretization_diff_array
2363（https://www.luogu.com.cn/problem/P2363）matrix_prefix_sum|brute_force
2706（https://www.luogu.com.cn/problem/P2706）max_sub_matrix_sum
2879（https://www.luogu.com.cn/problem/P2879）diff_array|greedy
3028（https://www.luogu.com.cn/problem/P3028）discretization_diff_array|range_cover
4030（https://www.luogu.com.cn/problem/P4030）brain_teaser|matrix_prefix_sum
4440（https://www.luogu.com.cn/problem/P4440）prefix_sum|counter
4623（https://www.luogu.com.cn/problem/P4623）discretization_diff_array|counter
6032（https://www.luogu.com.cn/problem/P6032）prefix_suffix|counter
6070（https://www.luogu.com.cn/problem/P6070）diff_matrix|greedy|prefix_sum
6278（https://www.luogu.com.cn/problem/P6278）reverse_order_pair|action_scope|diff_array|prefix_sum
6537（https://www.luogu.com.cn/problem/P6537）prefix_sum|brute_force
6877（https://www.luogu.com.cn/problem/P6877）sorting|greedy|prefix_suffix|dp|brute_force
6878（https://www.luogu.com.cn/problem/P6878）prefix_suffix|brute_force
8081（https://www.luogu.com.cn/problem/P8081）diff_array|counter|action_scope
8033（https://www.luogu.com.cn/problem/P8033）matrix_prefix_sum|counter
7992（https://www.luogu.com.cn/problem/P7992）bucket_counter|action_scope|diff_array|counter
7948（https://www.luogu.com.cn/problem/P7948）sorting|prefix_suffix|pointer
8343（https://www.luogu.com.cn/problem/P8343）sub_matrix_prefix_sum|brute_force|two_pointers
8551（https://www.luogu.com.cn/problem/P8551）diff_array
8666（https://www.luogu.com.cn/problem/P8666）binary_search|md_diff_array|implemention
8715（https://www.luogu.com.cn/problem/P8715）prefix_suffix|counter
8783（https://www.luogu.com.cn/problem/P8783）O(n^3)|two_pointers|brute_force|counter|sub_matrix

===================================CodeForces===================================
33C（https://codeforces.com/problemset/problem/33/C）prefix_suffix|brute_force
797C（https://codeforces.com/problemset/problem/797/C）suffix_min|lexicographical_order|implemention
75D（https://codeforces.com/problemset/problem/75/D）max_sub_consequence_sum|compress_arrays
1355C（https://codeforces.com/problemset/problem/1355/C）action_scope|diff_array|triangle
1795C（https://codeforces.com/problemset/problem/1795/C）prefix_sum|binary_search|diff_array|counter|implemention
1343D（https://codeforces.com/problemset/problem/1343/D）brute_force|diff_array|counter
1722E（https://codeforces.com/problemset/problem/1722/E）data_range|matrix_prefix_sum
1772D（https://codeforces.com/contest/1772/problem/D）discretization_diff_array|action_scope|counter

====================================AtCoder=====================================
D - AtCoder Express 2（https://atcoder.jp/contests/abc106/tasks/abc106_d）prefix_sum|dp|counter

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
from collections import defaultdict, deque
from itertools import accumulate
from math import inf
from typing import List

from src.basis.binary_search.template import BinarySearch
from src.basis.diff_array.template import DiffMatrix, PreFixSumMatrix
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p3397(ac=FastIO()):
        # diff_matrix|prefix_sum
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
        # diff_array|题，明晰差分本质
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

        # 求最大子矩阵和，brute_force矩阵上下边界并prefix_sum
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
        # 根据数字范围，二位prefix_sum，求解子矩阵元素和
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
        # prefix_suffixcounter|和，分odd_even讨论
        n, m = ac.read_list_ints()
        number = ac.read_list_ints()
        colors = ac.read_list_ints()
        mod = 10007
        # 转换为求相同odd_even的 x 与 y 且颜色相同的和 x*ax+x*az+z*ax+z*az 即 (x+z)*(ax+az)
        ans = 0
        pre_sum = [[0, 0] for _ in range(m + 1)]
        pre_cnt = [[0, 0] for _ in range(m + 1)]
        for i in range(n):  # brute_force z  z*ax+z*az
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
        for i in range(n - 1, -1, -1):  # brute_force x  x*ax+x*az
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
        # 根据action_scopediff_array|counter
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
        url: https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/description/
        tag: greedy|diff_array|implemention
        """
        # greedy|diff_array|implemention
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
        url: https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/description/
        tag: matrix_prefix_sum|brute_force
        """
        # matrix_prefix_sum|brute_force上下边目标子矩阵的数量
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
        url: https://leetcode.com/problems/k-concatenation-maximum-sum/description/
        tag: prefix_suffix|max_sub_consequence_sum
        """
        # prefix_suffix最大连续子序列和
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
        # 模板: action_scope差分，合法三角形边长个数
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
        url: https://leetcode.com/problems/sum-of-total-strength-of-wizards/description/
        tag: monotonic_stack|counter|prefix_sum_of_prefix_sum
        """
        # monotonic_stack|counter与prefix_sum的prefix_sum
        n = len(nums)
        post = [n - 1] * n  # 这里可以是n/n-1/null，取决于用途
        pre = [0] * n  # 这里可以是0/-1/null，取决于用途
        stack = []
        for i in range(n):  # 这里也可以是从n-1到0reverse_order|，取决于用途
            while stack and nums[stack[-1]
            ] > nums[i]:  # 这里可以是"<" ">" "<=" ">="，取决于需要判断的大小关系
                post[stack.pop()] = i - 1  # 这里可以是i或者i-1，取决于是否包含i作为右端点
            if stack:  # 这里不一定可以同时，比如前后都是大于等于时，只有前后所求范围互斥时，可以
                # 这里可以是stack[-1]或者stack[-1]+1，取决于是否包含stack[-1]作为左端点
                pre[i] = stack[-1] + 1
            stack.append(i)
        mod = 10 ** 9 + 7
        s = list(accumulate(nums, initial=0))
        ss = list(accumulate(s, initial=0))
        ans = 0
        for i in range(n):
            left = pre[i]
            right = post[i]
            ans += nums[i] * ((i - left + 1) * (ss[right + 2] - \
                                                ss[i + 1]) - (right - i + 1) * (ss[i + 1] - ss[left]))
            ans %= mod
        return ans

    @staticmethod
    def lc_2281(strength: List[int]) -> int:
        """
        url: https://leetcode.com/problems/sum-of-total-strength-of-wizards/description/
        tag: monotonic_stack|counter|prefix_sum_of_prefix_sum
        """
        # monotonic_stack|确定|和范围，再prefix_sum的prefix_sumcounter
        n = len(strength)
        mod = 10 ** 9 + 7

        # monotonic_stack|
        pre = [0] * n
        post = [n - 1] * n
        stack = []
        for i in range(n):
            while stack and strength[stack[-1]] >= strength[i]:
                post[stack.pop()] = i - 1
            if stack:
                pre[i] = stack[-1] + 1
            stack.append(i)

        # prefix_sum
        s = [0] * (n + 1)
        for i in range(n):
            s[i + 1] = s[i] + strength[i]

        # prefix_sum的prefix_sum
        ss = [0] * (n + 2)
        for i in range(n + 1):
            ss[i + 1] = ss[i] + s[i]

        # 遍历|和
        ans = 0
        for i in range(n):
            left, right = pre[i], post[i]
            ans += strength[i] * ((i - left + 1) * (ss[right + 2] -
                                                    ss[i + 1]) - (right - i + 1) * (ss[i + 1] - ss[left]))
            ans %= mod
        return ans

    @staticmethod
    def lc_2565(s: str, t: str) -> int:
        """
        url: https://leetcode.com/problems/subsequence-with-the-minimum-score/
        tag: prefix_suffix|pointer|brute_force
        """
        # prefix_suffixgreedybrute_forceprefix_suffix最长匹配
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

        # greedybrute_force|差分验证
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
        # discretization_diff_arrayaction_scopecounter
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
        # matrix_prefix_sum|
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

        # prefix_sum|binary_search不短于k的子数组最大平均值
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

        error = 1
        low = 0
        high = max(nums) * 1000
        while low < high - error:
            mid = low + (high - low) // 2
            if check(mid):
                low = mid
            else:
                high = mid
        ac.st(high if check(high) else low)
        return

    @staticmethod
    def ac_121(ac=FastIO()):
        # discretizationprefix_sum，two_pointers|binary_search
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
                grid[i][j] = grid[i - 1][j] + grid[i][j - 1] - \
                             grid[i - 1][j - 1] + grid[i][j]

        def check(xx):
            up = 0
            for i in range(m):
                while up < m and lst_x[up] - lst_x[i] <= xx - 1:
                    up += 1
                right = 0
                for j in range(n):
                    while right < n and lst_y[right] - lst_y[j] <= xx - 1:
                        right += 1
                    cur = grid[up][right] - grid[up][j] - \
                          grid[i][right] + grid[i][j]
                    if cur >= c:
                        return True

            return False

        low = 0
        high = 10000
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                high = mid
            else:
                low = mid
        ans = low if check(low) else high
        ac.st(ans)
        return

    @staticmethod
    def ac_126(ac=FastIO()):
        # 最大子矩形和
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
        # prefix_suffixmedian大小值差值counter
        n, b = ac.read_list_ints()
        nums = ac.read_list_ints()
        i = nums.index(b)

        # 前缀差值counter
        pre = defaultdict(int)
        cnt = ans = 0
        for j in range(i - 1, -1, -1):
            num = nums[j]
            cnt += 1 if num > b else -1
            pre[cnt] += 1
            if cnt == 0:  # 只取前缀
                ans += 1

        # 后缀差值counter
        cnt = 0
        for j in range(i + 1, n):
            num = nums[j]
            cnt += 1 if num > b else -1
            ans += pre[-cnt]  # 取prefix_suffix
            ans += 1 if not cnt else 0  # 只取后缀
        ans += 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p1895(ac=FastIO()):

        # prefix_sumcounter|binary_search，最多不超多10**5
        n = 10 ** 5
        dp = [0] * (n + 1)
        for i in range(1, n + 1):  # 序列1234..
            dp[i] = dp[i - 1] + len(str(i))

        # 序列1121231234..
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

        # 前缀最大连续子段和与前缀最大值
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
        # hashdiscretization_diff_arraycounter
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
        # 从小到大区间占有次数
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
        # circular_array|差分
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
        # discretization_diff_array|
        diff = defaultdict(int)
        for _ in range(ac.read_int()):
            a, b = ac.read_list_ints()
            diff[a] += 1
            diff[b + 1] -= 1
        axis = sorted(list(diff.keys()))
        m = len(axis)
        ans = 0
        for i in range(1, m):
            diff[axis[i]] += diff[axis[i - 1]]
            # 注意此时选择右端点
            ans = ac.max(ans, diff[axis[i - 1]] * (axis[i] - 1))
        ac.st(ans)
        return

    @staticmethod
    def lg_p2363(ac=FastIO()):
        # matrix_prefix_sum|与brute_force
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
        # 不包含障碍点的最大子矩阵和
        m, n = ac.read_list_ints()
        grid = []
        while len(grid) < m * n:
            grid.extend(ac.read_list_ints())
        ans = 0
        for i in range(m):
            lst = [0] * n
            for j in range(i, m):
                floor = pre = 0
                for k in range(n):
                    num = grid[j * n + k]
                    lst[k] += num if num != 0 else -inf
                    floor = 0 if floor < 0 else floor
                    pre = pre if pre > 0 else 0
                    num = lst[k]
                    pre += num
                    ans = ans if ans > pre - floor else pre - floor
        ac.st(ans)
        return

    @staticmethod
    def lg_p2879(ac=FastIO()):
        # diff_array|题与greedy
        n, i, h, r = ac.read_list_ints()
        diff = [0] * n
        pre = set()
        for _ in range(r):
            a, b = ac.read_list_ints_minus_one()
            if (a, b) in pre:
                continue
            pre.add((a, b))
            if a < b:
                diff[a + 1] -= 1
                diff[b] += 1
            else:
                diff[b + 1] -= 1
                diff[a] += 1
        for i in range(1, n):
            diff[i] += diff[i - 1]
        gap = h - diff[i]
        for d in diff:
            ac.st(d + gap)
        return

    @staticmethod
    def lg_p3028(ac=FastIO()):
        # discretization_diff_array|覆盖区间最多的点
        n = ac.read_int()
        diff = defaultdict(int)
        for _ in range(n):
            a, b = ac.read_list_ints()
            if a > b:
                a, b = b, a
            diff[a] += 1
            diff[b + 1] -= 1
            # 增|右端点避免discretization带来重合
            diff[b] += 0
        axis = sorted(list(diff.keys()))
        ans = diff[axis[0]]
        m = len(axis)
        for i in range(1, m):
            # 当前点覆盖的区间数
            diff[axis[i]] += diff[axis[i - 1]]
            ans = ac.max(ans, diff[axis[i]])
        ac.st(ans)
        return

    @staticmethod
    def lg_p4030(ac=FastIO()):
        # brain_teaser|matrix_prefix_sum|
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
            if pm.query(x + 1, y + 1, x + k - 1, y +
                                                 k - 1) == (k - 1) * (k - 1):
                ac.st("Y")
            else:
                ac.st("N")
        return

    @staticmethod
    def lg_p4440(ac=FastIO()):
        # classicalprefix_sumcounter
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
            if all(pre[b + 1][j] - pre[a][j] == pre[d + 1]
            [j] - pre[c][j] for j in range(26)):
                ac.st("DA")
            else:
                ac.st("NE")
        return

    @staticmethod
    def lg_p4623(ac=FastIO()):
        # discretization_diff_array|counter
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

        # 差分
        for i in range(1, m):
            diff_x[i] += diff_x[i - 1]
        for i in range(1, m):
            diff_y[i] += diff_y[i - 1]

        # 查询
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
        # prefix_suffixcounter
        n, k, p = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        post = [0] * k
        for i in range(n):
            post[nums[i][0]] += 1

        pre = dict()
        ans = ss = 0
        for i in range(n):
            cc, pp = nums[i]
            if pp <= p:
                # 直接清空，前序ss个
                ans += ss + post[cc] - 1
                ss = 0
                pre = dict()
                post[cc] -= 1
                continue

            pre[cc] = pre.get(cc, 0) + 0
            ss -= pre.get(cc, 0)
            pre[cc] += 1

            post[cc] -= 1
            ss += post[cc]
        ac.st(ans)
        return

    @staticmethod
    def lg_p6070(ac=FastIO()):
        # diff_matrix|greedy修改实时维护差分与prefix_sum即矩阵最新值
        n, m, k = ac.read_list_ints()
        grid = [[0] * n for _ in range(n)]
        for _ in range(m):
            x, y, z = ac.read_list_ints_minus_one()
            grid[x][y] = z + 1
        diff = [[0] * (n + 2) for _ in range(n + 2)]

        ans = 0
        for i in range(n):
            for j in range(n):
                diff[i + 1][j + 1] += diff[i + 1][j] + \
                                      diff[i][j + 1] - diff[i][j]
                d = diff[i + 1][j + 1] + grid[i][j]
                # diff_matrix|，索引从 0 开始， 注意这里的行列索引范围，是从左上角到右下角
                if d:
                    if i + k + 1 > n + 1 or j + k + 1 > n + 1:
                        ac.st(-1)
                        return
                    diff[i + 1][j + 1] -= d
                    diff[i + 1][j + k + 1] += d
                    diff[i + k + 1][j + 1] += d
                    diff[i + k + 1][j + k + 1] -= d
                    ans += abs(d)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6278(ac=FastIO()):
        # reverse_order_pair|action_scope与差分prefix_sum
        n = ac.read_int()
        nums = ac.read_list_ints()
        diff = [0] * (n + 1)
        pre = []
        for num in nums:
            # num 作为最小值时的reverse_order_pair|个数
            diff[num] += len(pre) - bisect.bisect_right(pre, num)
            bisect.insort_left(pre, num)
        diff = ac.accumulate(diff)
        for i in range(n):
            # 减少到 i 时前面小于 i 的对应reverse_order_pair|不受改变
            ac.st(diff[i])
        return

    @staticmethod
    def lg_p6537(ac=FastIO()):
        # preprocessprefix_sum|brute_force
        n = ac.read_int()
        grid = [ac.read_list_ints() for _ in range(n)]
        pre = PreFixSumMatrix(grid)
        ans = 0
        for i in range(n):
            for j in range(n):
                # 左上右下
                dct = dict()
                for x in range(i + 1):
                    for y in range(j + 1):
                        val = pre.query(x, y, i, j)
                        dct[val] = dct.get(val, 0) + 1
                for x in range(i + 1, n):
                    for y in range(j + 1, n):
                        val = pre.query(i + 1, j + 1, x, y)
                        ans += dct.get(val, 0)
                # 左下右上
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
        # sortinggreedyprefix_suffix DP brute_force
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
        # prefix_suffixbrute_force
        n, k = ac.read_list_ints()
        s = ac.read_str()
        pre = [-1] * n
        stack = deque()
        for i in range(n):
            while len(stack) > k:
                stack.popleft()
            if s[i] == "J":
                stack.append(i)
            elif s[i] == "O":
                if len(stack) == k:
                    pre[i] = stack[0]

        post_o = [-1] * n
        post = [-1] * n
        stack = deque()
        stack_o = deque()
        for i in range(n - 1, -1, -1):
            while len(stack) > k:
                stack.popleft()
            if s[i] == "I":
                stack.append(i)
            if s[i] == "O":
                stack_o.append(i)
            while len(stack_o) > k:
                stack_o.popleft()
            if s[i] == "O":
                if len(stack) == k:
                    post[i] = stack[0]
                if len(stack_o) == k:
                    post_o[i] = stack_o[0]

        ans = inf
        for i in range(n):
            if pre[i] != -1 and post_o[i] != -1 and post[post_o[i]] != -1:
                ans = ac.min(ans, post[post_o[i]] - pre[i] + 1 - 3 * k)
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def lg_p8081(ac=FastIO()):
        # 差分counteraction_scope
        n = ac.read_int()
        nums = ac.read_list_ints()
        diff = [0] * (n + 1)
        pre = 0
        ceil = 0
        for i in range(n):
            if nums[i] < 0:
                pre += 1
            else:
                if pre:
                    ceil = ac.max(ceil, pre)
                    low, high = ac.max(0, i - 3 * pre), i - pre - 1
                    if low <= high:
                        diff[low] += 1
                        diff[high + 1] -= 1
                pre = 0
        if pre:
            ceil = ac.max(ceil, pre)
            low, high = ac.max(0, n - 3 * pre), n - pre - 1
            if low <= high:
                diff[low] += 1
                diff[high + 1] -= 1

        diff = [int(num > 0) for num in ac.accumulate(diff)][1:n + 1]
        diff = ac.accumulate(diff)
        ans = diff[-1]
        pre = 0
        res = 0
        for i in range(n):
            if nums[i] < 0:
                pre += 1
            else:
                if pre == ceil:
                    low, high = i - 4 * pre, i - 3 * pre - 1
                    low = ac.max(low, 0)
                    if low <= high:
                        res = ac.max(res, high - low + 1 -
                                     diff[high + 1] + diff[low])
                pre = 0
        if pre == ceil:
            low, high = n - 4 * pre, n - 3 * pre - 1
            low = ac.max(low, 0)
            if low <= high:
                res = ac.max(res, high - low + 1 - diff[high + 1] + diff[low])
        ac.st(ans + res)
        return

    @staticmethod
    def lg_p8033(ac=FastIO()):
        # matrix_prefix_sum|counter
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
        # bucket_counter与action_scope差分counter
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
        # sorting后preprocessprefix_suffix信息pointer查询
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
        # 子矩阵prefix_sumbrute_force与two_pointers
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
                        ans = ac.min(
                            ans, abs(cur - lst[ind_a] - a) + abs(cur - lst[ind_a] - b))
                        ind_a += 1

                    while ind_b + 1 < j + 1 and cur - lst[ind_b] <= b:
                        ans = ac.min(
                            ans, abs(cur - lst[ind_b] - a) + abs(cur - lst[ind_b] - b))
                        ind_b += 1

                    ans = ac.min(
                        ans, abs(cur - lst[ind_a] - a) + abs(cur - lst[ind_a] - b))
                    ans = ac.min(
                        ans, abs(cur - lst[ind_b] - a) + abs(cur - lst[ind_b] - b))
                    if ans == b - a:
                        ac.st(ans)
                        return
        ac.st(ans)
        return

    @staticmethod
    def lg_p8551(ac=FastIO()):
        # diff_array|灵活应用
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
        # binary_search|md_diff_array题
        a, b, c, m = ac.read_list_ints()
        grid = [[[0] * (c + 1) for _ in range(b + 1)] for _ in range(a + 1)]
        nums = ac.read_list_ints()
        for i in range(1, a + 1):
            for j in range(1, b + 1):
                for k in range(1, c + 1):
                    grid[i][j][k] = nums[((i - 1) * b + (j - 1)) * c + k - 1]
        del nums
        lst = [ac.read_list_ints() for _ in range(m)]

        def check(x):
            # 三位差分
            diff = [[[0] * (c + 2) for _ in range(b + 2)]
                    for _ in range(a + 2)]
            for i1, i2, j1, j2, k1, k2, h in lst[:x]:
                # 差分值更新索引从 1 开始
                diff[i1][j1][k1] += h

                diff[i2 + 1][j1][k1] -= h
                diff[i1][j2 + 1][k1] -= h
                diff[i1][j1][k2 + 1] -= h

                diff[i2 + 1][j2 + 1][k1] += h
                diff[i1][j2 + 1][k2 + 1] += h
                diff[i2 + 1][j1][k2 + 1] += h

                diff[i2 + 1][j2 + 1][k2 + 1] -= h

            for i1 in range(1, a + 1):
                for j1 in range(1, b + 1):
                    for k1 in range(1, c + 1):
                        # prefix_sum索引从 1 开始
                        diff[i1][j1][k1] += diff[i1 - 1][j1][k1] + diff[i1][j1 - 1][k1] + diff[i1][j1][k1 - 1] - \
                                            diff[i1][j1 - 1][k1 - 1] - diff[i1 - 1][j1][k1 - 1] - diff[i1 - 1][j1 - 1][
                                                k1] + \
                                            diff[i1 - 1][j1 - 1][k1 - 1]
                        if diff[i1][j1][k1] > grid[i1][j1][k1]:
                            return True

            return False

        ans = BinarySearch().find_int_left(1, m, check)
        ac.st(ans)
        return

    @staticmethod
    def lc_891(nums: List[int]) -> int:
        """
        url: https://leetcode.com/problems/sum-of-subsequence-widths/description/
        tag: prefix_suffix|brute_force|counter
        """
        # prefix_suffixbrute_force最大值与最小值counter
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
        url: https://leetcode.com/problems/maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold/
        tag: O(mn)|brute_force
        """
        # O(mn)复杂度brute_force
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
        url: https://leetcode.com/problems/minimum-moves-to-make-array-complementary/
        tag: diff_array|action_scope|counter
        """
        # diff_array|action_scopecounter
        n = len(nums)
        diff = [0] * (2 * limit + 2)
        for i in range(n // 2):
            x, y = nums[i], nums[n - 1 - i]
            # 需要操作 1 次能达到的区间
            low_1 = 1 + x if x < y else 1 + y
            high_1 = limit + x if x > y else limit + y
            if low_1 <= high_1:
                diff[low_1] += 1
                diff[high_1 + 1] -= 1
            diff[x + y] -= 1  # 刨除操作 0 次的点
            diff[x + y + 1] += 1

            # 需要操作 2 次能达到的区间
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
        url: https://leetcode.com/problems/find-kth-largest-xor-coordinate-value/
        tag: matrix_prefix_xor_sum
        """

        # matrix_prefix_xor_sum
        m, n = len(matrix), len(matrix[0])
        # 原地异或运算
        for i in range(1, m):
            matrix[i][0] = matrix[i][0] ^ matrix[i - 1][0]
        for j in range(1, n):
            matrix[0][j] = matrix[0][j] ^ matrix[0][j - 1]
        for i in range(1, m):
            for j in range(1, n):
                matrix[i][j] = matrix[i - 1][j - 1] ^ matrix[i -
                                                             1][j] ^ matrix[i][j - 1] ^ matrix[i][j]

        # sorting后返回结果
        lst = []
        for i in range(m):
            lst.extend(matrix[i])
        lst.sort()
        return lst[-k]

    @staticmethod
    def lc_2132(grid: List[List[int]], h: int, w: int) -> bool:
        """
        url: https://leetcode.com/problems/stamping-the-grid/
        tag: prefix_sum|brute_force|diff_matrix|implemention
        """
        # 用prefix_sumbrute_force可行的邮票左上端点，然后查看空白格点左上方是否有可行的邮票点，也可以的diff_matrix|滚动implemention覆盖解决
        m, n = len(grid), len(grid[0])

        # brute_force可以贴邮票的左上角位置
        pre = PreFixSumMatrix(grid)
        dp = [[0] * n for _ in range(m)]
        for i in range(m - h + 1):
            for j in range(n - w + 1):
                cur = pre.query(i, j, i + h - 1, j + w - 1)
                if cur == 0:
                    dp[i][j] = 1
        # 位置的prefix_sum
        pre = PreFixSumMatrix(dp)
        for i in range(m):
            for j in range(n):
                if not grid[i][j]:
                    # 检查是否有邮票覆盖
                    x = i - h + 1 if i - h + 1 > 0 else 0
                    y = j - w + 1 if j - w + 1 > 0 else 0
                    cur = pre.query(x, y, i, j)
                    if not cur:
                        return False
        return True

    @staticmethod
    def ac_3993(ac=FastIO()):
        # 后缀和data_rangebrain_teaser|
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        low = min(nums)
        high = max(nums)
        if low == high:
            ac.st(0)
            return
        # 按照data_rangecounter
        cnt = [0] * (high - low + 1)
        for num in nums:
            cnt[num - low] += 1
        ans = post_cnt = post_sum = 0
        for i in range(high - low, 0, -1):
            post_cnt += cnt[i]  # counter
            post_sum += post_cnt  # 变为i-1需要的代价
            # 假如往下变为 i-1的代价不可行，则先变为 i
            if post_sum > k:
                ans += 1
                post_sum = post_cnt
        # 此时还需要变为0
        ans += post_sum > 0
        ac.st(ans)
        return

    @staticmethod
    def lc_837(n: int, k: int, max_pts: int) -> float:
        """
        url: https://leetcode.com/problems/new-21-game/description/
        tag: diff_array|implemention|probability
        """
        # diff_array|implemention概率
        s = k + max_pts
        dp = [0] * (s + 1)
        dp[0] = 1
        diff = [0] * (s + 1)
        for i in range(s + 1):
            diff[i] += diff[i - 1]
            dp[i] += diff[i]
            if i < k:
                diff[i + 1] += dp[i] / max_pts
                diff[i + max_pts + 1] -= dp[i] / max_pts
        return sum(dp[k:n + 1])