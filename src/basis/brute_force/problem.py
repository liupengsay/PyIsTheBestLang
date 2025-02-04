"""
Algorithm：brute_force|matrix_rotate|matrix_spiral|contribution_method
Description：brute force according to the data range

====================================LeetCode====================================
670（https://leetcode.cn/problems/maximum-swap/）greedy|brute_force
395（https://leetcode.cn/problems/longest-substring-with-at-least-k-repeating-characters/）brute_force|divide_and_conquer
1330（https://leetcode.cn/problems/reverse-subarray-to-maximize-array-value/）brute_force
2488（https://leetcode.cn/problems/count-subarrays-with-median-k/）median|brute_force|pre_consequence|post_consequence
2484（https://leetcode.cn/problems/count-palindromic-subsequences/）prefix_suffix|hash|brute_force|palindrome_substring
2322（https://leetcode.cn/problems/minimum-score-after-removals-on-a-tree/）brute_force|tree_dp|union_find|xor_min
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
1215（https://leetcode.cn/problems/stepping-numbers/）data_range|back_trace|brute_force
2245（https://leetcode.cn/problems/maximum-trailing-zeros-in-a-cornered-path/）prefix_sum|brute_force
1878（https://leetcode.cn/problems/get-biggest-three-rhombus-sums-in-a-grid/）prefix_sum|brute_force
2018（https://leetcode.cn/problems/check-if-word-can-be-placed-in-crossword/description/）brute_force
2591（https://leetcode.cn/problems/distribute-money-to-maximum-children/）brute_force
910（https://leetcode.cn/problems/smallest-range-ii/description/）brute_force|data_range
1131（https://leetcode.cn/problems/maximum-of-absolute-value-expression/description/）manhattan_distance|brute_force
1761（https://leetcode.cn/problems/minimum-degree-of-a-connected-trio-in-a-graph/）directed_graph|undirected_graph|brute_force
1178（https://leetcode.cn/problems/number-of-valid-words-for-each-puzzle/）hash|counter|brute_force|bit_operation
1638（https://leetcode.cn/problems/count-substrings-that-differ-by-one-character/description/）brute_force|dp|brute_force
2212（https://leetcode.cn/problems/maximum-points-in-an-archery-competition/）bit_operation|brute_force|back_trace
2749（https://leetcode.cn/problems/minimum-operations-to-make-the-integer-zero/）brute_force|bit_operation
2094（https://leetcode.cn/problems/finding-3-digit-even-numbers/description/）brain_teaser|brute_force|data_range
842（https://leetcode.cn/problems/split-array-into-fibonacci-sequence/description/）brain_teaser|brute_force|back_trace
2122（https://leetcode.cn/problems/recover-the-original-array/）brute_force|hash|implemention
1782（https://leetcode.cn/problems/count-pairs-of-nodes/description/）brute_force
3102（https://leetcode.cn/problems/minimize-manhattan-distances/）manhattan_distance|brain_teaser|implemention|prefix_suffix|classical
3272（https://leetcode.cn/problems/find-the-count-of-good-integers/）brute_force|palindrome_num
3405（https://leetcode.cn/problems/count-the-number-of-arrays-with-k-matching-adjacent-elements/）counter|brute_force|hash|preprocess
3104（https://leetcode.cn/problems/find-longest-self-contained-substring/）brute_force|prefix_sum
2910（https://leetcode.cn/problems/minimum-number-of-groups-to-create-a-valid-assignment/）brute_force|greedy

=====================================LuoGu======================================
P1548（https://www.luogu.com.cn/problem/P1548）brute_force
P1632（https://www.luogu.com.cn/problem/P1632）brute_force
P2128（https://www.luogu.com.cn/problem/P2128）brute_force
P2191（https://www.luogu.com.cn/problem/P2191）reverse_thinking|matrix_rotate
P2699（https://www.luogu.com.cn/problem/P2699）classification_discussion|brute_force|implemention
P1371（https://www.luogu.com.cn/problem/P1371）prefix_suffix|brute_force|counter
P1369（https://www.luogu.com.cn/problem/P1369）matrix_dp|greedy|brute_force
P1158（https://www.luogu.com.cn/problem/P1158）sort|brute_force|suffix_maximum
P8928（https://www.luogu.com.cn/problem/P8928）brute_force|counter
P8892（https://www.luogu.com.cn/problem/P8892）brute_force
P8799（https://www.luogu.com.cn/problem/P8799）brute_force
P3142（https://www.luogu.com.cn/problem/P3142）brute_force|sorted_list
P3143（https://www.luogu.com.cn/problem/P3143）brute_force|prefix_suffix
P3670（https://www.luogu.com.cn/problem/P3670）hash|brute_force|counter
P3799（https://www.luogu.com.cn/problem/P3799）brute_force|regular_triangle
P3910（https://www.luogu.com.cn/problem/P3910）factorization|brute_force
P4086（https://www.luogu.com.cn/problem/P4086）suffix|reverse_order|brute_force
P4596（https://www.luogu.com.cn/problem/P4596）brute_force
P4759（https://www.luogu.com.cn/problem/P4759）factorization|brute_force
P6267（https://www.luogu.com.cn/problem/P6267）factorization|brute_force
P5077（https://www.luogu.com.cn/problem/P5077）factorization|brute_force
P4960（https://www.luogu.com.cn/problem/P4960）implemention|brute_force
P4994（https://www.luogu.com.cn/problem/P4994）implemention|pi(n)<=6n
P5190（https://www.luogu.com.cn/problem/P5190）counter|prefix_sum|harmonic_progression|O(nlogn)
P5614（https://www.luogu.com.cn/problem/P5614）brute_force
P6014（https://www.luogu.com.cn/problem/P6014）hash|mod|counter
P6067（https://www.luogu.com.cn/problem/P6067）sort|prefix_suffix|brute_force
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
P7286（https://www.luogu.com.cn/problem/P7286）sort|brute_force|counter
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
P8872（https://www.luogu.com.cn/problem/P8872）sort|prefix_suffix|brute_force
P1989（https://www.luogu.com.cn/problem/P1989）ternary_circle|dag|build_graph|counter|brute_force|classical
P8599（https://www.luogu.com.cn/problem/P8599）brute_force

===================================CodeForces===================================
1426F（https://codeforces.com/problemset/problem/1426/F）classification_discussion|brute_force|counter|fast_power
1400D（https://codeforces.com/problemset/problem/1400/D）brute_force|binary_search
1793D（https://codeforces.com/contest/1793/problem/D）brute_force|counter
584D（https://codeforces.com/problemset/problem/584/D）brute_force|prime|decompose_into_sum_of_prime_at_most_3
1311D（https://codeforces.com/problemset/problem/1311/D）greedy|brute_force
1181C（https://codeforces.com/problemset/problem/1181/C）column_wised|brute_force
484B（https://codeforces.com/problemset/problem/484/B）sort|brute_force|binary_search
382C（https://codeforces.com/problemset/problem/382/C）classification_discussion
988E（https://codeforces.com/contest/988/problem/E）brain_teaser|greedy|brute_force
1661B（https://codeforces.com/contest/1661/problem/B）brute_force
1692F（https://codeforces.com/contest/1692/problem/F）brute_force
978D（https://codeforces.com/contest/978/problem/D）brute_force
1029D（https://codeforces.com/contest/1029/problem/D）brute_force|hash|math
1029C（https://codeforces.com/contest/1029/problem/C）classical|prefix_suffix|brute_force
1077E（https://codeforces.com/contest/1077/problem/E）brute_force|data_range
1203D2（https://codeforces.com/contest/1203/problem/D2）brute_force|data_range|prefix_suffix
1216E2（https://codeforces.com/contest/1216/problem/E2）brute_force|data_range|classical
1328F（https://codeforces.com/contest/1328/problem/F）brute_force|prefix_sum|binary_search
1426F（https://codeforces.com/contest/1426/problem/F）brute_force|implemention
1618F（https://codeforces.com/contest/1618/problem/F）brute_force|bfs|data_range
1744F（https://codeforces.com/contest/1744/problem/F）brute_force
1729F（https://codeforces.com/contest/1729/problem/F）brute_force
1846E2（https://codeforces.com/contest/1846/problem/E2）brute_force|binary_search
1926F（https://codeforces.com/contest/1926/problem/F）brute_force|greedy|classical
1689D（https://codeforces.com/contest/1689/problem/D）manhattan_distance|prefix_suffix|classical
1992E（https://codeforces.com/contest/1992/problem/E）brute_force|observation|data_range
1648B（https://codeforces.com/problemset/problem/1648/B）observation|brute_force|data_range|euler_series
466B（https://codeforces.com/problemset/problem/466/B）observation|brute_force
1804C（https://codeforces.com/problemset/problem/1804/C）brute_force|math

====================================AtCoder=====================================
ARC060B（https://atcoder.jp/contests/abc044/tasks/arc060_b）base|classification_discussion|brute_force|factorization
ARC069B（https://atcoder.jp/contests/abc055/tasks/arc069_b）brain_teaser|brute_force
ARC072A（https://atcoder.jp/contests/abc059/tasks/arc072_a）brute_force|prefix_sum|greedy
ARC074A（https://atcoder.jp/contests/abc062/tasks/arc074_a）brute_force
ARC083A（https://atcoder.jp/contests/abc074/tasks/arc083_a）brute_force|math
ARC091B（https://atcoder.jp/contests/abc090/tasks/arc091_b）brute_force
ABC085D（https://atcoder.jp/contests/abc085/tasks/abc085_d）brute_force
ABC338C（https://atcoder.jp/contests/abc338/tasks/abc338_c）brute_force|data_range
ABC334C（https://atcoder.jp/contests/abc334/tasks/abc334_c）brute_force|prefix_suffix
ABC330D（https://atcoder.jp/contests/abc330/tasks/abc330_d）brute_force|prefix_sum
ABC330C（https://atcoder.jp/contests/abc330/tasks/abc330_c）brute_force|prefix_sum
ABC324D（https://atcoder.jp/contests/abc324/tasks/abc324_d）brute_force
ABC323F（https://atcoder.jp/contests/abc323/tasks/abc323_f）brute_force
ABC320C（https://atcoder.jp/contests/abc320/tasks/abc320_c）brute_force
ABC319C（https://atcoder.jp/contests/abc319/tasks/abc319_c）brute_force
ABC310D（https://atcoder.jp/contests/abc310/tasks/abc310_d）brute_force|implemention|state_compression
ABC307C（https://atcoder.jp/contests/abc307/tasks/abc307_c）brute_force|implemention
ABC345D（https://atcoder.jp/contests/abc345/tasks/abc345_d）brute_force|implemention
ABC345C（https://atcoder.jp/contests/abc345/tasks/abc345_c）contribution_method
ABC302G（https://atcoder.jp/contests/abc302/tasks/abc302_g）brute_force
ABC300F（https://atcoder.jp/contests/abc300/tasks/abc300_f）brute_force|circular_array|prefix_sum|greedy
ABC296D（https://atcoder.jp/contests/abc296/tasks/abc296_d）brute_force|math
ABC346D（https://atcoder.jp/contests/abc346/tasks/abc346_d）brute_force|prefix_suffix
ABC290E（https://atcoder.jp/contests/abc290/tasks/abc290_e）brute_force|contribution_method
ABC178E（https://atcoder.jp/contests/abc178/tasks/abc178_e）manhattan_distance|prefix_suffix|classical
ABC260F（https://atcoder.jp/contests/abc260/tasks/abc260_f）brute_force|data_range|brain_teaser|classical
ABC255E（https://atcoder.jp/contests/abc255/tasks/abc255_e）brute_force|hash|counter|contribution_method|data_range
ABC256C（https://atcoder.jp/contests/abc256/tasks/abc256_c）brute_force
ABC252C（https://atcoder.jp/contests/abc252/tasks/abc252_c）brute_force
ABC251C（https://atcoder.jp/contests/abc251/tasks/abc251_c）brute_force
ABC249D（https://atcoder.jp/contests/abc249/tasks/abc249_d）euler_series|nlogn|brute_force|contribution_method|classical
ABC246F（https://atcoder.jp/contests/abc246/tasks/abc246_f）brute_force|inclusion_exclusion|math|counter|bit_operation
ABC247F（https://atcoder.jp/contests/abc247/tasks/abc247_f）brute_force|guess_table|union_find|dp|brain_teaser
ABC247E（https://atcoder.jp/contests/abc247/tasks/abc247_e）inclusion_exclusion|two_pointers|counter|brain_teaser|classical
ABC238C（https://atcoder.jp/contests/abc238/tasks/abc238_c）digit_num|counter|brute_force|classical
ABC234E（https://atcoder.jp/contests/abc234/tasks/abc234_e）brute_force|data_range|classical
ABC232C（https://atcoder.jp/contests/abc232/tasks/abc232_c）brute_force|implemention
ABC230C（https://atcoder.jp/contests/abc230/tasks/abc230_c）brute_force|implemention
ABC223E（https://atcoder.jp/contests/abc223/tasks/abc223_e）brute_force|implemention
ABC353D（https://atcoder.jp/contests/abc353/tasks/abc353_d）contribution_method|implemention
ABC220E（https://atcoder.jp/contests/abc220/tasks/abc220_e）contribution_method|classical|brute_force|binary_tree
ABC219E（https://atcoder.jp/contests/abc219/tasks/abc219_e）brute_force|union_find|brain_teaser
ABC366E（https://atcoder.jp/contests/abc366/tasks/abc366_e）brute_force|prefix_suffix
ABC198D（https://atcoder.jp/contests/abc198/tasks/abc198_d）brute_force|permutation
ABC184C（https://atcoder.jp/contests/abc184/tasks/abc184_c）brute_force|brain_teaser|observation
ABC181E（https://atcoder.jp/contests/abc181/tasks/abc181_e）brute_force|binary_search|greedy|prefix_suffix
ABC180D（https://atcoder.jp/contests/abc180/tasks/abc180_d）brute_force
ABC176E（https://atcoder.jp/contests/abc176/tasks/abc176_e）brute_force|observation|greedy
ABC379E（https://atcoder.jp/contests/abc379/tasks/abc379_e）brute_force|contribution_method
ABC173F（https://atcoder.jp/contests/abc173/tasks/abc173_f）union_find|contribution_method|brute_force
ABC165C（https://atcoder.jp/contests/abc165/tasks/abc165_c）brute_force|dfs

===================================CodeForces===================================
1971F（https://codeforces.com/contest/1971/problem/F）brute_force|high_precision
1422C（https://codeforces.com/problemset/problem/1422/C）contribution_method|brute_force
451D（https://codeforces.com/problemset/problem/451/D）contribution_method|brute_force
1995B2（https://codeforces.com/problemset/problem/1995/B2）brute_force
1379B（https://codeforces.com/problemset/problem/1379/B）brute_force

=====================================AcWing=====================================
97（https://www.acwing.com/problem/content/description/97/）brute_force

=====================================CodeChef=====================================
1（https://www.codechef.com/problems/GRAPHCOST）prefix_suffix|brute_force|brain_teaser

"""
import bisect
import math
from collections import defaultdict, deque, Counter
from functools import reduce, lru_cache
from itertools import combinations, permutations
from operator import mul, or_, and_
from typing import List

from src.graph.union_find.template import UnionFind
from src.util.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1311d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1311/D
        tag: greedy|brute_force|specific_plan|data_range|classical
        """
        for _ in range(ac.read_int()):
            a, b, c = ac.read_list_ints()
            ans = math.inf
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
        tag: brute_force|prime|decompose_into_sum_of_prime_at_most_3|classical
        """

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

        n = ac.read_int()
        assert 3 <= n < 10 ** 9

        if is_prime4(n):
            ac.st(1)
            ac.st(n)
            return

        # there is a fact that the distance between adjacent prime numbers is not big
        # for n=10**9 the maximal distance is 282
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

        def check():  # greedy
            lst = list(str(num))
            n = len(lst)
            post = list(range(n))
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
        tag: sort|brute_force|binary_search|classical|maximum_mod_pair|euler_series|O(nlogn)
        """
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
                    ans = max(ans, dp[min(a, ceil)] % num)
        ac.st(ans)
        return

    @staticmethod
    def cf_382c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/382/C
        tag: classification_discussion|brute_force
        """
        n = ac.read_int()
        nums = sorted(ac.read_list_ints())

        if n == 1:
            ac.st(-1)
            return

        diff = [nums[i] - nums[i - 1] for i in range(1, n)]
        high = max(diff)
        low = min(diff)
        cnt = len(set(diff))

        if cnt >= 3:
            ac.st(0)
            return
        elif cnt == 2:
            if high != 2 * low or diff.count(high) != 1:
                ac.st(0)
                return

            for i in range(1, n):
                if nums[i] - nums[i - 1] == high:
                    ac.st(1)
                    ac.st(nums[i - 1] + low)
                    return
        else:
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
    def arc_060b(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc044/tasks/arc060_b
        tag: n_base|classification_discussion|brute_force|factorization|math

        """

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
            ans = math.inf
            for x in range(1, n - s + 1):
                if x * x > n - s:
                    break
                if (n - s) % x == 0:
                    # brute_force (n-s) % (b-1) == 0
                    y = (n - s) // x
                    b = x + 1
                    if check():
                        ans = b if ans > b else ans
                    b = y + 1
                    if check():
                        ans = b if ans > b else ans
            ac.st(-1 if ans == math.inf else ans)
        return

    @staticmethod
    def abc_072a(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc059/tasks/arc072_a
        tag: brute_force|prefix_sum|greedy
        """
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
        ac.st(min(ans1, ans2))
        return

    @staticmethod
    def abc_074a(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc062/tasks/arc074_a
        tag: brute_force
        """
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

        ans = math.inf
        check1()
        check2()
        m, n = n, m
        check1()
        check2()
        ac.st(ans)
        return

    @staticmethod
    def abc_083a(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc074/tasks/arc083_a
        tag: brute_force|math
        """
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
    def ac_97(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/97/
        tag: brute_force
        """
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
        n, k = ac.read_list_ints()
        nums = ac.read_list_str()

        ans = 0
        for item in combinations(list(range(1, n)), k):
            cur = nums[:]
            for i in item:
                cur[i] = "*" + cur[i]
            res = [int(w) for w in ("".join(cur)).split("*")]
            cur = reduce(mul, res)
            ans = max(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1311(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1311
        tag: brute_force|counter
        """
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
        n, m = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(m)]

        cnt = [0] * (n + 1)
        for num in nums:
            cnt[num] += 1

        aa = [0] * (n + 1)
        bb = [0] * (n + 1)
        cc = [0] * (n + 1)
        dd = [0] * (n + 1)

        # brute_force b-a=x
        for x in range(1, n // 9 + 1):
            if 1 + 9 * x + 1 > n:
                break

            # ab counter
            pre_ab = [0] * (n + 1)
            for b in range(2 * x + 1, n + 1):
                pre_ab[b] = pre_ab[b - 1]
                pre_ab[b] += cnt[b] * cnt[b - 2 * x]

            # cd
            for c in range(n - x, -1, -1):
                if c - 6 * x - 1 >= 1:
                    cc[c] += pre_ab[c - 6 * x - 1] * cnt[c + x]
                    dd[c + x] += pre_ab[c - 6 * x - 1] * cnt[c]
                else:
                    break

            # cd
            post_cd = [0] * (n + 2)
            for c in range(n - x, -1, -1):
                post_cd[c] = post_cd[c + 1]
                post_cd[c] += cnt[c] * cnt[c + x]

            # ab counter
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
                ans = min(ans, n - (j - i))
        ac.st(ans)
        return

    @staticmethod
    def lg_p2994(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2994
        tag: brute_force|reverse_thinking
        """

        def dis():
            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        n, m = ac.read_list_ints()
        cow = [ac.read_list_ints() for _ in range(n)]
        pos = [ac.read_list_ints() for _ in range(n)]
        visit = [0] * n
        for j in range(m):
            ceil = math.inf
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
            if ceil < math.inf:
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
        n, m, r = ac.read_list_ints()
        cow = [ac.read_int() for _ in range(n)]
        cow.sort()
        nums1 = [ac.read_list_ints()[::-1] for _ in range(m)]
        nums1.sort(key=lambda it: -it[0])
        nums2 = [ac.read_int() for _ in range(r)]
        nums2.sort(reverse=True)

        ind = 0
        post = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            cur = 0
            while ind < m and cow[i]:
                if nums1[ind][1] == 0:
                    ind += 1
                    continue
                x = min(nums1[ind][1], cow[i])
                cow[i] -= x
                nums1[ind][1] -= x
                cur += nums1[ind][0] * x
            post[i] = post[i + 1] + cur

        ans = post[0]
        pre = 0
        for i in range(min(r, n)):
            pre += nums2[i]
            ans = max(ans, pre + post[i + 1])
        ac.st(ans)
        return

    @staticmethod
    def lg_p6149(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6149
        tag: brute_force|triangle|prefix_sum|binary_search
        """
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
        n, a, b, c, d = ac.read_list_ints()
        if b * c > a * d:
            a, b, c, d = c, d, a, b

        ans = math.inf
        for x in range(10 ** 5 + 1):
            cur = x * d + b * max(math.ceil((n - x * c) / a), 0)
            ans = min(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p8270(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8270
        tag: brain_teaser|brute_force
        """
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
        tag: string|brute_force|permutation_circle|counter|classical|brain_teaser
        """
        s = ac.read_str()
        n = len(s)
        dct = dict()
        dct["B"] = s.count("B")
        dct["A"] = s.count("A")
        dct["T"] = s.count("T")
        ans = math.inf
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
                        x = min(cnt[w + p], cnt[p + w])
                        cur += x
                        cnt[w + p] -= x
                        cnt[p + w] -= x
            rest = sum(cnt.values())
            cur += rest * 2 // 3
            ans = min(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p9076(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P9076
        tag: factorization|brute_force|classical|square_complexity|sqrt_n
        """
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
                        if x * x > bc:
                            break
                        if bc % x == 0:
                            y = bc // x
                            if x == y:
                                if x >= 3:
                                    ans += 1
                                continue
                            if y >= 3:
                                ans += 1
                            if x >= 3:
                                ans += 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p9008(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P9008
        tag: inclusion_exclusion|brute_force|counter|classical
        """
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
        tag: brute_force|preprocess
        """
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
        tag: data_range|brute_force|prefix_sum|counter|inclusion_exclusion
        """
        n = ac.read_int()
        mod = 998244353
        nums = [ac.read_list_ints() for _ in range(n)]
        ceil = max(q for _, q in nums)
        low = min(p for p, _ in nums)
        ans = pre = 0
        for s in range(low, ceil + 1):
            cnt = 1
            flag = 0
            for p, q in nums:
                if p <= s <= q:
                    cnt *= (s - p + 1)
                    flag = 1
                    cnt %= mod
                elif s > q:
                    cnt *= (q - p + 1)
                    cnt %= mod
                elif s < p:
                    cnt = 0
                    break
            if flag:
                ans = (ans + (cnt - pre) * s) % mod
                pre = cnt
        ac.st(ans)
        return

    @staticmethod
    def lg_p8872(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8872
        tag: sort|prefix_suffix|brute_force|reverse_order
        """
        n, m = ac.read_list_ints()
        nums = sorted(ac.read_list_ints())
        ans = math.inf
        for i in range(n):
            if i > m:
                break
            right = (m - i) // 2
            if right >= n - i - 1:
                ac.st(0)
                return
            cur = nums[-right - 1] - nums[i]
            ans = min(ans, cur)

        for i in range(n - 1, -1, -1):
            if n - i - 1 > m:
                break
            left = (m - n + i + 1) // 2
            if left >= i:
                ac.st(0)
                return
            cur = nums[i] - nums[left]
            ans = min(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lc_2018(board: List[List[str]], word: str) -> bool:
        """
        url: https://leetcode.cn/problems/check-if-word-can-be-placed-in-crossword/description/
        tag: brute_force
        """
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

        a = b = 0
        for num in even:
            if even[num] >= a:
                a, b = even[num], a
            elif even[num] >= b:
                b = even[num]

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
        tag: hash|counter|brute_force|bit_operation|subset_enumeration|classical
        """
        n = 7
        dct = defaultdict(int)
        for word in words:
            cur = set(word)
            lst = [ord(w) - ord("a") for w in cur]
            state = reduce(or_, [1 << x for x in lst])
            if len(cur) <= n:
                dct[state] += 1

        ans = []
        for word in puzzles:
            lst = [ord(w) - ord("a") for w in word]
            start = 1 << lst[0]
            cur = 0
            mask = reduce(or_, [1 << x for x in lst])
            sub = mask
            while sub:  # classical
                if sub & start:
                    cur += dct[sub]
                sub = (sub - 1) & mask
            ans.append(cur)
        return ans

    @staticmethod
    def lc_1215(low: int, high: int) -> List[int]:
        """
        url: https://leetcode.cn/problems/stepping-numbers/
        tag: data_range|brute_force|preprocess
        """
        res = []
        pre = list(range(10))
        res.extend(pre)
        ceil = 2 * 10 ** 9
        for _ in range(10):
            cur = []
            for num in pre:
                if str(num)[0] == "0":
                    continue
                d = int(str(num)[-1])
                for w in [d - 1, d + 1]:
                    if 0 <= w <= 9 and num * 10 + w <= ceil:
                        cur.append(num * 10 + w)
            pre = cur[:]
            res.extend(pre)

        i, j = bisect.bisect_left(res, low), bisect.bisect_right(res, high)
        return res[i:j]

    @staticmethod
    def lc_1131(arr1: List[int], arr2: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-of-absolute-value-expression/description/
        tag: manhattan_distance|brute_force|classical
        """
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
        tag: brute_force|dp|brute_force|classical|brain_teaser
        """
        m = len(s)
        n = len(t)
        cnt = [[0] * (n + 1) for _ in range(m + 1)]
        same = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if s[i] == t[j]:
                    same[i + 1][j + 1] = same[i][j] + 1
                    cnt[i + 1][j + 1] = cnt[i][j]
                else:
                    same[i + 1][j + 1] = 0
                    cnt[i + 1][j + 1] = same[i][j] + 1
        return sum(sum(d) for d in cnt)

    @staticmethod
    def lc_1761(n: int, edges: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-degree-of-a-connected-trio-in-a-graph/
        tag: directed_graph|undirected_graph|brute_force
        """
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
        ans = math.inf
        for i in range(n):
            for j in directed[i]:
                for k in directed[j]:
                    if k in dct[i]:
                        x = degree[i] + degree[j] + degree[k] - 6
                        if x < ans:
                            ans = x
        return ans if ans < math.inf else -1

    @staticmethod
    def lc_1878(grid: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/get-biggest-three-rhombus-sums-in-a-grid/
        tag: prefix_sum|brute_force
        """
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
        tag: bit_operation|brute_force|back_trace
        """
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

    @staticmethod
    def abc_296d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc296/tasks/abc296_d
        tag: brute_force|math
        """
        n, m = ac.read_list_ints()
        x = m
        while x <= n * n:
            for y in range(x // n, int(x ** 0.5) + 1):
                if 1 <= y <= n and 1 <= x // y <= n and x % y == 0:
                    ac.st(x)
                    return
            x += 1
        ac.st(-1)
        return

    @staticmethod
    def abc_178e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc178/tasks/abc178_e
        tag: manhattan_distance|prefix_suffix|classical
        """

        nums = [ac.read_list_ints() for _ in range(ac.read_int())]
        nums.sort()

        def check():
            n = len(nums)
            pos = [-math.inf, -1]
            neg = [-math.inf, -1]
            res = [-math.inf, (-1, -1)]
            for i in range(n):
                x, y = nums[i]
                if pos[0] - y + x > res[0]:
                    res = [pos[0] - y + x, (pos[1], i)]
                if neg[0] + y + x > res[0]:
                    res = [neg[0] + y + x, (neg[1], i)]
                if y - x > pos[0]:
                    pos = [y - x, i]
                if -y - x > neg[0]:
                    neg = [-y - x, i]
            return res

        ans = check()[0]
        ac.st(ans)
        return

    @staticmethod
    def lc_3102_1(points: List[List[int]]) -> int:

        """
        url: https://leetcode.cn/problems/minimize-manhattan-distances/
        tag: manhattan_distance|brain_teaser|implemention|prefix_suffix|classical
        """

        points.sort()

        def check():
            n = len(nums)
            pos = [-math.inf, -1]
            neg = [-math.inf, -1]
            res = [-math.inf, (-1, -1)]
            for i in range(n):
                x, y = nums[i]
                if pos[0] - y + x > res[0]:
                    res = [pos[0] - y + x, (pos[1], i)]
                if neg[0] + y + x > res[0]:
                    res = [neg[0] + y + x, (neg[1], i)]
                if y - x > pos[0]:
                    pos = [y - x, i]
                if -y - x > neg[0]:
                    neg = [-y - x, i]
            return res

        ans = math.inf
        nums = points[:]
        m = len(points)
        lst = check()[1][:]
        for xx in lst:
            nums = [points[i] for i in range(m) if i != xx]
            ans = min(ans, check()[0])
        return ans

    @staticmethod
    def lc_3102_2(points: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimize-manhattan-distances/
        tag: manhattan_distance|brain_teaser|implemention|prefix_suffix|classical
        """
        ceil_x1, ceil_x2 = -math.inf, -math.inf
        floor_x1, floor_x2 = math.inf, math.inf
        ceil_ix, floor_ix = -1, -1

        ceil_y1, ceil_y2 = -math.inf, -math.inf
        floor_y1, floor_y2 = math.inf, math.inf
        ceil_iy, floor_iy = -1, -1
        for i, (x, y) in enumerate(points):
            x, y = x + y, x - y
            if x >= ceil_x1:
                ceil_x1, ceil_x2 = x, ceil_x1
                ceil_ix = i
            elif x > ceil_x2:
                ceil_x2 = x

            if x <= floor_x2:
                floor_x1, floor_x2 = floor_x2, x
                floor_ix = i
            elif x < floor_x1:
                floor_x1 = x

            if y >= ceil_y1:
                ceil_y1, ceil_y2 = y, ceil_y1
                ceil_iy = i
            elif y > ceil_y2:
                ceil_y2 = y

            if y <= floor_y2:
                floor_y1, floor_y2 = floor_y2, y
                floor_iy = i
            elif y < floor_y1:
                floor_y1 = y

        ans = 0
        for i in ceil_ix, floor_ix, ceil_iy, floor_iy:
            dx = max(ceil_x1 if i != ceil_ix else ceil_x2) - min(floor_x2 if i != floor_ix else floor_x1)
            dy = max(ceil_y1 if i != ceil_iy else ceil_y2) - min(floor_y2 if i != floor_iy else floor_y1)
            ans = min(ans, max(dx, dy))
        return ans

    @staticmethod
    def abc_260f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc260/tasks/abc260_f
        tag: brute_force|data_range|brain_teaser|classical
        """
        s, t, m = ac.read_list_ints()
        edge = [[] for _ in range(s)]
        pre = [-1] * t * t
        for _ in range(m):
            u, v = ac.read_list_ints_minus_one()
            v -= s
            for i in edge[u]:
                if pre[i * t + v] != -1:
                    ans = [pre[i * t + v] + 1, i + s + 1, v + s + 1, u + 1]
                    ac.lst(ans)
                    return
                pre[i * t + v] = pre[v * t + i] = u
            edge[u].append(v)
        ac.st(-1)
        return

    @staticmethod
    def lg_p1989(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1989
        tag: ternary_circle|dag|build_graph|counter|brute_force|classical
        """
        n, m = ac.read_list_ints()  # TLE
        edge = [set() for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            edge[i].add(j)
            edge[j].add(i)
        degree = [len(e) for e in edge]
        dct = [[] for _ in range(n)]
        for i in range(n):
            for j in edge[i]:
                if (degree[i] < degree[j]) or (degree[i] == degree[j] and i < j):
                    dct[i].append(j)
        ans = 0
        for i in range(n):
            for j in dct[i]:
                ans += sum(k in edge[i] for k in dct[j])
        ac.st(ans)
        return

    @staticmethod
    def abc_249d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc249/tasks/abc249_d
        tag: euler_series|nlogn|brute_force|contribution_method|classical
        """
        ac.read_int()
        nums = ac.read_list_ints()
        n = 2 * 10 ** 5
        cnt = [0] * (n + 1)
        for num in nums:
            cnt[num] += 1
        ans = cnt[1] ** 3
        for i in range(2, n + 1):
            ans += 2 * cnt[1] * cnt[i] * cnt[i]

        for i in range(2, n + 1):
            for j in range(i, n + 1):
                if i * j > n:
                    break
                if i == j:
                    ans += cnt[i] * cnt[j] * cnt[i * j]
                else:
                    ans += cnt[i] * cnt[j] * cnt[i * j] * 2
        ac.st(ans)
        return

    @staticmethod
    def abc_246f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc246/tasks/abc246_f
        tag: brute_force|inclusion_exclusion|math|counter|bit_operation
        """
        n, ll = ac.read_list_ints()
        lst = [sum(1 << (ord(w) - ord("a")) for w in ac.read_str()) for _ in range(n)]
        mod = 998244353
        p = [pow(x, ll, mod) for x in range(27)]
        res = 0
        for x in range(1, 1 << n):
            cur = [lst[j] for j in range(n) if x & (1 << j)]
            val = reduce(and_, cur)
            res += (-1) ** (len(cur) - 1) * p[val.bit_count()]
            res %= mod
        ac.st(res)
        return

    @staticmethod
    def main(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc247/tasks/abc247_f
        tag: brute_force|guess_table|union_find|dp|brain_teaser
        """
        mod = 998244353
        n = ac.read_int()
        dp = [0] * (2 * 10 ** 5 + 1)
        dp[2] = 3
        dp[3] = 4
        for x in range(4, 2 * 10 ** 5 + 1):
            dp[x] = (dp[x - 1] + dp[x - 2]) % mod
        p = ac.read_list_ints_minus_one()
        q = ac.read_list_ints_minus_one()
        uf = UnionFind(n)
        for i in range(n):
            uf.union(p[i], q[i])
        ans = 1
        group = uf.get_root_size()
        for g in group.values():
            if g != 1:
                ans *= dp[g]
                ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def abc_247e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc247/tasks/abc247_e
        tag: inclusion_exclusion|two_pointers|counter|brain_teaser|classical
        """
        n, x, y = ac.read_list_ints()
        nums = ac.read_list_ints()
        pre_x = pre_y = -1
        last = ans = 0
        for i, num in enumerate(nums):
            if num < y or num > x:
                last = i + 1
                pre_x = pre_y = -1
            if num == x:
                pre_x = i
            if num == y:
                pre_y = i
            if pre_x != -1 and pre_y != -1:
                ans += min(pre_x, pre_y) - last + 1
        ac.st(ans)
        return

    @staticmethod
    def abc_238c(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc238/tasks/abc238_c
        tag: digit_num|counter|brute_force|classical
        """
        n = ac.read_int()
        m = len(str(n))
        mod = 998244353
        ans = 0
        for x in range(1, m + 1):
            cnt = min(10 ** x, n) - 10 ** (x - 1)
            if n < 10 ** x:
                cnt += 1
            ans += (1 + cnt) * cnt // 2
            ans %= mod
        ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def abc_220e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc220/tasks/abc220_e
        tag: contribution_method|classical|brute_force|binary_tree
        """
        mod = 998244353
        n, d = ac.read_list_ints()
        p = [1] * (n + 1)
        for x in range(1, n + 1):
            p[x] = (p[x - 1] * 2) % mod

        ans = 0
        for ll in range(d + 1):
            rr = d - ll
            ss = max(ll, rr)
            if ss <= n - 1:
                root = p[n - ss] - 1
                left = p[max(0, ll - 1)]
                right = p[max(0, rr - 1)]
                ans += root * left * right * 2
                ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def abc_219e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc219/tasks/abc219_e
        tag: brute_force|union_find|brain_teaser
        """
        grid = [ac.read_list_ints() for _ in range(4)]
        ans = 0
        nodes = []
        for i in range(4):
            for j in range(4):
                if grid[i][j]:
                    nodes.append(i * 4 + j)

        m = 16
        target = sum(1 << x for x in nodes)
        uf = UnionFind(36)
        for xx in range(1, 1 << m):
            if xx & target != target:
                continue
            mat = [[0] * 6 for _ in range(6)]
            uf.initialize()
            ind = [i for i in range(16) if xx & (1 << i)]
            for i in ind:
                mat[i // 4 + 1][i % 4 + 1] = 1
            edge = 0
            for i in range(6):
                for j in range(6):
                    if i + 1 < 6 and mat[i + 1][j] == mat[i][j]:
                        uf.union(i * 6 + j, i * 6 + j + 6)
                        edge += 1
                    if j + 1 < 6 and mat[i][j + 1] == mat[i][j]:
                        uf.union(i * 6 + j, i * 6 + j + 1)
                        edge += 1
            if uf.part == 2:
                ans += 1
        ac.st(ans)
        return

    @staticmethod
    def lc_3272(n: int, k: int) -> int:

        """
        url: https://leetcode.cn/problems/find-the-count-of-good-integers/
        tag: brute_force|palindrome_num
        """
        length = n // 2 + n % 2
        ans = 0
        dct = set()
        for x in range(10 ** (length - 1), 10 ** length):
            s = str(x)
            if n % 2:
                s = s + s[:-1][::-1]
            else:
                s += s[::-1]

            if int(s) % k == 0:
                sk = "".join(sorted(s))
                if sk in dct:
                    continue
                dct.add(sk)
                cnt = Counter(sk)
                pre = n
                cur = 1
                if cnt["0"]:
                    cur = math.comb(pre - 1, cnt["0"])
                    pre -= cnt["0"]
                for w in cnt:
                    if w != "0":
                        cur = cur * math.comb(pre, cnt[w])
                        pre -= cnt[w]
                ans += cur
        return ans

    @staticmethod
    def abc_176e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc176/tasks/abc176_e
        tag: brute_force|observation|greedy
        """
        m, n, k = ac.read_list_ints()
        row = defaultdict(list)
        col = [0] * n
        for _ in range(k):
            x, y = ac.read_list_ints_minus_one()
            row[x].append(y)
            col[y] += 1

        ceil = max(col)
        cnt = col.count(ceil)
        ans = ceil
        for r in row:
            cur = len(row[r])
            c = sum(col[y] == ceil for y in row[r])
            cur += ceil if c < cnt else ceil - 1
            ans = max(cur, ans)
        ac.st(ans)
        return

    @staticmethod
    def abc_173f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc173/tasks/abc173_f
        tag: union_find|contribution_method|brute_force
        """
        n = ac.read_int()
        ans = 0
        for i in range(1, n + 1):
            x = n - i + 1
            ans += x * (x + 1) // 2
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            if i > j:
                i, j = j, i
            ans -= (i + 1) * (n - j)
        ac.st(ans)
        return

    @staticmethod
    def abc_165c(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc165/tasks/abc165_c
        tag: brute_force|dfs
        """

        n, m, q = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(q)]

        def dfs(i):
            if i == n:
                ans[0] = max(ans[0], sum(d if res[b - 1] - res[a - 1] == c else 0 for a, b, c, d in nums))
                return

            low = 1 if i == 0 else res[i - 1]
            for x in range(low, m + 1):
                res[i] = x
                dfs(i + 1)
            return

        res = [0] * n
        ans = [0]

        dfs(0)

        ac.st(ans[0])
        return

    @staticmethod
    def lc_3405(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/count-the-number-of-arrays-with-k-matching-adjacent-elements/
        tag: counter|brute_force|hash|preprocess
        """
        ceil = 1000
        dp = [[0] * (ceil + 1) for _ in range(ceil + 1)]

        for x in range(1, ceil + 1):
            for y in range(1, ceil + 1):
                g = math.gcd(x, y)
                dp[x][y] = (x // g) * (ceil + 1) + y // g

        n = len(nums)
        ans = 0
        post = dict()
        for r in range(n - 1, -1, -1):
            for s in range(r + 2, n):
                post[dp[nums[s]][nums[r]]] += post.get(dp[nums[s]][nums[r]], 0) + 1

        for q in range(n):
            pre = defaultdict(int)
            for p in range(q - 1):
                pre[dp[nums[p]][nums[q]]] += 1
                ans += post.get(dp[nums[p]][nums[q]], 0)
            for s in range(q + 3, n):
                ans -= pre[dp[nums[s]][nums[q + 1]]]
            for s in range(q + 2, n):
                ans -= pre[dp[nums[s]][nums[q]]]
            for s in range(q + 2, n):
                post[dp[nums[s]][nums[q]]] -= 1
        return ans
