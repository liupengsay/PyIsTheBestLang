"""
Algorithm：two_pointers|fast_slow_pointers|bucket_counter|tree_pointers
Description：sliding_window|two_pointers|center_extension_method

====================================LeetCode====================================
167（https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/）two_pointers
259（https://leetcode.cn/problems/3sum-smaller/）two_pointers|counter|brute_force
2444（https://leetcode.cn/problems/count-subarrays-with-fixed-bounds/）same_direction|two_pointers|counter
2398（https://leetcode.cn/problems/maximum-number-of-robots-within-budget/）same_direction|two_pointers|sorted_list|sliding_window
2302（https://leetcode.cn/problems/count-subarrays-with-score-less-than-k/）same_direction|two_pointers|pointer|counter
2301（https://leetcode.cn/problems/match-substring-after-replacement/）brute_force|two_pointers
2106（https://leetcode.cn/problems/maximum-fruits-harvested-after-at-most-k-steps/）two_pointers
6293（https://leetcode.cn/problems/count-the-number-of-good-subarrays/）two_pointers|counter
16（https://leetcode.cn/problems/3sum-closest/）tree_pointers
15（https://leetcode.cn/problems/3sum/）two_pointers
2422（https://leetcode.cn/problems/range_merge_to_disjoint-operations-to-turn-array-into-a-palindrome/）opposite_direction|two_pointers|greedy
2524（https://leetcode.cn/problems/maximum-frequency-score-of-a-subarray/）sliding_window|mod|power
239（https://leetcode.cn/problems/sliding-window-maximum/）sliding_window
2447（https://leetcode.cn/problems/number-of-subarrays-with-gcd-equal-to-k/）sliding_window|gcd
2654（https://leetcode.cn/problems/minimum-number-of-operations-to-make-all-array-elements-equal-to-1/）sliding_window|gcd
1163（https://leetcode.cn/problems/last-substring-in-lexicographical-order/）minimum_expression|two_pointers
2555（https://leetcode.cn/problems/maximize-win-from-two-segments/description/）same_direction|two_pointers|liner_dp
992（https://leetcode.cn/problems/subarrays-with-k-different-integers/）tree_pointers|fast_slow_pointers
2747（https://leetcode.cn/problems/count-zero-request-servers/）offline_query|tree_pointers|fast_slow_pointers
2516（https://leetcode.cn/problems/take-k-of-each-character-from-left-and-right/）reverse_thinking|inclusion_exclusion|two_pointers
1537（https://leetcode.cn/problems/get-the-maximum-score/description/）two_pointers|liner_dp|topological_sorting
1712（https://leetcode.cn/problems/ways-to-split-array-into-three-subarrays/description/）three_pointers|fast_slow_pointers
948（https://leetcode.cn/problems/bag-of-tokens/）two_pointers|greedy
2953（https://leetcode.cn/problems/count-complete-substrings/）two pointers|brute force

=====================================LuoGu======================================
P2381（https://www.luogu.com.cn/problem/P2381）circular_array|sliding_window|two_pointers
P3353（https://www.luogu.com.cn/problem/P3353）sliding_window|two_pointers
P3662（https://www.luogu.com.cn/problem/P3662）sliding_window|sub_consequence_sum
P4995（https://www.luogu.com.cn/problem/P4995）sort|greedy|two_pointers|implemention
P2207（https://www.luogu.com.cn/problem/P2207）greedy|same_direction|two_pointers
P7542（https://www.luogu.com.cn/problem/P7542）bucket_counter|two_pointers
P4653（https://www.luogu.com.cn/problem/P4653）greedy|sort|two_pointers
P3029（https://www.luogu.com.cn/problem/P3029）two_pointers
P5583（https://www.luogu.com.cn/problem/P5583）two_pointers
P6465（https://www.luogu.com.cn/problem/P6465）sliding_window|two_pointers|counter


===================================CodeForces===================================
1328D（https://codeforces.com/problemset/problem/1328/D）circular_array|sliding_window|odd_even
1333C（https://codeforces.com/problemset/problem/1333/C）two_pointers|prefix_sum
1381A2（https://codeforces.com/problemset/problem/1381/A2）two_pointers|implemention|reverse_array|greedy
1611F（https://codeforces.com/contest/1611/problem/F）two_pointers|classical
1413C（https://codeforces.com/problemset/problem/1413/C）two_pointers

====================================AtCoder=====================================
ARC100B（https://atcoder.jp/contests/abc102/tasks/arc100_b）two_pointers|brute_force
ABC337F（https://atcoder.jp/contests/abc337/tasks/abc337_f）two_pointers|implemention|greedy
ABC353C（https://atcoder.jp/contests/abc353/tasks/abc353_c）two_pointers|brute_force

=====================================AcWing=====================================
4217（https://www.acwing.com/problem/content/4220/）two_pointers|sliding_window

"""
import math
from collections import Counter, defaultdict
from functools import reduce
from itertools import accumulate
from math import gcd, inf
from operator import add
from typing import List

from src.basis.two_pointer.template import SlidingWindowAggregation
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lg_p4653(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4653
        tag: greedy|sort|two_pointers|brain_teaser
        """
        n = ac.read_int()

        nums1 = []
        nums2 = []
        for _ in range(n):
            x, y = ac.read_list_floats()
            nums1.append(x)
            nums2.append(y)
        nums1.sort(reverse=True)
        nums2.sort(reverse=True)
        nums1 = list(accumulate(nums1, add))
        nums2 = list(accumulate(nums2, add))

        ans = i = j = 0
        while i < n and j < n:
            if nums1[i] < nums2[j]:
                ans = ac.max(ans, nums1[i] - i - j - 2)
                i += 1
            else:
                ans = ac.max(ans, nums2[j] - i - j - 2)
                j += 1
        ac.st("%.4f" % ans)
        return

    @staticmethod
    def lc_16(nums, target):
        """
        url: https://leetcode.cn/problems/3sum-closest/
        tag: tree_pointers|classical
        """
        n = len(nums)
        nums.sort()
        ans = nums[0] + nums[1] + nums[2]
        for i in range(n - 2):
            j, k = i + 1, n - 1
            x = nums[i]
            while j < k:
                cur = x + nums[j] + nums[k]
                ans = ans if abs(target - ans) < abs(target - cur) else cur
                if cur > target:
                    k -= 1
                elif cur < target:
                    j += 1
                else:
                    return target
        return ans

    @staticmethod
    def lc_15(nums):
        """
        url: https://leetcode.cn/problems/3sum/
        tag: two_pointers|classical
        """
        nums.sort()
        n = len(nums)
        ans = set()
        for i in range(n - 2):
            j, k = i + 1, n - 1
            x = nums[i]
            while j < k:
                cur = x + nums[j] + nums[k]
                if cur > 0:
                    k -= 1
                elif cur < 0:
                    j += 1
                else:
                    ans.add((x, nums[j], nums[k]))
                    j += 1
                    k -= 1
        return [list(a) for a in ans]

    @staticmethod
    def lc_259(nums: List[int], target: int) -> int:
        """
        url: https://leetcode.cn/problems/3sum-smaller/
        tag: two_pointers|counter|brute_force
        """
        nums.sort()
        n = len(nums)
        ans = 0
        for i in range(n - 2):
            x = nums[i]
            j, k = i + 1, n - 1
            while j < k:
                cur = x + nums[j] + nums[k]
                if cur < target:
                    ans += k - j
                    j += 1
                else:
                    k -= 1
        return ans

    @staticmethod
    def lc_239(nums: List[int], k: int) -> List[int]:
        """
        url: https://leetcode.cn/problems/sliding-window-maximum/
        tag: sliding_window
        """
        n = len(nums)
        swa = SlidingWindowAggregation(-inf, max)
        ans = []
        for i in range(n):
            swa.append(nums[i])
            if i >= k - 1:
                ans.append(swa.query())
                swa.popleft()
        return ans

    @staticmethod
    def lc_2516(s: str, k: int) -> int:
        """
        url: https://leetcode.cn/problems/take-k-of-each-character-from-left-and-right/
        tag: reverse_thinking|inclusion_exclusion|two_pointers
        """
        cnt = Counter(s)
        n = len(s)
        if any(cnt[w] < k for w in "abc"):
            return -1
        ans = 0
        dct = defaultdict(int)
        j = 0
        for i in range(n):
            while j < n and cnt[s[j]] - dct[s[j]] - 1 >= k:
                dct[s[j]] += 1
                j += 1
            if j - i > ans:
                ans = j - i
            dct[s[i]] -= 1
        return n - ans

    @staticmethod
    def lc_2555(prize_positions: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/maximize-win-from-two-segments/description/
        tag: same_direction|two_pointers|liner_dp
        """
        n = len(prize_positions)

        pre = [0] * n
        pre_max = [0] * (n + 1)
        ans = 0
        i = 0
        for j in range(n):
            while prize_positions[j] - prize_positions[i] > k:
                i += 1
            pre[j] = j - i + 1
            ans = max(ans, pre[j] + pre_max[i])
            pre_max[j + 1] = max(pre_max[j], pre[j])
        return ans

    @staticmethod
    def lc_2747(n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/count-zero-request-servers/
        tag: offline_query|tree_pointers|fast_slow_pointers
        """
        m = len(queries)
        ans = [0] * m
        ind = list(range(m))
        ind.sort(key=lambda it: queries[it])
        logs.sort(key=lambda it: it[1])
        i1 = i2 = 0
        k = len(logs)
        dct = dict()
        for j in ind:
            left = queries[j] - x
            right = queries[j]
            while i2 < k and logs[i2][1] <= right:
                dct[logs[i2][0]] = dct.get(logs[i2][0], 0) + 1
                i2 += 1
            while i1 < k and logs[i1][1] < left:
                dct[logs[i1][0]] -= 1
                if not dct[logs[i1][0]]:
                    del dct[logs[i1][0]]
                i1 += 1
            ans[j] = n - len(dct)
        return ans

    @staticmethod
    def lc_2654(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-number-of-operations-to-make-all-array-elements-equal-to-1/
        tag: sliding_window|gcd
        """
        if gcd(*nums) != 1:
            return -1
        if 1 in nums:
            return len(nums) - nums.count(1)

        swa = SlidingWindowAggregation(0, gcd)
        res, n = inf, len(nums)
        for i in range(n):
            swa.append(nums[i])
            while swa and swa.query() == 1:
                res = res if res < swa.size else swa.size
                swa.popleft()
        return res - 1 + len(nums) - 1

    @staticmethod
    def lc_992(nums: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/subarrays-with-k-different-integers/
        tag: tree_pointers|fast_slow_pointers|classical
        """
        n = len(nums)
        ans = j1 = j2 = 0
        pre1 = dict()
        pre2 = dict()
        for i in range(n):
            while j1 < n and len(pre1) < k:
                pre1[nums[j1]] = pre1.get(nums[j1], 0) + 1
                j1 += 1

            while j2 < n and (len(pre2) < k or nums[j2] in pre2):
                pre2[nums[j2]] = pre2.get(nums[j2], 0) + 1
                j2 += 1

            if len(pre1) == k:
                ans += j2 - j1 + 1
            pre1[nums[i]] -= 1
            if not pre1[nums[i]]:
                pre1.pop(nums[i])
            pre2[nums[i]] -= 1
            if not pre2[nums[i]]:
                pre2.pop(nums[i])
        return ans

    @staticmethod
    def lc_1163(s: str) -> str:
        """
        url: https://leetcode.cn/problems/last-substring-in-lexicographical-order/
        tag: minimum_expression|two_pointers|classical
        """
        i, j, n = 0, 1, len(s)
        while j < n:
            k = 0
            while j + k < n and s[i + k] == s[j + k]:
                k += 1
            if j + k < n and s[i + k] < s[j + k]:
                i, j = j, j + 1 if j + 1 > i + k + 1 else i + k + 1
            else:
                j = j + k + 1
        return s[i:]

    @staticmethod
    def lc_1537(nums1: List[int], nums2: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/get-the-maximum-score/description/
        tag: two_pointers|liner_dp|topological_sort
        """
        mod = 10 ** 9 + 7
        m, n = len(nums1), len(nums2)
        i = j = pre1 = pre2 = 0
        while i < m and j < n:
            if nums1[i] < nums2[j]:
                pre1 += nums1[i]
                i += 1
            elif nums1[i] > nums2[j]:
                pre2 += nums2[j]
                j += 1
            else:
                pre1 += nums1[i]
                pre2 += nums2[j]
                pre1 = pre2 = pre1 if pre1 > pre2 else pre2
                i += 1
                j += 1
        pre1 += sum(nums1[i:])
        pre2 += sum(nums2[j:])
        return max(pre1, pre2) % mod

    @staticmethod
    def lc_1712(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/ways-to-split-array-into-three-subarrays/description/
        tag: three_pointers|fast_slow_pointers
        """
        mod = 10 ** 9 + 7
        ans = 0
        pre = list(accumulate(nums, initial=0))
        j1 = j2 = 0
        n = len(nums)
        for i in range(n):
            while j1 <= i or (j1 < n and pre[j1 + 1] - pre[i + 1] < pre[i + 1]):
                j1 += 1
            while j2 < j1 or (j2 < n - 1 and pre[-1] - pre[j2 + 1] >= pre[j2 + 1] - pre[i + 1]):
                j2 += 1
            if j2 >= j1:
                ans += j2 - j1
        return ans % mod

    @staticmethod
    def lc_2447(nums: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/number-of-subarrays-with-gcd-equal-to-k/
        tag: sliding_window|gcd
        """
        n = len(nums)
        e = reduce(math.lcm, nums + [k])
        e *= 2

        swa1 = SlidingWindowAggregation(e, gcd)
        ans = j1 = j2 = 0
        swa2 = SlidingWindowAggregation(e, gcd)
        for i in range(n):

            if j1 < i:
                swa1 = SlidingWindowAggregation(e, gcd)
                j1 = i
            if j2 < i:
                swa2 = SlidingWindowAggregation(e, gcd)
                j2 = i

            while j1 < n and swa1.query() > k:
                swa1.append(nums[j1])
                j1 += 1
                if swa1.query() == k:
                    break

            while j2 < n and gcd(swa2.query(), nums[j2]) >= k:
                swa2.append(nums[j2])
                j2 += 1

            if swa1.query() == k:
                ans += j2 - j1 + 1

            swa1.popleft()
            swa2.popleft()
        return ans

    @staticmethod
    def lg_p5583(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5583
        tag: two_pointers
        """
        n, m, d = ac.read_list_ints()
        nums = ac.read_list_ints()
        cnt = dict()
        for num in nums:
            cnt[num] = cnt.get(num, 0) + 1
        nums = [ac.read_list_ints() for _ in range(n)]
        flag = 0
        ans = [-1]
        not_like = inf
        power = -inf
        cur_cnt = defaultdict(int)
        cur_power = cur_not_like = j = 0
        for i in range(n):
            while j < n and (flag < len(cnt) or all(num in cnt for num in nums[j][2:])):
                cur_power += nums[j][0]
                for num in nums[j][2:]:
                    cur_cnt[num] += 1
                    if cur_cnt[num] == 1 and num in cnt:
                        flag += 1
                    if num not in cnt:
                        cur_not_like += 1
                j += 1
            if flag == len(cnt):
                if cur_not_like < not_like or (cur_not_like == not_like and cur_power > power):
                    not_like = cur_not_like
                    power = cur_power
                    ans = [i + 1, j]
            cur_power -= nums[i][0]
            for num in nums[i][2:]:
                cur_cnt[num] -= 1
                if cur_cnt[num] == 0 and num in cnt:
                    flag -= 1
                if num not in cnt:
                    cur_not_like -= 1
        ac.lst(ans)
        return

    @staticmethod
    def lg_p6465(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6465
        tag: sliding_window|two_pointers|counter
        """
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            nums = ac.read_list_ints()
            m = m if m > 2 else 2
            ans = j = s = 0
            cnt = dict()
            for i in range(n):
                x = nums[i]
                if i and x == nums[i - 1]:
                    cnt = dict()
                    s = 0
                    j = i
                    continue
                while j <= i - m + 1:
                    y = nums[j]
                    cnt[y] = cnt.get(y, 0) + 1
                    s += 1
                    j += 1
                ans += s - cnt.get(x, 0)
            ac.st(ans)
        return

    @staticmethod
    def ac_4217(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4220/
        tag: two_pointers|sliding_window
        """
        n = ac.read_int()
        s = ac.read_str()
        a, b = ac.read_list_ints()
        ans = inf
        j = 0
        ind = dict()
        ind["U"] = [0, 1]
        ind["D"] = [0, -1]
        ind["L"] = [-1, 0]
        ind["R"] = [1, 0]
        lst = [0, 0]
        for w in s:
            lst[0] += ind[w][0]
            lst[1] += ind[w][1]

        def check():
            rest = [lst[0] - pre[0], lst[1] - pre[1]]
            return abs(rest[0] - a) + abs(rest[1] - b) == j - i

        pre = [0, 0]
        for i in range(n):
            while j < n and not check():
                w = ind[s[j]]
                pre[0] += w[0]
                pre[1] += w[1]
                j += 1
            if check() and j - i < ans:
                ans = j - i
            pre[0] -= ind[s[i]][0]
            pre[1] -= ind[s[i]][1]
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def abc_337f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc337/tasks/abc337_f
        tag: two_pointers|implemention|greedy
        """
        n, m, k = ac.read_list_ints()
        cnt = [0] * (n + 1)
        nums = ac.read_list_ints()
        for num in nums:
            cnt[num] += 1
        j = 0
        color = 0
        cur = dict()
        tot = 0
        for i in range(n):
            while color < m and j < i + n:
                cur[nums[j]] = cur.get(nums[j], 0) + 1
                if k > 1:
                    if cur[nums[j]] == 1:
                        tot += ac.min(cnt[nums[j]], k)
                        color += 1
                    elif cur[nums[j]] % k == 1:
                        color += 1
                        x = cur[nums[j]] // k
                        tot += ac.min(k, cnt[nums[j]] - x * k)
                else:
                    tot += 1
                    color += 1
                j += 1
            ac.st(tot)
            nums.append(nums[i])
            if k > 1:
                cur[nums[i]] -= 1
                if cur[nums[i]] % k == 0:
                    x = cur[nums[i]] // k
                    color -= 1
                    tot -= ac.min(k, cnt[nums[i]] - x * k)
            else:
                cur[nums[i]] -= 1

                color -= 1
                tot -= 1
        return
