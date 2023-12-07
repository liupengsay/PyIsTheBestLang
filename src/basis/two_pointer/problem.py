"""
Algorithm：two_pointer、快慢pointer、先后pointer、bucket_counter
Function：通过相对移动，来减少复杂度，分为同向two_pointer，相反two_pointer，以及中心扩展法


====================================LeetCode====================================
167（https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/）two_pointer应用
259（https://leetcode.com/problems/3sum-smaller/）two_pointer或者counterbrute_force的方式
2444（https://leetcode.com/problems/count-subarrays-with-fixed-bounds/）通向two_pointer的移动来根据两个pointer的位置来counter
2398（https://leetcode.com/problems/maximum-number-of-robots-within-budget/）同向two_pointer移动的条件限制有两个需要用sorted_list来维护sliding_window过程
2302（https://leetcode.com/problems/count-subarrays-with-score-less-than-k/）同向two_pointer维护pointer位置与counter
2301（https://leetcode.com/problems/match-substring-after-replacement/）brute_force匹配字符起点并two_pointer维护可行长度
2106（https://leetcode.com/problems/maximum-fruits-harvested-after-at-most-k-steps/）巧妙利用行走距离的更新two_pointer
6293（https://leetcode.com/problems/count-the-number-of-good-subarrays/）two_pointercounter
16（https://leetcode.com/problems/3sum-closest/）三pointer确定最接近目标值的和
15（https://leetcode.com/problems/3sum/）寻找三个元素和为 0 的不重复组合
2422（https://leetcode.com/problems/merge-operations-to-turn-array-into-a-palindrome/）相反方向two_pointergreedy|和
2524（https://leetcode.com/problems/maximum-frequency-score-of-a-subarray/）sliding_window维护数字数量与幂次mod|
239（https://leetcode.com/problems/sliding-window-maximum/）sliding_window最大值，sliding_window类维护
2447（https://leetcode.com/problems/number-of-subarrays-with-gcd-equal-to-k/）sliding_window区间 gcd，sliding_window类维护
6392（https://leetcode.com/problems/minimum-number-of-operations-to-make-all-array-elements-equal-to-1/）sliding_window区间 gcd，sliding_window类维护
1163（https://leetcode.com/problems/last-substring-in-lexicographical-order/）类似最小表示法的two_pointer
2555（https://leetcode.com/problems/maximize-win-from-two-segments/description/）同向two_pointer|线性DP
992（https://leetcode.com/problems/subarrays-with-k-different-integers/）三pointer，即快慢two_pointer维护连续子区间个数
2747（https://leetcode.com/problems/count-zero-request-servers/）offline_query与三pointer，即快慢two_pointer维护连续区间的不同值个数
2516（https://leetcode.com/problems/take-k-of-each-character-from-left-and-right/）reverse_thinkinginclusion_exclusiontwo_pointer
1537（https://leetcode.com/problems/get-the-maximum-score/description/）two_pointer|线性DP或者拓扑sorting做
1712（https://leetcode.com/problems/ways-to-split-array-into-three-subarrays/description/）三pointer，即快慢two_pointer维护满足条件的分割点个数
948（https://leetcode.com/problems/bag-of-tokens/description/）two_pointergreedy
2953（https://leetcode.com/contest/weekly-contest-374/problems/count-complete-substrings/）two pointers|brute force

=====================================LuoGu======================================
2381（https://www.luogu.com.cn/problem/P2381）环形数组，sliding_windowtwo_pointer
3353（https://www.luogu.com.cn/problem/P3353）sliding_window|two_pointer
3662（https://www.luogu.com.cn/problem/P3662）滑动子数组和
4995（https://www.luogu.com.cn/problem/P4995）sorting后利用greedy与two_pointerimplemention
2207（https://www.luogu.com.cn/problem/P2207）greedy|同向two_pointer
7542（https://www.luogu.com.cn/problem/P7542）bucket_counter|two_pointer
4653（https://www.luogu.com.cn/problem/P4653）greedysorting后two_pointer
3029（https://www.luogu.com.cn/problem/P3029）two_pointer记录包含k个不同颜色的最短连续子序列
5583（https://www.luogu.com.cn/problem/P5583）two_pointer
6465（https://www.luogu.com.cn/problem/P6465）sliding_window与two_pointercounter


===================================CodeForces===================================
1328D（https://codeforces.com/problemset/problem/1328/D）环形数组sliding_window，记录变化次数并根据奇偶变换次数与环形首尾元素确定染色数量
1333C（https://codeforces.com/problemset/problem/1333/C）two_pointer，prefix_sum不重复即没有区间段和为0的个数
1381A2（https://codeforces.com/problemset/problem/1381/A2）two_pointerimplemention翻转匹配与greedy

====================================AtCoder=====================================
D - Equal Cut（https://atcoder.jp/contests/abc102/tasks/arc100_b）two_pointerbrute_force

=====================================AcWing=====================================
4217（https://www.acwing.com/problem/content/4220/）two_pointersliding_window题目

"""
import math
from collections import Counter, defaultdict
from functools import reduce
from itertools import accumulate
from math import gcd, inf
from operator import add
from typing import List

from src.basis.two_pointer.template import SlidingWindowAggregation, INF
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lg_p4653(ac=FastIO()):

        # greedysorting后two_pointer
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
        # 寻找最接近目标值的三个元素和
        n = len(nums)
        nums.sort()
        ans = nums[0] + nums[1] + nums[2]
        for i in range(n - 2):
            j, k = i + 1, n - 1
            x = nums[i]
            while j < k:  # 遍历数组作为第一个pointer，另外两个pointer相向而行
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
        # 寻找三个元素和为 0 的不重复组合
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
        # 相反方向的two_pointer统计和小于 target 的三元组数量
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
        # sliding_window最大值
        n = len(nums)
        swa = SlidingWindowAggregation(-INF, max)
        ans = []
        for i in range(n):
            swa.append(nums[i])
            if i >= k - 1:
                ans.append(swa.query())
                swa.popleft()
        return ans

    @staticmethod
    def lc_2516(s: str, k: int) -> int:
        # reverse_thinkinginclusion_exclusiontwo_pointer
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
        # 同向two_pointer|线性DP
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
        # offline_query与三pointer，即快慢two_pointer维护连续区间的不同值个数
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
    def lc_6392(nums: List[int]) -> int:
        # sliding_window维护区间 gcd 为 1 的长度信息
        if gcd(*nums) != 1:
            return -1
        if 1 in nums:
            return len(nums) - nums.count(1)

        swa = SlidingWindowAggregation(0, gcd)
        res, n = INF, len(nums)
        # brute_force右端点
        for i in range(n):
            swa.append(nums[i])
            while swa and swa.query() == 1:
                res = res if res < swa.size else swa.size
                swa.popleft()
        return res - 1 + len(nums) - 1

    @staticmethod
    def lc_992(nums: List[int], k: int) -> int:
        # 三pointer，即快慢two_pointer维护连续子区间个数
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
        # two_pointer
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
        # two_pointer|线性DP或者拓扑sorting做
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
        # 三pointer，即快慢two_pointer维护满足条件的分割点个数
        mod = 10 ** 9 + 7
        ans = 0
        pre = list(accumulate(nums, initial=0))
        j1 = j2 = 0
        n = len(nums)
        for i in range(n):
            # mid的区间范围为 [i+1, j1~(j2-1)]
            while j1 <= i or (j1 < n and pre[j1 + 1] - pre[i + 1] < pre[i + 1]):
                j1 += 1  # j1 必须大于 i 且 mid >= left 区间的和
            while j2 < j1 or (j2 < n - 1 and pre[-1] - pre[j2 + 1] >= pre[j2 + 1] - pre[i + 1]):
                j2 += 1  # j2 必须大于等于j1 且 mid < right 非空因此 j2 < n-1
            if j2 >= j1:  # 此时[i+1, j1] 到 区间[i+1, j2-1] 是合法的
                ans += j2 - j1
        return ans % mod

    @staticmethod
    def lc_2447(nums: List[int], k: int) -> int:
        # sliding_windowtwo_pointer三pointer维护区间 gcd 为 k 的子数组数量信息
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
        # two_pointer与变量维护区间信息
        n, m, d = ac.read_list_ints()
        nums = ac.read_list_ints()
        cnt = dict()
        for num in nums:
            cnt[num] = cnt.get(num, 0) + 1
        nums = [ac.read_list_ints() for _ in range(n)]
        # 动态维护的变量
        flag = 0
        ans = [-1]
        not_like = inf
        power = -inf
        cur_cnt = defaultdict(int)
        cur_power = cur_not_like = j = 0
        for i in range(n):
            # 移动右pointer
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
            # 删除左pointer
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
        # sliding_window与two_pointercounter
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            nums = ac.read_list_ints()
            m = ac.max(m, 2)
            ans = j = s = 0
            cnt = dict()
            for i in range(n):
                if i and nums[i] == nums[i - 1]:
                    cnt = dict()
                    s = 0
                    j = i
                    continue
                while j <= i - m + 1:
                    cnt[nums[j]] = cnt.get(nums[j], 0) + 1
                    s += 1
                    j += 1
                ans += s - cnt.get(nums[i], 0)
            ac.st(ans)
        return

    @staticmethod
    def ac_4217(ac=FastIO()):
        # two_pointer移动
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