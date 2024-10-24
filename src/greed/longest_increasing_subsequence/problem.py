"""

Algorithm：longest_increasing_subsequence|lis|lds|longest_decreasing_subsequence|longest_monotonic_subsequence|lms
Description：longest_non_increasing_subsequence|longest_non_decreasing_subsequence|dilworth|lexicographical_order
Dilworth：
minimum_group_non_decreasing_subsequence_partition=length_of_longest_increasing_subsequence
minimum_group_non_increasing_subsequence_partition=length_of_longest_decreasing_subsequence
minimum_group_increasing_subsequence_partition=length_of_longest_non_increasing_subsequence
minimum_group_decreasing_subsequence_partition=length_of_longest_non_decreasing_subsequence

====================================LeetCode====================================
354（https://leetcode.cn/problems/russian-doll-envelopes/）partial_order|lis
673（https://leetcode.cn/problems/number-of-longest-increasing-subsequence/）lis|counter|O(nlogn)|classical
1092（https://leetcode.cn/problems/shortest-common-supersequence/）lcs_by_lis|super_sequence
1671（https://leetcode.cn/problems/minimum-number-of-removals-to-make-mountain-array/）lis|prefix_suffix
2111（https://leetcode.cn/problems/minimum-operations-to-make-the-array-k-increasing/）lis|dp|greed
17（https://leetcode.cn/problems/circus-tower-lcci/）partial_order|greed|sort|lis
1691（https://leetcode.cn/problems/maximum-height-by-stacking-cuboids/submissions/）md_partial_order
1713（https://leetcode.cn/problems/minimum-operations-to-make-a-subsequence/）lcs_by_lis
1940（https://leetcode.cn/problems/longest-common-subsequence-between-sorted-arrays/）lcs_by_lis
2826（https://leetcode.cn/problems/sorting-three-groups/）longest_non_decreasing_subsequence|classical
1964（https://leetcode.cn/problems/find-the-longest-valid-obstacle-course-at-each-position/）lis
2945（https://leetcode.cn/problems/find-maximum-non-decreasing-array-length/description/）linear dp|deque|greed|prefix_sum
1035（https://leetcode.cn/problems/uncrossed-lines/description/）lcs|classical
3288（https://leetcode.cn/problems/length-of-the-longest-increasing-path/）lis|partial_order|classical

===================================CodeForces===================================
1682C（https://codeforces.com/contest/1682/problem/C）lis|lds|greed|counter
486E（https://codeforces.com/problemset/problem/486/E）lis|greed|brain_teaser|classical
650D（https://codeforces.com/problemset/problem/650/D）lis|brain_teaser|classical|offline_query
1922E（https://codeforces.com/problemset/problem/1922/E）lis|construction|divide_and_conquer

=====================================LuoGu======================================
P1020（https://www.luogu.com.cn/problem/P1020）greed|binary_search|longest_non_increasing_subsequence|longest_non_decreasing_subsequence
P1439（https://www.luogu.com.cn/problem/P1439）greed|binary_search|lis
P1091（https://www.luogu.com.cn/problem/P1091）prefix_suffix|lis
P1233（https://www.luogu.com.cn/problem/P1233）partial_order|lis
P2782（https://www.luogu.com.cn/problem/P2782）partial_order|lis
P3902（https://www.luogu.com.cn/problem/P3902）lis
P6403（https://www.luogu.com.cn/problem/P6403）longest_non_decreasing_subsequence
P5939（https://www.luogu.com.cn/problem/P5939）lis
P5978（https://www.luogu.com.cn/problem/P5978）lis|greed|brute_force
P7957（https://www.luogu.com.cn/problem/P7957）lis|lds|construction
P1410（https://www.luogu.com.cn/problem/P1410）dilworth|lis
P2516（https://www.luogu.com.cn/problem/P2516）length_of_lcs|cnt_of_lcs
P1108（https://www.luogu.com.cn/problem/P1108）matrix_dp|lis|classical|brain_teaser

=====================================AcWing=====================================
3549（https://www.acwing.com/problem/content/3552/）liner_dp|greed
2694（https://www.acwing.com/problem/content/description/2696/）lcs_by_lis|counter|dp
3662（https://www.acwing.com/problem/content/description/3665/）lis|counter|discretization|tree_array|liner_dp|segment_tree

====================================AtCoder=====================================
ABC134E（https://atcoder.jp/contests/abc134/tasks/abc134_e）minimum_group_increasing_subsequence_partition|length_of_longest_non_increasing_subsequence
ABC354F（https://atcoder.jp/contests/abc354/tasks/abc354_f）lis|classical
ABC360G（https://atcoder.jp/contests/abc360/tasks/abc360_g）lis|greed|implemention|classical|linear_dp|prefix_suffix

（https://www.nowcoder.com/questionTerminal/30fb9b3cab9742ecae9acda1c75bf927?orderByHotValue=1&questionTypes=000100&difficulty=11111&mutiTagIds=593&page=10&onlyReference=false）lis|lexicographical_order

"""

import bisect
import random
from collections import deque, Counter
from itertools import accumulate

from typing import List

from src.structure.segment_tree.template import RangeAscendRangeMax
from src.structure.tree_array.template import PointAscendPreMax
from src.greed.longest_increasing_subsequence.template import LongestIncreasingSubsequence, LcsComputeByLis
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def abc_134e(ac=FastIO()):
        # 分成最少组数的上升子序列，等于最长不上升的子序列长度
        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]
        ans = LongestIncreasingSubsequence().definitely_not_increase(nums)
        ac.st(ans)
        return

    @staticmethod
    def lc_1713(target: List[int], arr: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-operations-to-make-a-subsequence/
        tag: lcs_by_lis
        """
        # 最长递增子序列模板题
        ind = {num: i for i, num in enumerate(target)}
        lst = [ind[num] for num in arr if num in ind]
        return len(target) - LongestIncreasingSubsequence().definitely_increase(lst)

    @staticmethod
    def lc_1964(obstacles: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/find-the-longest-valid-obstacle-course-at-each-position/
        tag: lis
        """
        # LIS求以每个位置结尾的最长不降子序列长度
        pre = []
        dp = []
        for num in obstacles:
            i = bisect.bisect_right(dp, num)
            if 0 <= i < len(dp):
                dp[i] = num
                pre.append(i + 1)
            else:
                dp.append(num)
                pre.append(len(dp))
        return pre

    @staticmethod
    def lc_2111(arr: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/minimum-operations-to-make-the-array-k-increasing/
        tag: lis|dp|greed
        """
        # 最长不降子序列
        ans = 0
        for i in range(k):
            lst = arr[i::k]
            ans += len(lst) - LongestIncreasingSubsequence().definitely_not_reduce(lst)
        return ans

    @staticmethod
    def lc_2826(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/sorting-three-groups/
        tag: longest_non_decreasing_subsequence|classical
        """
        # 转换为求最长不降子序列
        n = len(nums)
        return n - LongestIncreasingSubsequence().definitely_not_reduce(nums)

    @staticmethod
    def lc_2945(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/find-maximum-non-decreasing-array-length/description/
        tag: linear dp|deque|greed|prefix_sum
        """
        n = len(nums)
        stack = deque([0])
        dp = [0] * (n + 1)
        last = [0] * (n + 1)
        pre = list(accumulate(nums, initial=0))
        for i in range(n):
            while len(stack) > 1 and last[stack[1]] + pre[stack[1]] <= pre[i + 1]:
                stack.popleft()
            dp[i + 1] = dp[stack[0]] + 1
            last[i + 1] = pre[i + 1] - pre[stack[0]]
            while stack and last[stack[-1]] + pre[stack[-1]] >= last[i + 1] + pre[i + 1]:
                stack.pop()
            stack.append(i + 1)
        return dp[n]

    @staticmethod
    def lc_p1020(ac=FastIO()):
        # 根据 dilworth 最长不升子序列的长度与分成不降子序列的最小组数（最长上升子序列的长度）
        nums = ac.read_list_ints()
        lis = LongestIncreasingSubsequence()
        ac.st(lis.definitely_not_increase(nums))
        ac.st(lis.definitely_increase(nums))
        return

    @staticmethod
    def lg_1439(ac=FastIO()):
        # 最长公共子序列求解hash映射转换为最长上升子序列
        n = ac.read_int()
        nums = ac.read_list_ints()
        ind = [0] * (n + 1)
        for i, num in enumerate(nums):
            ind[num] = i
        nums = [ind[x] for x in ac.read_list_ints()]
        ac.st(LongestIncreasingSubsequence().definitely_increase(nums))
        return

    @staticmethod
    def lg_p5939(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5939
        tag: lis
        """
        # 旋转后转换为 LIS 问题
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums = [[x + y, y - x] for x, y in nums]
        nums.sort(key=lambda it: [it[0], -it[1]])
        dp = []
        for _, y in nums:
            i = bisect.bisect_left(dp, y)
            if 0 <= i < len(dp):
                dp[i] = y
            else:
                dp.append(y)
        ac.st(len(dp))
        return

    @staticmethod
    def lg_p5978(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5978
        tag: lis|greed|brute_force
        """
        #  LIS 变形问题，greedybrute_force前半部分
        n, x = ac.read_list_ints()
        nums = ac.read_list_ints()
        # preprocess后缀部分的最长 LIS 序列
        post = [0] * (n + 1)
        dp = []
        for i in range(n - 1, -1, -1):
            j = bisect.bisect_left(dp, -nums[i])
            post[i] = j + 1
            if 0 <= j < len(dp):
                dp[j] = -nums[i]
            else:
                dp.append(-nums[i])

        # greedy减少前缀值并维护最长子序列
        ans = max(post)
        dp = []
        for i in range(n):
            j = bisect.bisect_left(dp, nums[i])
            ans = max(ans, j + post[i])
            j = bisect.bisect_left(dp, nums[i] - x)
            if 0 <= j < len(dp):
                dp[j] = nums[i] - x
            else:
                dp.append(nums[i] - x)
        ac.st(ans)
        return

    @staticmethod
    def lg_p7957(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P7957
        tag: lis|lds|construction
        """
        #  LMS 逆问题construction
        n, k = ac.read_list_ints()
        if k * k < n:
            ac.st(-1)
            return
        ans = []
        x = 1
        while len(ans) < n:
            rest = min(n - len(ans), k)
            for y in range(x + rest - 1, x - 1, -1):
                ans.append(y)
            x = x + rest
        ac.lst(ans)
        return

    @staticmethod
    def cf_1682c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1682/problem/C
        tag: lis|lds|greed|counter
        """
        for _ in range(ac.read_int()):
            ac.read_int()
            nums = ac.read_list_ints()
            cnt = Counter([num ^ ac.random_seed for num in nums])
            s = t = 0
            for va in cnt.values():
                if va >= 2:
                    s += 1
                else:
                    t += 1
            ac.st(s + (t + 1) // 2)
        return

    @staticmethod
    def lc_673(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/number-of-longest-increasing-subsequence/
        tag: lis|counter|O(nlogn)|classical
        """
        return LcsComputeByLis().length_and_cnt_of_lis(nums)

    @staticmethod
    def lc_1092(str1: str, str2: str) -> str:
        """
        url: https://leetcode.cn/problems/shortest-common-supersequence/
        tag: lcs_by_lis|super_sequence
        """
        # 利用LIS求LCS的最短公共超序列
        if len(str1) > len(str2):
            str1, str2 = str2, str1
        lcs_lis = LcsComputeByLis().index_of_lcs(str1, str2)
        i = j = 0
        ans = ""
        for ind in lcs_lis:
            w = str2[ind]
            while str1[i] != w:
                ans += str1[i]
                i += 1
            while str2[j] != w:
                ans += str2[j]
                j += 1
            ans += w
            i += 1
            j += 1
        ans += str1[i:] + str2[j:]
        return ans

    @staticmethod
    def lg_p1410(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1410
        tag: dilworth|lis
        """
        # 最长不上升子序列
        while True:
            lst = ac.read_list_ints()
            if not lst:
                break
            lst = lst[1:]
            dp = []
            for num in lst:
                i = bisect.bisect_right(dp, -num)
                if i < len(dp):
                    dp[i] = -num
                else:
                    dp.append(-num)
            ac.st("Yes!" if len(dp) <= 2 else "No!")
        return

    @staticmethod
    def ac_3549(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/3552/
        tag: liner_dp|greed
        """
        # 翻转连续子数组获得最长不降子序列
        ac.read_int()
        nums = ac.read_list_ints()
        s1 = s12 = s121 = s1212 = 0
        for num in nums:
            if num == 1:
                s1 += 1
                s121 = max(s12 + 1, s121 + 1)
            else:
                s12 = max(s1 + 1, s12 + 1)
                s1212 = max(s121 + 1, s1212 + 1)
        ac.st(max(s1212, s1, s12, s121))
        return

    @staticmethod
    def ac_3662_1(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/3665/
        tag: lis|counter|discretization|tree_array|liner_dp|segment_tree
        """
        # 所有长度的严格上升子序列的最大子序列和，discretizationtree_array|与liner_dp，也可segment_tree|
        ac.read_int()
        nums = ac.read_list_ints()
        ind = {num: i for i, num in enumerate(sorted(list(set(nums))))}
        n = len(ind)
        tree = PointAscendPreMax(n)

        for num in nums:
            if ind[num] == 0:
                tree.point_ascend(1, num)
            else:
                tree.point_ascend(ind[num] + 1, tree.pre_max(ind[num]) + num)
        ac.st(tree.pre_max(n))
        return

    @staticmethod
    def ac_3662_2(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/3665/
        tag: lis|counter|discretization|tree_array|liner_dp|segment_tree
        """
        # 所有长度的严格上升子序列的最大子序列和，discretizationtree_array|与liner_dp，也可segment_tree|
        ac.read_int()
        nums = ac.read_list_ints()
        ind = {num: i for i, num in enumerate(sorted(list(set(nums))))}
        n = len(ind)
        tree = RangeAscendRangeMax(n)
        for num in nums:
            if ind[num] == 0:
                tree.range_ascend(0, 0, num)
            else:
                pre = tree.range_max(0, ind[num] - 1)
                tree.range_ascend(ind[num], ind[num], pre + num)
        ac.st(tree.range_max(0, n - 1))
        return

    @staticmethod
    def ac_2694(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/2696/
        tag: lcs_by_lis|counter|dp
        """
        # LIS的方法求解LCS的长度与个数
        mod = 10 ** 8
        s1 = ac.read_str()[:-1]
        s2 = ac.read_str()[:-1]
        length, cnt = LcsComputeByLis().length_and_cnt_of_lcs(s1, s2, mod)
        ac.st(length)
        ac.st(cnt % mod)
        return

    @staticmethod
    def lg_p2516(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2516
        tag: lcs|matrix_dp|length_of_lcs|cnt_of_lcs|rolling_array
        """
        mod = 10 ** 8
        ans = LcsComputeByLis().length_and_cnt_of_lcs(ac.read_str()[:-1], ac.read_str()[:-1], mod)
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def main(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc354/tasks/abc354_f
        tag: lis|classical
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            pre = [0] * n
            dp = []
            for x, num in enumerate(nums):
                i = bisect.bisect_left(dp, num)
                pre[x] = i
                if 0 <= i < len(dp):
                    dp[i] = num
                else:
                    dp.append(num)
            ceil = len(dp)
            post = [0] * n
            dp = []
            for x in range(n - 1, -1, -1):
                num = -nums[x]
                i = bisect.bisect_left(dp, num)
                post[x] = i
                if 0 <= i < len(dp):
                    dp[i] = num
                else:
                    dp.append(num)
            res = []
            for i in range(n):
                if pre[i] + post[i] + 1 == ceil:
                    res.append(i + 1)
            ac.st(len(res))
            ac.lst(res)
        return

    @staticmethod
    def abc_354f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc354/tasks/abc354_f
        tag: lis|classical
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            pre = [0] * n
            dp = []
            for x, num in enumerate(nums):
                i = bisect.bisect_left(dp, num)
                pre[x] = i
                if 0 <= i < len(dp):
                    dp[i] = num
                else:
                    dp.append(num)
            ceil = len(dp)
            post = [0] * n
            dp = []
            for x in range(n - 1, -1, -1):
                num = -nums[x]
                i = bisect.bisect_left(dp, num)
                post[x] = i
                if 0 <= i < len(dp):
                    dp[i] = num
                else:
                    dp.append(num)
            res = []
            for i in range(n):
                if pre[i] + post[i] + 1 == ceil:
                    res.append(i + 1)
            ac.st(len(res))
            ac.lst(res)
        return

    @staticmethod
    def cf_486e_1(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/486/E
        tag: lis|greed|brain_teaser|classical
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        pre = [0] * n
        dp = []
        for x, num in enumerate(nums):
            i = bisect.bisect_left(dp, num)
            pre[x] = i
            if 0 <= i < len(dp):
                dp[i] = num
            else:
                dp.append(num)
        ceil = len(dp)
        post = [0] * n
        dp = []
        for x in range(n - 1, -1, -1):
            num = -nums[x]
            i = bisect.bisect_left(dp, num)
            post[x] = i
            if 0 <= i < len(dp):
                dp[i] = num
            else:
                dp.append(num)

        cnt = [0] * n
        for i in range(n):
            if pre[i] + post[i] + 1 == ceil:
                cnt[pre[i]] += 1

        ans = ["1"] * n
        for i in range(n):
            if pre[i] + post[i] + 1 != ceil:
                continue
            if cnt[pre[i]] == 1:
                ans[i] = "3"
            else:
                ans[i] = "2"
        ac.st("".join(ans))
        return
    
    @staticmethod
    def cf_486e_2(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/486/E
        tag: lis|greed|brain_teaser|classical
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        mod = random.getrandbits(64)
        dp = []  # LIS array
        s = []  # index is length and value is sum
        q = []  # index is length and value is [num, cnt]
        pre = [0] * n
        cnt = [0] * n
        for x, num in enumerate(nums):
            if not dp or num > dp[-1]:
                dp.append(num)
                length = len(dp)
            else:
                i = bisect.bisect_left(dp, num)
                dp[i] = num
                length = i + 1
            while len(s) <= len(dp):
                s.append(0)
            while len(q) <= len(dp):
                q.append(deque())

            if length == 1:
                s[length] += 1
                q[length].append([num, 1])
            else:
                while q[length - 1] and q[length - 1][0][0] >= num:
                    s[length - 1] -= q[length - 1].popleft()[1]
                s[length] += s[length - 1]
                s[length] %= mod
                q[length].append([num, s[length - 1]])
            pre[x] = length - 1
            cnt[x] = s[length - 1] if length - 1 else 1

        dp = []  # LIS array
        s = []  # index is length and value is sum
        q = []  # index is length and value is [num, cnt]
        post = [0] * n
        cnt_post = [0] * n
        for x in range(n - 1, -1, -1):
            num = -nums[x]
            if not dp or num > dp[-1]:
                dp.append(num)
                length = len(dp)
            else:
                i = bisect.bisect_left(dp, num)
                dp[i] = num
                length = i + 1
            while len(s) <= len(dp):
                s.append(0)
            while len(q) <= len(dp):
                q.append(deque())

            if length == 1:
                s[length] += 1
                q[length].append([num, 1])
            else:
                while q[length - 1] and q[length - 1][0][0] >= num:
                    s[length - 1] -= q[length - 1].popleft()[1]
                s[length] += s[length - 1]
                s[length] %= mod
                q[length].append([num, s[length - 1]])
            post[x] = length - 1
            cnt_post[x] = s[length - 1] if length - 1 else 1

        ceil = len(dp)
        tot = s[-1] % mod
        ans = ["1"] * n
        for i in range(n):
            if pre[i] + post[i] + 1 != ceil:
                continue
            if (cnt[i] * cnt_post[i]) % mod == tot:
                ans[i] = "3"
            else:
                ans[i] = "2"
        ac.st("".join(ans))
        return

    @staticmethod
    def cf_650d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/650/D
        tag: lis|brain_teaser|classical|offline_query
        """
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        queries = [1] * m
        pos = [[] for _ in range(n)]
        for x in range(m):
            i, j = ac.read_list_ints()
            pos[i - 1].append((j, x))

        pre = [0] * n
        dp = []
        for x, num in enumerate(nums):
            i = bisect.bisect_left(dp, num)
            for a, j in pos[x]:
                queries[j] += bisect.bisect_left(dp, a)
            pre[x] = i
            if 0 <= i < len(dp):
                dp[i] = num
            else:
                dp.append(num)

        ceil = len(dp)
        post = [0] * n
        dp = []
        for x in range(n - 1, -1, -1):
            num = -nums[x]
            i = bisect.bisect_left(dp, num)
            for a, j in pos[x]:
                queries[j] += bisect.bisect_left(dp, -a)
            post[x] = i
            if 0 <= i < len(dp):
                dp[i] = num
            else:
                dp.append(num)

        cnt = [0] * n
        for i in range(n):
            if pre[i] + post[i] + 1 == ceil:
                cnt[pre[i]] += 1
        for i in range(n):
            for a, j in pos[i]:
                if cnt[pre[i]] > 1 or pre[i] + post[i] + 1 < ceil:
                    queries[j] = max(queries[j], ceil)
                else:
                    queries[j] = max(queries[j], ceil - 1)
        for q in queries:
            ac.st(q)
        return

    @staticmethod
    def abc_360g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc360/tasks/abc360_g
        tag: lis|greed|implemention|classical|linear_dp|prefix_suffix
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        post = [(0, 0) for _ in range(n + 1)]
        post[n] = (0, math.inf)
        dp = []
        for x in range(n - 1, -1, -1):
            num = -nums[x]
            i = bisect.bisect_left(dp, num)
            post[x] = post[x + 1]
            if 0 <= i < len(dp):
                dp[i] = num
                cur = i + 1
            else:
                dp.append(num)
                cur = len(dp)
            if (cur, -num) > post[x]:
                post[x] = (cur, -num)

        ans = post[0][0]
        pre = (0, math.inf)
        dp = []
        for x, num in enumerate(nums):
            i = bisect.bisect_left(dp, num)
            if x + 1 < n and post[x + 1][1] > -pre[1] + 1:
                ans = max(ans, post[x + 1][0] + pre[0] + 1)
            if 0 <= i < len(dp):
                dp[i] = num
                cur = i + 1
            else:
                dp.append(num)
                cur = len(dp)
            if (cur, -num) > pre:
                pre = (cur, -num)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1108(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1108
        tag: matrix_dp|lis|classical|brain_teaser
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        length = [1] * n
        dedup = [1] * n
        res2 = 0
        for i in range(n):
            for j in range(i):
                if nums[j] > nums[i]:
                    if length[j] + 1 > length[i]:
                        length[i] = length[j] + 1
                        dedup[i] = dedup[j]
                    elif length[j] + 1 == length[i]:
                        dedup[i] += dedup[j]
                elif nums[j] == nums[i]:
                    dedup[j] = length[j] = 0
        res1 = max(length)
        for i in range(n):
            if length[i] == res1:
                res2 += dedup[i]
        ac.lst([res1, res2])
        return

    @staticmethod
    def lc_3288(coordinates: List[List[int]], k: int) -> int:
        """
        url: https://leetcode.cn/problems/length-of-the-longest-increasing-path/
        tag: lis|partial_order|classical
        """
        kx, ky = coordinates[k]
        coordinates.sort(key=lambda it: (it[0], -it[1]))
        dp = []
        for x, y in coordinates:
            if (x < kx and y < ky) or (x > kx and y > ky):
                i = bisect.bisect_left(dp, y)
                if 0 <= i < len(dp):
                    dp[i] = y
                else:
                    dp.append(y)
        return len(dp) + 1
