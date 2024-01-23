"""

Algorithm：sorted_list
Description：range_query|binary_search


====================================LeetCode====================================
295（https://leetcode.cn/problems/find-median-from-data-stream/）sorted_list
2426（https://leetcode.cn/problems/number-of-pairs-satisfying-inequality/）math|sorted_list|binary_search
2179（https://leetcode.cn/problems/count-good-triplets-in-an-array/）sorted_list|binary_search
2141（https://leetcode.cn/problems/maximum-running-time-of-n-computers/）greedy|binary_search|implemention
2102（https://leetcode.cn/problems/sequentially-ordinal-rank-tracker/）sorted_list
2519（https://leetcode.cn/problems/count-the-number-of-k-big-indices/）sorted_list
1912（https://leetcode.cn/problems/design-movie-rental-system/）sorted_list|classical
1825（https://leetcode.cn/problems/finding-mk-average/）sorted_list|deque
2250（https://leetcode.cn/problems/count-number-of-rectangles-containing-each-point/）offline_query|pointer|sort|binary_search
2426（https://leetcode.cn/problems/number-of-pairs-satisfying-inequality/）math|sorted_list|binary_search
2276（https://leetcode.cn/problems/count-integers-in-intervals/）sorted_list|implemention|classical
3013（https://leetcode.com/problems/divide-an-array-into-subarrays-with-minimum-cost-ii/）sorted_list|top_k_sum
1851（https://leetcode.cn/problems/minimum-interval-to-include-each-query）


=====================================LuoGu======================================
P7333（https://www.luogu.com.cn/problem/P7333）sort|sorted_list|circular_array|range_query
P7391（https://www.luogu.com.cn/problem/P7391）sorted_list|greedy|implemention|lazy_heapq
P7910（https://www.luogu.com.cn/problem/P7910）sorted_list
P4375（https://www.luogu.com.cn/problem/P4375）bubble_sort|sorted_list
P1908（https://www.luogu.com.cn/problem/P1908）reverse_order_pair|merge_sort
P1966（https://www.luogu.com.cn/problem/P1966）reverse_order_pair|greedy
P2161（https://www.luogu.com.cn/problem/P2161）range_merge_to_disjoint
P1637（https://www.luogu.com.cn/problem/P1637）sorted_list|prefix_suffix|counter
P2234（https://www.luogu.com.cn/problem/P2234）sorted_list
P2804（https://www.luogu.com.cn/problem/P2804）prefix_sum|sorted_list
P3608（https://www.luogu.com.cn/problem/P3608）sorted_list
P5076（https://www.luogu.com.cn/problem/P5076）sorted_list|implemention
P5149（https://www.luogu.com.cn/problem/P5149）reverse_order_pair|bisect
P5459（https://www.luogu.com.cn/problem/P5459）prefix_sum|sorted_list|binary_search
P6538（https://www.luogu.com.cn/problem/P6538）sorted_list|greedy
P7912（https://www.luogu.com.cn/problem/P7912）sorted_list|implemention
P8667（https://www.luogu.com.cn/problem/P8667）sorted_list

===================================CodeForces===================================
459D（https://codeforces.com/problemset/problem/459/D）sorted_list|counter
61E（https://codeforces.com/problemset/problem/61/E）sorted_list|prefix_suffix|counter
1354D（https://codeforces.com/problemset/problem/1354/D）sorted_list
1005E2（https://codeforces.com/contest/1005/problem/E2）median|inclusion_exclusion|prefix_sum|sorted_list|binary_search
1619E（https://codeforces.com/contest/1619/problem/E）mex|greedy


===================================CodeForces===================================
129（https://www.acwing.com/problem/content/129/）greedy|classical|sorted_list

"""
import bisect
from bisect import insort_left, bisect_left
from typing import List

from src.data_structure.sorted_list.template import SortedList, TopKSum
from src.utils.fast_io import FastIO
from src.utils.fast_io import inf


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lg_4375(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4375
        tag: bubble_sort|sorted_list|hard
        """
        n = ac.read_int()
        ans = 1
        nums = [ac.read_int() for _ in range(n)]
        tmp = sorted(nums)
        ind = {num: i + 1 for i, num in enumerate(tmp)}
        lst = SortedList()
        for i in range(n):
            lst.add(ind[nums[i]])
            ans = ac.max(ans, i + 1 - lst.bisect_right(i + 1))
        ac.st(ans)
        return

    @staticmethod
    def cf_61e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/61/E
        tag: sorted_list|prefix_suffix|counter
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        pre = [0] * (n + 1)
        lst = SortedList()
        for i in range(n):
            pre[i + 1] = i - lst.bisect_right(nums[i])
            lst.add(nums[i])

        post = [0] * (n + 1)
        lst = SortedList()
        for i in range(n - 1, -1, -1):
            post[i] = lst.bisect_left(nums[i])
            lst.add(nums[i])

        ans = 0
        for i in range(1, n - 1):
            ans += pre[i + 1] * post[i]
        ac.st(ans)
        return

    @staticmethod
    def cf_1005e2(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1005/problem/E2
        tag: median|inclusion_exclusion|prefix_sum|sorted_list|binary_search|brain_teaser
        """
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()

        def check(x):
            lst = [1 if num >= x else -1 for num in nums]
            pre = SortedList([0])
            cur = res = 0
            for num in lst:
                cur += num
                res += pre.bisect_left(cur)
                pre.add(cur)
            return res

        ac.st(- check(m + 1) + check(m))
        return

    @staticmethod
    def lc_2426(nums1: List[int], nums2: List[int], diff: int) -> int:
        """
        url: https://leetcode.cn/problems/number-of-pairs-satisfying-inequality/
        tag: math|sorted_list|binary_search
        """
        n = len(nums1)
        ans = 0
        lst = SortedList([nums1[n - 1] - nums2[n - 1] + diff])
        for i in range(n - 2, -1, -1):
            k = lst.bisect_left(nums1[i] - nums2[i])
            ans += len(lst) - k
            lst.add(nums1[i] - nums2[i] + diff)
        return ans

    @staticmethod
    def lg_1966(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1966
        tag: reverse_order_pair|greedy|classical
        """
        n = ac.read_int()
        ans = 0
        mod = 10 ** 8 - 3
        nums1 = ac.read_list_ints()
        ind1 = list(range(n))
        ind1.sort(key=lambda it: nums1[it])

        nums2 = ac.read_list_ints()
        ind2 = list(range(n))
        ind2.sort(key=lambda it: nums2[it])

        q = [0] * n
        for i in range(n):
            q[ind1[i]] = ind2[i]
        lst = SortedList()
        for num in q:
            ans += len(lst) - lst.bisect_right(num)
            lst.add(num)
        ac.st(ans % mod)
        return

    @staticmethod
    def ac_129(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/129/
        tag: greedy|classical|sorted_list
        """
        n, m = ac.read_list_ints()
        machine = [ac.read_list_ints() for _ in range(n)]
        task = [ac.read_list_ints() for _ in range(m)]
        machine.sort(reverse=True)
        task.sort(reverse=True)
        lst = []
        ans = money = j = 0
        for i in range(m):
            tm, level = task[i]
            while j < n and machine[j][0] >= tm:
                insort_left(lst, machine[j][1])
                j += 1
            ind = bisect_left(lst, level)
            if ind < len(lst):
                lst.pop(ind)
                ans += 1
                money += 500 * tm + 2 * level
        ac.lst([ans, money])
        return

    @staticmethod
    def lg_p1637(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1637
        tag: sorted_list|prefix_suffix|counter
        """
        n = ac.read_int()
        nums = ac.read_list_ints()

        pre = [0] * n
        lst = SortedList()
        for i in range(n):
            pre[i] = lst.bisect_left(nums[i])
            lst.add(nums[i])

        lst = SortedList()
        post = [0] * n
        for i in range(n - 1, -1, -1):
            post[i] = n - i - 1 - lst.bisect_right(nums[i])
            lst.add(nums[i])
        ans = sum(pre[i] * post[i] for i in range(n))
        ac.st(ans)
        return

    @staticmethod
    def lg_p2234(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2234
        tag: sorted_list
        """
        n = ac.read_int()
        ans = 0
        lst = SortedList()
        for _ in range(n):
            x = ac.read_int()
            if not lst:
                ans += x
            else:
                i = lst.bisect_left(x)
                cur = inf
                for j in [i - 1, i]:
                    if 0 <= j < len(lst):
                        cur = ac.min(cur, abs(lst[j] - x))
                ans += cur
            lst.add(x)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2804(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2804
        tag: prefix_sum|sorted_list|classical|brain_teaser
        """
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        mod = 92084931
        pre = 0
        lst = SortedList()
        lst.add(0)
        ans = 0
        for num in nums:
            pre += num - m
            ans += lst.bisect_left(pre)
            lst.add(pre)
        ac.st(ans % mod)
        return

    @staticmethod
    def lg_p5076(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5076
        tag: sorted_list|implemention
        """
        q = ac.read_int()
        dct = set()
        lst = []
        lst_set = []
        for _ in range(q):
            op, x = ac.read_list_ints()
            if op == 1:
                i = bisect.bisect_left(lst, x)
                ac.st(i + 1)
            elif op == 2:
                ac.st(lst_set[x - 1])
            elif op == 3:
                i = bisect.bisect_left(lst, x)
                n = len(lst)
                ans = -2147483647
                if 0 <= i - 1 < n and lst_set[i - 1] < x:
                    ans = lst_set[i - 1]
                ac.st(ans)
            elif op == 4:
                i = bisect.bisect_right(lst, x)
                n = len(lst)
                if 0 <= i < n:
                    ac.st(lst[i])
                else:
                    ac.st(2147483647)
            else:
                bisect.insort_left(lst, x)
                if x not in dct:
                    bisect.insort_left(lst_set, x)
                    dct.add(x)
        return

    @staticmethod
    def lg_p5149(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5149
        tag: reverse_order_pair|bisect
        """
        ac.read_int()
        lst = ac.read_list_strs()
        ind = {st: i for i, st in enumerate(lst)}
        lst = [ind[s] for s in ac.read_list_strs()]
        ans = 0
        pre = []
        for num in lst:
            ans += len(pre) - bisect.bisect_left(pre, num)
            bisect.insort_left(pre, num)
        ac.st(ans)
        return

    @staticmethod
    def lg_p5459(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5459
        tag: prefix_sum|sorted_list|binary_search
        """
        n, low, high = ac.read_list_ints()
        a = ac.read_list_ints()
        ans = 0
        lst = []
        s = sum(a)
        bisect.insort_left(lst, s)
        for i in range(n - 1, -1, -1):
            s -= a[i]
            x = bisect.bisect_left(lst, s + low)
            ans += bisect.bisect_right(lst, s + high) - x
            bisect.insort_left(lst, s)
        ac.st(ans)
        return

    @staticmethod
    def lg_p7912(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P7912
        tag: sorted_list|implemention|classical
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        lst = [SortedList([i + 1 for i in range(n) if not nums[i]]),
               SortedList([i + 1 for i in range(n) if nums[i]])]

        while lst[0] and lst[1]:
            ans = []
            if lst[0][0] < lst[1][0]:
                i = 0
            else:
                i = 1
            ans.append(lst[i].pop(0))
            i = 1 - i
            while lst[i] and lst[i][-1] > ans[-1]:
                j = lst[i].bisect_left(ans[-1])
                ans.append(lst[i].pop(j))
                i = 1 - i
            ac.lst(ans)

        for a in lst[0]:
            ac.st(a)
        for a in lst[1]:
            ac.st(a)
        return

    @staticmethod
    def lc_2276():
        """
        url: https://leetcode.cn/problems/count-integers-in-intervals/
        tag: sorted_list|implemention|classical
        """

        class CountIntervals:

            def __init__(self):
                self.lst = SortedList()
                self.sum = 0

            def add(self, left: int, right: int) -> None:
                x = self.lst.bisect_left([left, left])
                while x - 1 >= 0 and self.lst[x - 1][1] >= left:
                    x -= 1

                while 0 <= x < len(self.lst) and not (self.lst[x][0] > right or self.lst[x][1] < left):
                    a, b = self.lst.pop(x)
                    left = left if left < a else a
                    right = right if right > b else b
                    self.sum -= b - a + 1
                self.sum += right - left + 1
                self.lst.add([left, right])

            def count(self) -> int:
                return self.sum

        return CountIntervals

    @staticmethod
    def lc_3013(nums: List[int], k: int, dist: int) -> int:
        """
        url: https://leetcode.com/problems/divide-an-array-into-subarrays-with-minimum-cost-ii/
        tag: sorted_list|top_k_sum
        """
        n = len(nums)
        ans = inf
        lst = TopKSum(k - 2)
        j = 2
        for i in range(1, n):
            if i >= 2:
                lst.remove(nums[i])
            if n - i - 1 < k - 2:
                break
            while j <= i + dist and j < n:
                lst.add(nums[j])
                j += 1
            if len(lst.lst) >= k - 2:
                cur = nums[0] + nums[i] + lst.top_k_sum
                if cur < ans:
                    ans = cur
        return ans
