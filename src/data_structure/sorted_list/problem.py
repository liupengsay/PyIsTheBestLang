"""

Algorithm：sorted_list
Function：确定最优选择，通常可以SortedList用于维护和查询sorted_list信息


====================================LeetCode====================================
295（https://leetcode.com/problems/find-median-from-data-stream/）一个SortedList即可
2426（https://leetcode.com/problems/number-of-pairs-satisfying-inequality/）根据不等式变换和sorted_listbinary_search
2179（https://leetcode.com/problems/count-good-triplets-in-an-array/）维护区间范围内的个数
2141（https://leetcode.com/problems/maximum-running-time-of-n-computers/）greedy选择最大的 N 个电池作为基底，然后binary_search确定在其余电池的|持下可以运行的最长时间
2102（https://leetcode.com/problems/sequentially-ordinal-rank-tracker/）sorted_list维护优先级姓名实时查询
2519（https://leetcode.com/problems/count-the-number-of-k-big-indices/）sorted_list维护数量
1912（https://leetcode.com/problems/design-movie-rental-system/）典型SortedList应用
1825（https://leetcode.com/problems/finding-mk-average/）SortedList与deque应用
2250（https://leetcode.com/problems/count-number-of-rectangles-containing-each-point/）offline_query，pointersortingbinary_search
2426（https://leetcode.com/problems/number-of-pairs-satisfying-inequality/）根据不等式变换和sorted_listbinary_search

=====================================LuoGu======================================
7333（https://www.luogu.com.cn/problem/P7333）sorting预处理后，动态更新sorted_list查询，注意是circular_array|
7391（https://www.luogu.com.cn/problem/P7391）sorted_listgreedyimplemention，延迟替换，类似课程表3
7910（https://www.luogu.com.cn/problem/P7910）sorted_list维护
4375（https://www.luogu.com.cn/problem/P4375）bubble_sort，sorted_list维护
1908（https://www.luogu.com.cn/problem/P1908）问题求reverse_order_pair|，可以merge_sort
1966（https://www.luogu.com.cn/problem/P1966）reverse_order_pair|greedy题目
2161（https://www.luogu.com.cn/problem/P2161）range_merge_to_disjoint与删除处理
1637（https://www.luogu.com.cn/problem/P1637）典型STL应用题，prefix_suffix大小值counter
2234（https://www.luogu.com.cn/problem/P2234）典型STL应用题
2804（https://www.luogu.com.cn/problem/P2804）prefix_sum| STL 平均值大于 m 的连续子数组个数
3608（https://www.luogu.com.cn/problem/P3608）典型STL应用题
5076（https://www.luogu.com.cn/problem/P5076）sorted_list与sorted_list名次implemention
5149（https://www.luogu.com.cn/problem/P5149）reverse_order_pair| bisect 实现
5459（https://www.luogu.com.cn/problem/P5459）prefix_sum与sorted_listbinary_search
6538（https://www.luogu.com.cn/problem/P6538）典型STL维护greedy
7912（https://www.luogu.com.cn/problem/P7912） STL 应用implemention题
8667（https://www.luogu.com.cn/problem/P8667）典型STL应用题

===================================CodeForces===================================
459D（https://codeforces.com/problemset/problem/459/D）sorted_list大小counter查找
61E（https://codeforces.com/problemset/problem/61/E）典型应用场景，prefix_suffix大于小于值counter
1354D（https://codeforces.com/problemset/problem/1354/D）sorted_list的维护与查询
1005E2（https://codeforces.com/contest/1005/problem/E2）特定median的连续子数组个数，inclusion_exclusion|prefix_sumsorted_listbinary_search
1619E（https://codeforces.com/contest/1619/problem/E）MEXgreedy

"""
import bisect
from bisect import insort_left, bisect_left
from math import inf
from typing import List

from sortedcontainers import SortedList

from src.data_structure.sorted_list.template import LocalSortedList
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lg_4375d(ac=FastIO()):
        # 双向bubble_sort所需要的比较轮数
        n = ac.read_int()
        ans = 1
        nums = [ac.read_int() for _ in range(n)]
        tmp = sorted(nums)
        ind = {num: i + 1 for i, num in enumerate(tmp)}
        lst = LocalSortedList()
        for i in range(n):
            lst.add(ind[nums[i]])
            ans = ac.max(ans, i + 1 - lst.bisect_right(i + 1))
        ac.st(ans)
        return

    @staticmethod
    def cf_61e(ac=FastIO()):
        # 典型 i < j < k 但是 nums[i] > nums[j] > nums[k] 的组合数
        n = ac.read_int()
        nums = ac.read_list_ints()
        pre = [0] * (n + 1)
        lst = LocalSortedList()
        for i in range(n):
            pre[i + 1] = i - lst.bisect_right(nums[i])
            lst.add(nums[i])

        post = [0] * (n + 1)
        lst = LocalSortedList()
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
        # 特定median的连续子数组个数，inclusion_exclusion|prefix_sumsorted_listbinary_search
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()

        def check(x):
            lst = [1 if num >= x else -1 for num in nums]
            pre = LocalSortedList([0])
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
        # math|与sorted_listbinary_searchcounter
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
        # reverse_order_pair|greedy题目
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
        lst = LocalSortedList()
        for num in q:
            ans += len(lst) - lst.bisect_right(num)
            lst.add(num)
        ac.st(ans % mod)
        return

    @staticmethod
    def ac_127(ac=FastIO()):
        # 二维sortinggreedy
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
                insort_left(lst, machine[j][1])  # bisect代替Sortedlist
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
        # 典型STL应用题，prefix_suffix大小值counter
        n = ac.read_int()
        nums = ac.read_list_ints()

        pre = [0] * n
        lst = LocalSortedList()
        for i in range(n):
            pre[i] = lst.bisect_left(nums[i])
            lst.add(nums[i])

        lst = LocalSortedList()
        post = [0] * n
        for i in range(n - 1, -1, -1):
            post[i] = n - i - 1 - lst.bisect_right(nums[i])
            lst.add(nums[i])
        ans = sum(pre[i] * post[i] for i in range(n))
        ac.st(ans)
        return

    @staticmethod
    def lg_p2234(ac=FastIO()):
        # 典型STL应用题
        n = ac.read_int()
        ans = 0
        lst = LocalSortedList()
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
        # prefix_sum| STL 平均值大于 m 的连续子数组个数
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        mod = 92084931
        pre = 0
        lst = LocalSortedList()
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
        # sorted_list与sorted_list名次implemention
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
        # reverse_order_pair| bisect 实现
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
        # prefix_sum与sorted_listbinary_search
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
        #  STL 应用implemention题 STL implemention删除
        n = ac.read_int()
        nums = ac.read_list_ints()
        lst = [LocalSortedList([i + 1 for i in range(n) if not nums[i]]),
               LocalSortedList([i + 1 for i in range(n) if nums[i]])]

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


class SolutionLC2276:

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