"""
Algorithm：heapq|monotonic_queue|huffman_tree
Description：greed

====================================LeetCode====================================
630（https://leetcode.cn/problems/course-schedule-iii/）delay_heapq|greed
2454（https://leetcode.cn/problems/next-greater-element-iv/）heapq|post_second_larger|hash|SortedList
2402（https://leetcode.cn/problems/meeting-rooms-iii/）heapq|implemention|counter
2386（https://leetcode.cn/problems/find-the-k-sum-of-an-array/）heapq|brain_teaser
2163（https://leetcode.cn/problems/minimum-difference-in-sums-after-removal-of-elements/）prefix_suffix|brute_force
1792（https://leetcode.cn/problems/maximum-average-pass-ratio/）greed
295（https://leetcode.cn/problems/find-median-from-data-stream/）heapq|median
2542（https://leetcode.cn/problems/maximum-subsequence-score/）greed|sort|brute_force|heapq
2263（https://leetcode.cn/problems/make-array-non-decreasing-or-non-increasing/）heapq|greed

=====================================LuoGu======================================
P1168（https://www.luogu.com.cn/problem/P1168）heapq|median
P1801（https://www.luogu.com.cn/problem/P1801）heapq
P2085（https://www.luogu.com.cn/problem/P2085）math|heapq
P1631（https://www.luogu.com.cn/problem/P1631）heapq|pointer
P4053（https://www.luogu.com.cn/problem/P4053）delay_heapq|greed
P1878（https://www.luogu.com.cn/problem/P1878）hash|heapq|implemention
P3620（https://www.luogu.com.cn/problem/P3620）greed|heapq|double_linked_list
P2168（https://www.luogu.com.cn/problem/P2168）huffman_tree|heapq|greed
P2278（https://www.luogu.com.cn/problem/P2278）heapq|implemention
P1717（https://www.luogu.com.cn/problem/P1717）brute_force|heapq|greed
P1905（https://www.luogu.com.cn/problem/P1905）heapq|greed
P2409（https://www.luogu.com.cn/problem/P2409）heapq
P2949（https://www.luogu.com.cn/problem/P2949）heapq|greed|implemention|delay_heapq|lazy_heapq
P6033（https://www.luogu.com.cn/problem/P6033）greed|deque
P4597（https://www.luogu.com.cn/problem/P4597）heapq|greed

=====================================CodeForces=====================================
1837F（https://codeforces.com/problemset/problem/1837/F）binary_search|priority_queue|implemention|greed|nlognlogn

=====================================AtCoder=====================================
ABC325D（https://atcoder.jp/contests/abc325/tasks/abc325_d）heapq|greed|implemention|classical
ABC250G（https://atcoder.jp/contests/abc250/tasks/abc250_g）regret_heapq|greed|brain_teaser|classical
ABC212D（https://atcoder.jp/contests/abc212/tasks/abc212_d）implemention|inclusion_exclusion|stack

=====================================AcWing=====================================
146（https://www.acwing.com/problem/content/description/148/）heapq
147（https://www.acwing.com/problem/content/description/149/）greed|heapq|double_linked_list
148（https://www.acwing.com/problem/content/150/）greed|heapq|huffman_tree
149（https://www.acwing.com/problem/content/description/151/）huffman_tree|heapq|greed

=====================================CodeChef=====================================
1（https://www.codechef.com/problems/OKLAMA）priority_queue|greed|implemention|math|brain_teaser|induction_method

"""

import heapq
import math
from collections import deque, defaultdict
from heapq import heappushpop, heappush, heappop, heapify
from typing import List

from src.structure.priority_queue.template import MedianFinder, HeapqMedian
from src.structure.sorted_list.template import SortedList
from src.structure.tree_array.template import PointAddRangeSum
from src.util.fast_io import FastIO



class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_2454_1(nums: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/next-greater-element-iv/
        tag: heapq|post_second_larger|hash|SortedList|classical|bucket
        """
        n = len(nums)
        dct = defaultdict(list)
        for i in range(n):
            dct[nums[i]].append(i)
        lst = SortedList()
        ans = [-1] * n
        for num in sorted(dct, reverse=True):
            for i in dct[num]:
                j = lst.bisect_left(i)
                if 0 <= j + 1 < len(lst):
                    ans[i] = nums[lst[j + 1]]
            for i in dct[num]:
                lst.add(i)
        return ans

    @staticmethod
    def lc_2454_2(nums: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/next-greater-element-iv/
        tag: heapq|post_second_larger|hash|SortedList
        """
        n = len(nums)
        ans = [-1] * n
        mono_stack = []
        small_stack = []
        for i in range(n):
            while small_stack and small_stack[0][0] < nums[i]:
                ans[heapq.heappop(small_stack)[1]] = nums[i]

            while mono_stack and nums[mono_stack[-1]] < nums[i]:
                j = mono_stack.pop()
                heapq.heappush(small_stack, [nums[j], j])
            mono_stack.append(i)

        return ans

    @staticmethod
    def lg_1168(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1168
        tag: heapq|median
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        arr = MedianFinder()
        for i in range(n):
            arr.add_num(nums[i])
            if i % 2 == 0:
                ac.st(arr.find_median())
        return

    @staticmethod
    def lc_1792(classes, extra_students):
        """
        url: https://leetcode.cn/problems/maximum-average-pass-ratio/
        tag: greed|math|classical
        """
        stack = []
        for p, t in classes:
            heapq.heappush(stack, [p / t - (p + 1) / (t + 1), p, t])
        for _ in range(extra_students):
            r, p, t = heapq.heappop(stack)
            p += 1
            t += 1
            heapq.heappush(stack, [p / t - (p + 1) / (t + 1), p, t])
        return sum(p / t for _, p, t in stack) / len(classes)

    @staticmethod
    def lc_630(courses: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/course-schedule-iii/
        tag: delay_heapq|greed|regret_heapq|classical
        """
        courses.sort(key=lambda x: x[1])
        stack = []
        day = 0
        for duration, last in courses:
            if day + duration <= last:
                day += duration
                heapq.heappush(stack, -duration)
            else:
                if stack and -stack[0] > duration:
                    day += heapq.heappop(stack) + duration
                    heapq.heappush(stack, -duration)
        return len(stack)

    @staticmethod
    def ac_146(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/148/
        tag: heapq|classical|dp|greed|pointer
        """
        for _ in range(ac.read_int()):
            m, n = ac.read_list_ints()
            pre = sorted(ac.read_list_ints())
            for _ in range(m - 1):
                cur = ac.read_list_ints()
                cur.sort()
                stack = [(pre[0] + cur[j], 0, j) for j in range(n)]
                heapify(stack)
                nex = []
                for _ in range(n):
                    val, i, j = heappop(stack)
                    nex.append(val)
                    if i + 1 < n:
                        heappush(stack, (pre[i + 1] + cur[j], i + 1, j))
                pre = nex[:]
            ac.lst(pre)
        return

    @staticmethod
    def ac_147(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/149/
        tag: greed|heapq|double_linked_list|classical|hard
        """
        n, k = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(n)]

        diff = [math.inf] + [nums[i + 1] - nums[i] for i in range(n - 1)] + [math.inf]
        stack = [[diff[i], i] for i in range(1, n)]
        heapq.heapify(stack)
        pre = [i - 1 for i in range(n + 1)]
        post = [i + 1 for i in range(n + 1)]
        pre[0] = 0
        post[n] = n

        ans = 0
        delete = [0] * (n + 1)
        while k:
            val, i = heapq.heappop(stack)
            if delete[i]:
                continue
            ans += diff[i]

            left = diff[pre[i]]
            right = diff[post[i]]
            new = left + right - diff[i]
            diff[i] = new
            delete[pre[i]] = 1
            delete[post[i]] = 1

            pre[i] = pre[pre[i]]
            post[pre[i]] = i

            post[i] = post[post[i]]
            pre[post[i]] = i
            heapq.heappush(stack, [new, i])
            k -= 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p2168(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2168
        tag: huffman_tree|heapq|greed
        """
        n, k = ac.read_list_ints()
        stack = [[ac.read_int(), 0] for _ in range(n)]
        heapq.heapify(stack)
        while (len(stack) - 1) % (k - 1) != 0:
            heapq.heappush(stack, [0, 0])
        ans = 0
        while len(stack) > 1:
            cur = 0
            dep = 0
            for _ in range(k):
                val, d = heapq.heappop(stack)
                cur += val
                dep = max(dep, d)
            ans += cur
            heapq.heappush(stack, [cur, dep + 1])
        ac.st(ans)
        ac.st(stack[0][1])
        return

    @staticmethod
    def lg_p1631(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1631
        tag: heapq|pointer|classical
        """
        n = ac.read_int()
        nums1 = ac.read_list_ints()
        nums2 = ac.read_list_ints()
        stack = [(nums1[0] + nums2[j], 0, j) for j in range(n)]
        heapq.heapify(stack)
        ans = []
        for _ in range(n):
            val, i, j = heapq.heappop(stack)
            ans.append(val)
            if i + 1 < n:
                heapq.heappush(stack, (nums1[i + 1] + nums2[j], i + 1, j))
        ac.lst(ans)
        return

    @staticmethod
    def lg_p4053(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4053
        tag: delay_heapq|greed|regret_heapq|LC630
        """
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda it: it[1])
        pre = 0
        stack = []
        for a, b in nums:
            if pre + a <= b:
                heapq.heappush(stack, -a)
                pre += a
            else:
                if stack and -a > stack[0]:
                    pre += heapq.heappop(stack)
                    pre += a
                    heapq.heappush(stack, -a)
        ac.st(len(stack))
        return

    @staticmethod
    def lg_p2085(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2085
        tag: math|heapq
        """
        n, m = ac.read_list_ints()
        stack = []
        for _ in range(n):
            a, b, c = ac.read_list_ints()
            stack.append((a + b + c, 1, a, b, c))
        heapify(stack)
        ans = []
        while len(ans) < m:
            val, x, a, b, c = heapq.heappop(stack)
            ans.append(val)
            x += 1
            heapq.heappush(stack, (a * x * x + b * x + c, x, a, b, c))
        ac.lst(ans)
        return

    @staticmethod
    def lg_p2278(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2278
        tag: heapq|implemention
        """
        now = []
        ans = []
        stack = []
        pre = 0
        while True:
            lst = ac.read_list_ints()
            if not lst:
                break

            while now and now[-1] <= lst[1]:
                ans.append([now[0], now[-1]])
                pre = now[-1]
                if stack:
                    level, reach, need, idx = heapq.heappop(stack)
                    now = [idx, reach, need, -level, max(pre, reach) + need]
                else:
                    now = []

            if not now and stack:
                level, reach, need, idx = heapq.heappop(stack)
                now = [idx, reach, need, -level, max(pre, reach) + need]

            if now and now[3] >= lst[-1]:
                idx, reach, need, level = lst
                heapq.heappush(stack, (-level, reach, need, idx))
            elif now:
                idx, reach, need, level, end = now
                heapq.heappush(stack, (-level, reach, end - lst[1], idx))
                idx, reach, need, level = lst
                now = [idx, reach, need, level, max(pre, reach) + need]
            else:
                idx, reach, need, level = lst
                now = [idx, reach, need, level, max(pre, reach) + need]

        while stack:
            ans.append([now[0], now[-1]])
            pre = now[-1]
            level, reach, need, idx = heapq.heappop(stack)
            now = [idx, reach, need, -level, max(pre, reach) + need]
        ans.append([now[0], now[-1]])
        for a in ans:
            ac.lst(a)
        return

    @staticmethod
    def lg_p1717(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1717
        tag: brute_force|heapq|greed
        """
        ans = 0
        n = ac.read_int()
        h = ac.read_int() * 60
        f = ac.read_list_ints()
        d = ac.read_list_ints()
        t = [0] + ac.read_list_ints()
        for i in range(n):
            tm = sum(t[:i + 1]) * 5
            stack = [[-f[j], j] for j in range(i + 1)]
            heapq.heapify(stack)
            cur = 0
            while tm + 5 <= h and stack:
                val, j = heapq.heappop(stack)
                val = -val
                cur += val
                tm += 5
                if val - d[j] > 0:
                    heapq.heappush(stack, [-val + d[j], j])
            ans = max(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1905(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1905
        tag: heapq|greed
        """
        ac.read_int()
        p = ac.read_int()
        lst = ac.read_list_ints()
        ans = [[0] for _ in range(p)]
        stack = [(ans[i][0], i) for i in range(p)]
        lst.sort(reverse=True)
        for num in lst:
            d, i = heapq.heappop(stack)
            ans[i][0] += num
            ans[i].append(num)
            heapq.heappush(stack, (ans[i][0], i))
        for a in ans:
            ac.lst(a[1:])
        return

    @staticmethod
    def lg_p2409(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2409
        tag: heapq
        """
        n, k = ac.read_list_ints()
        pre = sorted(ac.read_list_ints()[1:])[:k]
        for _ in range(n - 1):
            cur = sorted(ac.read_list_ints()[1:])[:k]
            m = len(cur)
            stack = [(pre[0] + cur[j], 0, j) for j in range(m)]
            heapify(stack)
            nex = []
            while len(nex) < k and stack:
                val, i, j = heappop(stack)
                nex.append(val)
                if i + 1 < len(pre):
                    heappush(stack, (pre[i + 1] + cur[j], i + 1, j))
            pre = nex[:]
        ac.lst(pre)
        return

    @staticmethod
    def lg_p2949(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2949
        tag: heapq|greed|implemention|delay_heapq|lazy_heapq|regret_heapq
        """
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda it: it[0])
        ans = 0
        stack = []
        for d, p in nums:
            heapq.heappush(stack, p)
            ans += p
            if len(stack) > d:
                ans -= heapq.heappop(stack)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6033(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6033
        tag: greed|priority_queue|classical
        """
        ac.read_int()
        pre = deque(sorted(ac.read_list_ints()))
        post = deque()
        ans = 0
        while len(pre) + len(post) > 1:
            if not pre:
                cur = post.popleft() + post.popleft()
                ans += cur
                post.append(cur)
                continue
            if not post:
                cur = pre.popleft() + pre.popleft()
                ans += cur
                post.append(cur)
                continue
            if pre[0] < post[0]:
                a = pre.popleft()
            else:
                a = post.popleft()
            if pre and (not post or pre[0] < post[0]):
                b = pre.popleft()
            else:
                b = post.popleft()
            ans += a + b
            post.append(a + b)
        ac.st(ans)
        return

    @staticmethod
    def lc_2263(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/make-array-non-decreasing-or-non-increasing/
        tag: heapq|greed
        """

        def helper(lst: List[int]) -> int:
            res, pq = 0, []
            for num in lst:
                if not pq:
                    heappush(pq, -num)
                else:
                    pre = -pq[0]
                    if pre > num:
                        res += pre - num
                        heappushpop(pq, -num)
                    heappush(pq, -num)
            return res

        return min(helper(nums), helper(nums[::-1]))

    @staticmethod
    def lc_2386(nums: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/find-the-k-sum-of-an-array/
        tag: heapq|brain_teaser|dijkstra|classical|hard
        """
        n = len(nums)
        tot = 0
        for i in range(n):
            if nums[i] >= 0:
                tot += nums[i]
            else:
                nums[i] = -nums[i]
        nums.sort()

        stack = [(-tot, 0)]
        for _ in range(k - 1):
            pre, i = heappop(stack)
            if i < n:
                heapq.heappush(stack, (pre + nums[i], i + 1))
                if i:
                    heapq.heappush(stack, (pre + nums[i] - nums[i - 1], i + 1))
        return -stack[0][0]

    @staticmethod
    def lc_24_1(nums: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/5TxKeK/description/
        tag: heapq_median|brain_teaser|classical|median_greedy
        """
        mod = 1000000007
        n = len(nums)
        nums = [nums[i] - i for i in range(n)]
        median = HeapqMedian(nums[0])
        ans = [0]
        lst = [nums[0]]
        for num in nums[1:]:
            median.add(num)
            lst.append(num)
            cur = (median.mid * len(median.left) - median.left_sum)
            cur += (median.right_sum - median.mid * len(median.right))
            ans.append(cur % mod)
        return ans

    @staticmethod
    def lc_24_2(nums: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/5TxKeK/description/
        tag: heapq_median|brain_teaser|classical|median_greedy|tree_array|sorted_list
        """
        mod = 1000000007
        n = len(nums)
        nums = [nums[i] - i + n for i in range(n)]
        lst = SortedList()
        ans = []
        ceil = max(nums)
        tree_cnt = PointAddRangeSum(ceil)
        tree_sum = PointAddRangeSum(ceil)
        for num in nums:
            lst.add(num)
            tree_cnt.point_add(num, 1)
            tree_sum.point_add(num, num)
            x = lst[len(lst) // 2]
            cur = 0
            if x - 1 >= 1:
                cur += tree_cnt.range_sum(1, x - 1) * x - tree_sum.range_sum(1, x - 1)
            if x + 1 <= ceil:
                cur += tree_sum.range_sum(x + 1, ceil) - tree_cnt.range_sum(x + 1, ceil) * x
            ans.append(cur % mod)
        return ans

    @staticmethod
    def abc_250g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc250/tasks/abc250_g
        tag: regret_heapq|greed|brain_teaser|classical
        """
        ac.read_int()
        nums = ac.read_list_ints()
        ans = 0
        stack = []
        for num in nums:
            if stack and num > stack[0]:
                ans += num - heappop(stack)
                heappush(stack, num)
            heappush(stack, num)
        ac.st(ans)
        return

    @staticmethod
    def cc_1(ac=FastIO()):
        """
        url: https://www.codechef.com/problems/OKLAMA
        tag: priority_queue|greed|implemention|math|brain_teaser|induction_method
        """
        for _ in range(ac.read_int()):
            n, k = ac.read_list_ints()
            nums = ac.read_list_ints()
            ans = [0] * n
            big = []
            small = []
            ans[0] = nums[0]
            big_sum = 0
            small_sum = 0
            for i in range(n):
                if len(big) <= len(small):
                    heappush(big, nums[i])
                    big_sum += nums[i]
                else:
                    heappush(small, -nums[i])
                    small_sum += nums[i]
                if (i + 1) % 2 and i >= 2:
                    pre = (i + 1) // 2

                    while small and big and -small[0] > big[0]:
                        x = -heappop(small)
                        small_sum -= x
                        big_sum += x
                        heappush(big, x)

                        x = heappop(big)
                        big_sum -= x
                        small_sum += x
                        heappush(small, -x)

                    while len(small) > pre:
                        x = -heappop(small)
                        small_sum -= x
                        big_sum += x
                        heappush(big, x)
                    while len(small) < pre:
                        x = heappop(big)
                        big_sum -= x
                        small_sum += x
                        heappush(small, -x)

                    ans[i] = big_sum - small_sum
            res = [ans[q] for q in ac.read_list_ints_minus_one()]
            ac.lst(res)
        return

    @staticmethod
    def abc_212d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc212/tasks/abc212_d
        tag: implemention|inclusion_exclusion|stack
        """
        stack = []
        cur = 0
        q = ac.read_int()
        val = [0] * (q + 1)
        for i in range(q):
            tmp = ac.read_list_ints()
            if tmp[0] == 1:
                x = tmp[1]
                heapq.heappush(stack, x - cur)
            elif tmp[0] == 2:
                cur += tmp[1]
            else:
                x = heapq.heappop(stack)
                ac.st(x + cur)
            val[i] = cur
        return

    @staticmethod
    def cf_1837e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1837/F
        tag: binary_search|priority_queue|implemention|greed|nlognlogn
        """
        for _ in range(ac.read_int()):  # TLE
            n, k = ac.read_list_ints()
            nums = ac.read_list_ints()

            def check(x):
                stack = []
                s = 0
                for i, num in enumerate(nums):
                    if s + num <= x:
                        heappush(stack, -num)
                        s += num
                    elif stack and -stack[0] > num:
                        s += stack[0]
                        heappushpop(stack, -num)
                        s += num
                    pre[i + 1] = len(stack)
                    if pre[i + 1] >= k:
                        return True

                s = 0
                stack = []
                for i in range(n - 1, -1, -1):
                    num = nums[i]
                    if s + num <= x:
                        heappush(stack, -num)
                        s += num
                    elif stack and -stack[0] > num:
                        s += stack[0]
                        heappushpop(stack, -num)
                        s += num
                    if pre[i] + len(stack) >= k:
                        return True
                return False

            pre = [0] * (n + 1)

            low = 0
            high = sum(nums)
            while low < high:
                mid = low + (high - low) // 2
                if check(mid):
                    high = mid
                else:
                    low = mid + 1
            ac.st(low)
        return