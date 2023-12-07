"""
Algorithm：ST（Sparse-Table）稀疏表、倍增、数组积性函数聚合性质、连续子数组的聚合运算
Function：静态区间内的最大值、最小值、最大公约数、最大与、最大或
ST表算法全称Sparse-Table算法，是由Tarjan提出的一种解决RMQ问题（区间最值）的强力算法。 离线预处理时间复杂度θ（nlogn），在线查询时间θ（1），可以说是一种非常高效的算法。 不过ST表的应用场合也是有限的，它只能处理静态区间最值，不能维护动态的，也就是说不支持在预处理后对值修改。


====================================LeetCode====================================
1521（https://leetcode.com/problems/find-a-value-of-a-mysterious-function-closest-to-target/）与目标值最接近的连续子数组bit_operation与值
2411（https://leetcode.com/problems/smallest-subarrays-with-maximum-bitwise-or/）最大或值的最短连续子数组
2447（https://leetcode.com/problems/number-of-subarrays-with-gcd-equal-to-k/）最大公因数为 k 的连续子数组个数，可推广到bit_operation或与异或
2470（https://leetcode.com/problems/number-of-subarrays-with-lcm-equal-to-k/）最小公倍为 k 的连续子数组个数，可推广到bit_operation或与异或
2654（https://leetcode.com/problems/minimum-number-of-operations-to-make-all-array-elements-equal-to-1/）最大公因数为 1 的最短连续子数组

=====================================LuoGu======================================
3865（https://www.luogu.com.cn/problem/P3865）ST表静态查询区间最大值
2880（https://www.luogu.com.cn/problem/P2880）ST表预处理区间最大值与最小值
1890（https://www.luogu.com.cn/problem/P3865）ST表预处理区间的gcd
1816（https://www.luogu.com.cn/problem/P1816）ST表预处理区间的最小值
2412（https://www.luogu.com.cn/problem/P2412）预处理lexicographical_order之后ST表查询静态区间最大lexicographical_order
5097（https://www.luogu.com.cn/problem/P5097）静态区间最小值
5648（https://www.luogu.com.cn/problem/P5648）倍增 ST 表查询区间最大值的索引，monotonic_stack|建树距离

===================================CodeForces===================================
1691D（https://codeforces.com/problemset/problem/1691/D）monotonic_stack|brute_force|ST表最大值最小值查询
689D（https://codeforces.com/problemset/problem/689/D）binary_search|ST表个数
1359D（https://codeforces.com/problemset/problem/1359/D）monotonic_stack|brute_force|ST表最大值最小值查询
1548B（https://codeforces.com/problemset/problem/1548/B）ST表查询区间gcd并brute_force数组开头，binary_search确定长度
474F（https://codeforces.com/problemset/problem/474/F）稀疏表最小值和gcd，并binary_searchcounter
1834E（https://codeforces.com/contest/1834/problem/E）连续子数组的lcm信息
1878E（https://codeforces.com/contest/1878/problem/E）连续子数组的and信息
1547F（https://codeforces.com/contest/1547/problem/F）连续子数组的gcd信息
1579F（https://codeforces.com/contest/1579/problem/F）circular_section连续子数组的and信息
1709D（https://codeforces.com/contest/1709/problem/D）use sparse table to compute static range max and implemention
1516D（https://codeforces.com/contest/1516/problem/D）数组区间向右倍增

=====================================AcWing=====================================
109（https://www.acwing.com/problem/content/111/）greedy|倍增最少分段数

"""

import bisect
import math
from collections import defaultdict
from math import inf
from typing import List

from src.data_structure.sparse_table.template import SparseTable1, SparseTableIndex
from src.mathmatics.prime_factor.template import PrimeFactor
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p2880(ac=FastIO()):
        # 查询静态区间最大值与最小值
        n, q = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(n)]
        st1 = SparseTable1(nums, "max")
        st2 = SparseTable1(nums, "min")
        for _ in range(q):
            a, b = ac.read_list_ints()
            ac.st(st1.query(a, b) - st2.query(a, b))
        return

    @staticmethod
    def lg_p3865(ac=FastIO()):
        # 查询静态区间最大值
        n, m = ac.read_list_ints()
        st = SparseTable1(ac.read_list_ints())
        for _ in range(m):
            x, y = ac.read_list_ints()
            ac.st(st.query(x, y))
        return

    @staticmethod
    def cf_474f(ac=FastIO()):
        # 稀疏表查询静态区间 gcd 与最小值
        n = ac.read_int()
        nums = ac.read_list_ints()
        dct = defaultdict(list)
        for i, num in enumerate(nums):
            dct[num].append(i)
        st_gcd = SparseTable1(nums, "gcd")
        st_min = SparseTable1(nums, "min")
        for _ in range(ac.read_int()):
            x, y = ac.read_list_ints_minus_one()
            num1 = st_gcd.query(x + 1, y + 1)
            num2 = st_min.query(x + 1, y + 1)
            if num1 == num2:
                res = bisect.bisect_right(dct[num1], y) - bisect.bisect_left(dct[num1], x)
                ac.st(y - x + 1 - res)
            else:
                ac.st(y - x + 1)
        return

    @staticmethod
    def ac_109(ac=FastIO()):

        def range_merge_to_disjoint(lst1, lst2):
            a, b = len(lst1), len(lst2)
            x = y = 0
            res = []
            while x < a or y < b:
                if x == a or (y < b and lst2[y] < lst1[x]):
                    res.append(lst2[y])
                    y += 1
                else:
                    res.append(lst1[x])
                    x += 1
            return res

        def check(lst1):
            k = len(lst1)
            x, y = 0, k - 1
            res = cnt = 0
            while x < y and cnt < m:
                res += (lst1[x] - lst1[y]) ** 2
                if res > t:
                    return False
                x += 1
                y -= 1
                cnt += 1
            return True

        # 利用倍增与归并sorting的思想数组划分
        for _ in range(ac.read_int()):
            n, m, t = ac.read_list_ints()
            nums = ac.read_list_ints()
            ans = i = 0
            while i < n:
                p = 1
                lst = [nums[i]]
                right = i
                while p and right < n:
                    cur = nums[right + 1:right + p + 1]
                    cur.sort()
                    tmp = range_merge_to_disjoint(lst, cur)
                    if check(tmp):
                        lst = tmp[:]
                        right += p
                        p *= 2
                    else:
                        p //= 2
                ans += 1
                i = right + 1
            ac.st(ans)
        return

    @staticmethod
    def lg_p5648(ac=FastIO()):
        # 倍增 ST 表查询区间最大值的索引，monotonic_stack|建树距离
        n, t = ac.read_list_ints()
        nums = ac.read_list_ints()
        post = [n] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                post[stack.pop()] = i
            stack.append(i)
        edge = [[] for _ in range(n + 1)]
        for i in range(n):
            edge[post[i]].append(i)
        # 建树距离
        sub = [0] * (n + 1)
        stack = [n]
        while stack:
            i = stack.pop()
            for j in edge[i]:
                sub[j] = sub[i] + nums[j] * (i - j)
                stack.append(j)
        # 区间最大值索引
        st = SparseTableIndex(nums)
        last_ans = 0
        for _ in range(t):
            u, v = ac.read_list_ints()
            left = 1 + (u ^ last_ans) % n
            q = 1 + (v ^ (last_ans + 1)) % (n - left + 1)
            right = left + q - 1
            ceil_ind = st.query(left, right)
            last_ans = sub[left - 1] - sub[ceil_ind] + nums[ceil_ind] * (right - ceil_ind)
            ac.st(last_ans)
        return

    @staticmethod
    def lc_2447(nums: List[int], k: int) -> int:
        # 最大公因数等于 K 的子数组数目
        ans = 0
        pre = dict()
        for num in nums:
            cur = dict()
            for p in pre:
                x = math.gcd(p, num)
                if x % k == 0:
                    cur[x] = cur.get(x, 0) + pre[p]
            if num % k == 0:
                cur[num] = cur.get(num, 0) + 1
            ans += cur.get(k, 0)
            pre = cur
        return ans

    @staticmethod
    def lc_2470(nums: List[int], k: int) -> int:
        # 最小公倍数为 K 的子数组数目
        ans = 0
        pre = dict()
        for num in nums:
            cur = dict()
            for p in pre:
                x = math.lcm(p, num)
                if k % x == 0:
                    cur[x] = cur.get(x, 0) + pre[p]
            if k % num == 0:
                cur[num] = cur.get(num, 0) + 1
            ans += cur.get(k, 0)
            pre = cur
        return ans

    @staticmethod
    def lc_2411(nums: List[int]) -> List[int]:
        # 最大或值的最短连续子数组
        n = len(nums)
        ans = [0] * n
        post = dict()
        for i in range(n - 1, -1, -1):
            cur = dict()
            num = nums[i]
            for x in post:
                y = cur.get(x | num, inf)
                cur[x | num] = y if y < post[x] else post[x]
            cur[num] = i
            post = cur
            ans[i] = post[max(post)] - i + 1
        return ans

    @staticmethod
    def cf_1516d(ac=FastIO()):
        pf = PrimeFactor(10 ** 5)
        n, q = ac.read_list_ints()
        nums = ac.read_list_ints()
        post = [n] * (n + 1)
        ind = [n] * (10 ** 5)
        for i in range(n - 1, -1, -1):
            right = post[i + 1]
            for p, _ in pf.prime_factor[nums[i]]:
                right = ac.min(ind[p], right)
                ind[p] = i
            post[i] = right

        col = max(2, math.ceil(math.log2(n)))
        dp = [[n] * col for _ in range(n)]
        for i in range(n):
            dp[i][0] = post[i]
        for j in range(1, col):
            for i in range(n):
                father = dp[i][j - 1]
                if father <= n - 1:
                    dp[i][j] = dp[father][j - 1]

        for _ in range(q):
            x, y = ac.read_list_ints_minus_one()
            ans = 0
            for j in range(col - 1, -1, -1):
                if dp[x][j] <= y:
                    x = dp[x][j]
                    ans += 1 << j
            ac.st(ans + 1)
        return

    @staticmethod
    def cf_1709d(ac=FastIO()):
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        st = SparseTable1(nums, "max")
        for _ in range(ac.read_int()):
            x1, y1, x2, y2, k = ac.read_list_ints()
            if x1 % k != x2 % k or y1 % k != y2 % k:
                ac.st("NO")
                continue
            if y1 == y2:
                ac.st("YES")
                continue
            if y1 > y2:
                y1, y2 = y2, y1
            ceil = st.query(y1, y2)
            y = (n - x1) // k
            w = k * y + x1
            if w <= ceil:
                ac.st("NO")
            else:
                ac.st("YES")
        return

    @staticmethod
    def cf_1878e(ac=FastIO()):
        # 连续子数组的and信息
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            q = ac.read_int()
            query = [dict() for _ in range(n)]
            res = []
            for _ in range(q):
                ll, k = ac.read_list_ints()
                ll -= 1
                res.append([ll, k])
                query[ll][k] = -2
            post = dict()
            for i in range(n - 1, -1, -1):
                cur = dict()
                num = nums[i]
                for p in post:
                    x = p & num
                    if x not in cur or post[p] > cur[x]:
                        cur[x] = post[p]
                if num not in cur:
                    cur[num] = i
                lst = sorted(query[i].keys(), reverse=True)
                val = [[num, cur[num]] for num in cur]
                val.sort(reverse=True)
                right = -2
                m = len(val)
                p = 0
                for ke in lst:
                    while p < m and val[p][0] >= ke:
                        _, xx = val[p]
                        if xx > right:
                            right = xx
                        p += 1
                    query[i][ke] = right
                post = cur.copy()
            ac.lst([query[ll][k] + 1 for ll, k in res])
        return

    @staticmethod
    def lc_1521(arr: List[int], target: int) -> int:
        # 与目标值最接近的连续子数组bit_operation与值
        ans = abs(arr[0] - target)
        pre = {arr[0]}
        for num in arr[1:]:
            pre = {num & p for p in pre}
            pre.add(num)
            for x in pre:
                if abs(x - target) < ans:
                    ans = abs(x - target)
        return ans