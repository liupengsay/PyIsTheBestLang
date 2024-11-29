"""
Algorithm：monotonic_stack|contribution_method
Description：prefix_suffix|maximum|minimum|second_maximum

====================================LeetCode====================================
85（https://leetcode.cn/problems/maximal-rectangle/）brute_force|monotonic_stack|matrix
316（https://leetcode.cn/problems/remove-duplicate-letters/）monotonic_stack|hash|counter
321（https://leetcode.cn/problems/create-maximum-number/）brute_force|monotonic_stack
1081（https://leetcode.cn/problems/smallest-subsequence-of-distinct-characters/）monotonic_stack|hash|counter
2334（https://leetcode.cn/problems/subarray-with-elements-greater-than-varying-threshold/）sort|brute_force|contribution_method
2262（https://leetcode.cn/problems/total-appeal-of-a-string/）prefix_suffix|monotonic_stack
2355（https://leetcode.cn/problems/maximum-number-of-books-you-can-take/）monotonic_stack|liner_dp
255（https://leetcode.cn/problems/verify-preorder-sequence-in-binary-search-tree/）monotonic_stack|pre_order
654（https://leetcode.cn/problems/maximum-binary-tree/）monotonic_stack
1130（https://leetcode.cn/problems/minimum-cost-tree-from-leaf-values/）monotonic_stack|interval_dp
1504（https://leetcode.cn/problems/count-submatrices-with-all-ones/）brute_force|monotonic_stack|counter|sub_matrix
1673（https://leetcode.cn/problems/find-the-most-competitive-subsequence/）monotonic_stack|greedy
1776（https://leetcode.cn/problems/car-fleet-ii/）monotonic_stack|union_find|linked_list|implemention
1840（https://leetcode.cn/problems/maximum-building-height/）monotonic_stack|greedy|prefix_suffix|implemention
1944（https://leetcode.cn/problems/number-of-visible-people-in-a-queue/）reverse_thinking|monotonic_stack
1950（https://leetcode.cn/problems/maximum-of-minimum-values-in-all-subarrays/）monotonic_stack
2030（https://leetcode.cn/problems/smallest-k-length-subsequence-with-occurrences-of-a-letter/）monotonic_stack|greedy|lexicographical_order
2104（https://leetcode.cn/problems/sum-of-subarray-ranges/）monotonic_stack|contribution_method
2282（https://leetcode.cn/problems/number-of-people-that-can-be-seen-in-a-grid/）monotonic_stack
2289（https://leetcode.cn/problems/steps-to-make-array-non-decreasing/）monotonic_stack|implemention
907（https://leetcode.cn/problems/sum-of-subarray-minimums/）monotonic_stack|implemention
2454（https://leetcode.cn/problems/next-greater-element-iv/description/）monotonic_stack|post_second_larger
2866（https://leetcode.cn/problems/beautiful-towers-ii/）monotonic_stack|greedy
3113（https://leetcode.com/problems/find-the-number-of-subarrays-where-boundary-elements-are-maximum/）brute_force|two_pointer|monotonic_stack|classical）

=====================================LuoGu======================================
P1950（https://www.luogu.com.cn/problem/P1950）brute_force|monotonic_stack|sub_matrix|counter
P1901（https://www.luogu.com.cn/problem/P1901）monotonic_stack
P2866（https://www.luogu.com.cn/problem/P2866）monotonic_stack
P2947（https://www.luogu.com.cn/problem/P2947）monotonic_stack
P4147（https://www.luogu.com.cn/problem/P4147）brute_force|monotonic_stack|sub_matrix|area
P5788（https://www.luogu.com.cn/problem/P5788）monotonic_stack
P7314（https://www.luogu.com.cn/problem/P7314）brute_force|monotonic_stack|pre_larger|post_larger
P7399（https://www.luogu.com.cn/problem/P7399）monotonic_stack|greedy
P7410（https://www.luogu.com.cn/problem/P7410）inclusion_exclusion|monotonic_stack|counter
P7762（https://www.luogu.com.cn/problem/P7762）monotonic_stack|greedy|sort|contribution_method|area
P1578（https://www.luogu.com.cn/problem/P1578）monotonic_stack|discretization|brute_force|sub_matrix|area
P3467（https://www.luogu.com.cn/problem/P3467）greedy|monotonic_stack
P1191（https://www.luogu.com.cn/problem/P1191）monotonic_stack|sub_matrix|counter
P1323（https://www.luogu.com.cn/problem/P1323）heapq|monotonic_stack|lexicographical_order|greedy
P2422（https://www.luogu.com.cn/problem/P2422）monotonic_stack|prefix_sum
P3467（https://www.luogu.com.cn/problem/P3467）monotonic_stack
P6404（https://www.luogu.com.cn/problem/P6404）monotonic_stack|sub_matrix|counter
P6503（https://www.luogu.com.cn/problem/P6503）monotonic_stack|counter|contribution_method
P6510（https://www.luogu.com.cn/problem/P6510）monotonic_stack|sparse_table|hash|binary_search
P6801（https://www.luogu.com.cn/problem/P6801）monotonic_stack|sub_matrix|counter
P8094（https://www.luogu.com.cn/problem/P8094）monotonic_stack|pre_larger|post_larger

===================================CodeForces===================================
1795E（https://codeforces.com/problemset/problem/1795/E）monotonic_stack|liner_dp|greedy|counter|brute_force|prefix_suffix|dp
1313C2（https://codeforces.com/problemset/problem/1313/C2）monotonic_stack|liner_dp
1454F（https://codeforces.com/contest/1454/problem/F）monotonic_stack|brute_force
1092D2（https://codeforces.com/contest/1092/problem/D2）monotonic_stack|implemention
1092D1（https://codeforces.com/contest/1092/problem/D1）brain_teaser|greedy|implemention
1506G（https://codeforces.com/contest/1506/problem/G）monotonic_stack|greedy|classical
1919D（https://codeforces.com/problemset/problem/1919/D）brain_teaser|monotonic_stack|construction
1299C（https://codeforces.com/problemset/problem/1299/C）monotonic_stack|convex|brain_teaser|greedy|construction
1407D（https://codeforces.com/problemset/problem/1407/D）array_implemention|monotonic_stack|linear_dp|classical
2035D（https://codeforces.com/contest/2035/problem/D）monotonic_stack|data_range|classical

====================================AtCoder=====================================
ABC140E（https://atcoder.jp/contests/abc140/tasks/abc140_e）monotonic_stack|pre_pre_larger|post_post_larger
ABC336D（https://atcoder.jp/contests/abc336/tasks/abc336_d）monotonic_stack|linear_dp
ABC311G（https://atcoder.jp/contests/abc311/tasks/abc311_g）brute_force|prefix_sum|monotonic_stack|classical
ABC299G（https://atcoder.jp/contests/abc299/tasks/abc299_g）monotonic_stack|greedy|implemention|classical
ABC359E（https://atcoder.jp/contests/abc359/tasks/abc359_e）monotonic_stack|implemention|classical
ABC189C（https://atcoder.jp/contests/abc189/tasks/abc189_c）monotonic_stack|contribution_method|action_scope|classical
ABC379F（https://atcoder.jp/contests/abc379/tasks/abc379_f）sparse_table|monotonic_stack|binary_search

=====================================AcWing=====================================
131（https://www.acwing.com/problem/content/133/）monotonic_stack|sub_matrix
152（https://www.acwing.com/problem/content/description/154/）monotonic_stack|sub_matrix
3780（https://www.acwing.com/problem/content/description/3783/）monotonic_stack|greedy|linear_dp|construction

=====================================CodeChef=====================================
cc_1（https://www.codechef.com/problems/CNTEMPTY?tab=statement）monotonic_stack|contribution_method|prefix_sum

"""
import bisect
import heapq
import math
from collections import defaultdict, Counter
from typing import List

from src.basis.binary_search.template import BinarySearch
from src.basis.diff_array.template import PreFixSumMatrix
from src.struct.monotonic_stack.template import Rectangle
from src.struct.sparse_table.template import SparseTable
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def abc_140e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc140/tasks/abc140_e
        tag: monotonic_stack|pre_pre_larger|post_post_larger|classical|contribution_method
        """
        n = ac.read_int()
        nums = ac.read_list_ints()

        post = [-1] * n
        post2 = [-1] * n
        stack1 = []
        stack2 = []
        for i in range(n):
            while stack2 and stack2[0][0] < nums[i]:
                post2[heapq.heappop(stack2)[1]] = i
            while stack1 and nums[stack1[-1]] < nums[i]:
                j = stack1.pop()
                post[j] = i
                heapq.heappush(stack2, [nums[j], j])
            stack1.append(i)

        pre = [-1] * n
        pre2 = [-1] * n
        stack1 = []
        stack2 = []
        for i in range(n - 1, -1, -1):
            while stack2 and stack2[0][0] < nums[i]:
                pre2[heapq.heappop(stack2)[1]] = i
            while stack1 and nums[stack1[-1]] < nums[i]:
                j = stack1.pop()
                pre[j] = i
                heapq.heappush(stack2, [nums[j], j])
            stack1.append(i)

        # action_scope
        ans = 0
        for i in range(n):
            if pre[i] == -1:
                left_0 = i
                left_1 = 0
            else:
                left_0 = i - pre[i] - 1
                if pre2[i] == -1:
                    left_1 = pre[i] + 1
                else:
                    left_1 = pre[i] - pre2[i]

            if post[i] == -1:
                right_0 = n - 1 - i
                right_1 = 0
            else:
                right_0 = post[i] - i - 1
                if post2[i] == -1:
                    right_1 = n - 1 - post[i] + 1
                else:
                    right_1 = post2[i] - post[i]
            cnt = left_1 * (right_0 + 1) + right_1 * (left_0 + 1)
            ans += cnt * nums[i]
        ac.st(ans)
        return

    @staticmethod
    def ac_131(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/133/
        tag: monotonic_stack|sub_matrix
        """
        while True:
            lst = ac.read_list_ints()
            if lst[0] == 0:
                break
            n = lst.pop(0)
            post = [n - 1] * n
            pre = [0] * n
            stack = []
            for i in range(n):
                while stack and lst[stack[-1]] > lst[i]:
                    post[stack.pop()] = i - 1
                if stack:
                    pre[i] = stack[-1] + 1
                stack.append(i)
            ans = max(lst[i] * (post[i] - pre[i] + 1) for i in range(n))
            ac.st(ans)
        return

    @staticmethod
    def lc_2454(nums: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/next-greater-element-iv/description/
        tag: monotonic_stack|post_second_larger
        """
        n = len(nums)
        ans = [-1] * n
        stack1 = []
        stack2 = []
        for i in range(n):
            while stack2 and stack2[0][0] < nums[i]:
                ans[heapq.heappop(stack2)[1]] = nums[i]
            while stack1 and nums[stack1[-1]] < nums[i]:
                j = stack1.pop()
                heapq.heappush(stack2, [nums[j], j])
            stack1.append(i)
        return ans

    @staticmethod
    def lc_2866(max_heights: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/beautiful-towers-ii/
        tag: monotonic_stack|greedy
        """
        n = len(max_heights)
        pre = [0] * (n + 1)
        stack = []
        s = 0
        for i in range(n):
            h = max_heights[i]
            c = 1
            while stack and stack[-1][0] >= h:
                v, cc = stack.pop()
                c += cc
                s -= v * cc
            s += h * c
            pre[i + 1] = s
            stack.append([h, c])

        post = [0] * (n + 1)
        stack = []
        s = 0
        for i in range(n - 1, -1, -1):
            h = max_heights[i]
            c = 1
            while stack and stack[-1][0] >= h:
                v, cc = stack.pop()
                c += cc
                s -= v * cc
            s += h * c
            post[i] = s
            stack.append([h, c])

        return max(pre[i + 1] + post[i] - max_heights[i] for i in range(n))

    @staticmethod
    def lg_p1191(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1191
        tag: monotonic_stack|sub_matrix|counter|classical
        """
        n = ac.read_int()
        pre = [0] * n
        ans = 0
        for _ in range(n):
            s = ac.read_str()
            right = [n - 1] * n
            left = [0] * n
            stack = []
            for j in range(n):
                if s[j] == "W":
                    pre[j] += 1
                else:
                    pre[j] = 0
                while stack and pre[stack[-1]] > pre[j]:
                    right[stack.pop()] = j - 1
                if stack:
                    left[j] = stack[-1] + 1
                stack.append(j)
            ans += sum(pre[j] * (right[j] - j + 1) * (j - left[j] + 1) for j in range(n))
        ac.st(ans)
        return

    @staticmethod
    def lg_p1323(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1323
        tag: heapq|monotonic_stack|lexicographical_order|greedy
        """
        k, m = ac.read_list_ints()
        dct = set()
        ans = []
        stack = [1]
        while len(ans) < k:
            num = heapq.heappop(stack)
            if num in dct:
                continue
            ans.append(num)
            dct.add(num)
            heapq.heappush(stack, 2 * num + 1)
            heapq.heappush(stack, 4 * num + 5)

        res = "".join(str(x) for x in ans)
        ac.st(res)
        rem = m
        stack = []
        for w in res:
            while stack and rem and w > stack[-1]:
                stack.pop()
                rem -= 1
            stack.append(w)
        stack = stack[rem:]
        ac.st(int("".join(stack)))
        return

    @staticmethod
    def lg_p2422(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2422
        tag: monotonic_stack|prefix_sum
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        lst = ac.accumulate(nums)
        post = [n - 1] * n
        pre = [0] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] > nums[i]:
                post[stack.pop()] = i - 1
            if stack:
                pre[i] = stack[-1] + 1
            stack.append(i)
        ans = max(nums[i] * (lst[post[i] + 1] - lst[pre[i]]) for i in range(n))
        ac.st(ans)
        return

    @staticmethod
    def lg_p3467(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3467
        tag: monotonic_stack|classical
        """
        n = ac.read_int()
        nums = [ac.read_list_ints()[1] for _ in range(n)]
        stack = []
        ans = 0
        for i in range(n):
            while stack and nums[stack[-1]] >= nums[i]:
                j = stack.pop()
                if nums[j] == nums[i]:
                    ans += 1
            stack.append(i)
        ac.st(n - ans)
        return

    @staticmethod
    def lg_p1578(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1578
        tag: monotonic_stack|discretization|brute_force|sub_matrix|area
        """
        m, n = ac.read_list_ints()
        q = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(q)]
        node_col = [[] for _ in range(n + 1)]
        for x, y in nums:
            node_col[y].append(x)

        y_axis = sorted(set([y for _, y in nums] + [0, n]), reverse=True)
        ans = 0
        col = [n] * (m + 1)
        x_axis = sorted(set([x for x, _ in nums] + [0, m]))
        k = len(x_axis)
        for y in y_axis:
            height = [col[x] - y for x in x_axis]
            left = [0] * k
            right = [k - 1] * k
            stack = []
            for i in range(k):
                while stack and height[stack[-1]] > height[i]:
                    right[stack.pop()] = i
                if stack:
                    left[i] = stack[-1]
                stack.append(i)

            for i in range(k):
                cur = height[i] * (x_axis[right[i]] - x_axis[left[i]])
                ans = ans if ans > cur else cur

            for x in node_col[y]:
                col[x] = y
        # special case judge
        ceil = max(x_axis[i + 1] - x_axis[i] for i in range(k - 1))
        if ceil * n > ans:
            ans = ceil * n
        ac.st(ans)
        return

    @staticmethod
    def lc_255(preorder: List[int]) -> bool:
        """
        url: https://leetcode.cn/problems/verify-preorder-sequence-in-binary-search-tree/
        tag: monotonic_stack|pre_order|classical
        """
        pre_max = -math.inf
        n = len(preorder)
        stack = []
        for i in range(n):
            if preorder[i] < pre_max:
                return False
            while stack and preorder[stack[-1]] < preorder[i]:
                cur = preorder[stack.pop()]
                pre_max = pre_max if pre_max > cur else cur
            stack.append(i)
        return True

    @staticmethod
    def lc_85(matrix: List[List[str]]) -> int:
        """
        url: https://leetcode.cn/problems/maximal-rectangle/
        tag: brute_force|monotonic_stack|matrix
        """
        m, n = len(matrix), len(matrix[0])
        pre = [0] * n
        ans = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == "1":
                    pre[j] += 1
                else:
                    pre[j] = 0
            ans = max(ans, Rectangle().compute_area(pre))
        return ans

    @staticmethod
    def lg_p4147(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4147
        tag: brute_force|monotonic_stack|sub_matrix|area
        """
        n, m = ac.read_list_ints()
        pre = [0] * m
        ans = 0
        for _ in range(n):
            lst = ac.read_list_strs()
            for i in range(m):
                if lst[i] == "F":
                    pre[i] += 1
                else:
                    pre[i] = 0
            ans = max(ans, Rectangle().compute_area(pre))
        ac.st(3 * ans)
        return

    @staticmethod
    def lg_p1950(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1950
        tag: brute_force|monotonic_stack|sub_matrix|counter
        """
        m, n = ac.read_list_ints()
        ans = 0
        pre = [0] * n
        for _ in range(m):
            s = ac.read_str()
            for j in range(n):
                if s[j] == ".":
                    pre[j] += 1
                else:
                    pre[j] = 0
            ans += Rectangle().compute_number(pre)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6404(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6404
        tag: monotonic_stack|sub_matrix|counter
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        ans = 0
        rt = Rectangle()
        pre = [[0, 0] for _ in range(n)]
        for i in range(m):
            for j in range(n):
                if pre[j][0] == grid[i][j]:
                    pre[j][1] += 1
                else:
                    pre[j] = [grid[i][j], 1]

            lst = [pre[0][1]]
            num = pre[0][0]
            for x, c in pre[1:]:
                if x == num:
                    lst.append(c)
                else:
                    ans += rt.compute_number(lst)
                    lst = [c]
                    num = x
            ans += rt.compute_number(lst)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6503(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6503
        tag: monotonic_stack|counter|contribution_method
        """
        m = ac.read_int()
        nums = [ac.read_int() for _ in range(m)]
        left = [0] * m
        right = [m - 1] * m
        stack = []
        for i in range(m):
            while stack and nums[stack[-1]] < nums[i]:
                right[stack.pop()] = i - 1
            if stack:
                left[i] = stack[-1] + 1
            stack.append(i)
        ans = sum((right[i] - i + 1) * nums[i] * (i - left[i] + 1) for i in range(m))

        left = [0] * m
        right = [m - 1] * m
        stack = []
        for i in range(m):
            while stack and nums[stack[-1]] > nums[i]:
                right[stack.pop()] = i - 1
            if stack:
                left[i] = stack[-1] + 1
            stack.append(i)
        ans -= sum((right[i] - i + 1) * nums[i] * (i - left[i] + 1) for i in range(m))
        ac.st(ans)
        return

    @staticmethod
    def lg_p6510(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6510
        tag: monotonic_stack|sparse_table|hash|binary_search|classical
        """
        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]
        post = [n - 1] * n
        stack = []
        dct = defaultdict(list)
        for i in range(n):
            while stack and nums[stack[-1]] >= nums[i]:
                post[stack.pop()] = i - 1
            stack.append(i)
            dct[nums[i]].append(i)
        st = SparseTable(nums, max)
        ans = 0
        for i in range(n):
            x = st.query(i + 1, post[i] + 1)
            if x == nums[i]:
                continue
            j = bisect.bisect_left(dct[x], i)
            ans = max(ans, dct[x][j] - i + 1)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6801(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6801
        tag: monotonic_stack|sub_matrix|counter
        """

        def compute(x, y):
            return x * (x + 1) * y * (y + 1) // 4

        ans = 0
        mod = 10 ** 9 + 7
        n = ac.read_int()
        h = ac.read_list_ints()
        w = ac.read_list_ints()

        stack = []
        for i in range(n):
            ww, hh = w[i], h[i]
            while stack and stack[-1][1] >= hh:
                www, hhh = stack.pop()
                if stack and stack[-1][1] >= hh:
                    max_h = stack[-1][1]
                    ans += compute(www, hhh) - compute(www, max_h)
                    ans %= mod
                    stack[-1][0] += www
                else:
                    ww += www
                    ans += compute(www, hhh) - compute(www, hh)
                    ans %= mod
            stack.append([ww, hh])

        ww, hh = stack.pop()
        while stack:
            www, hhh = stack.pop()
            ans += compute(ww, hh) - compute(ww, hhh)
            ans %= mod
            ww += www
            hh = hhh
        ans += compute(ww, hh)
        ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def lg_p8094(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8094
        tag: monotonic_stack|pre_larger|post_larger
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        ans = 0
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                ans += i - stack.pop() + 1
            if stack:
                ans += i - stack[-1] + 1
            stack.append(i)
        ac.st(ans)
        return

    @staticmethod
    def lc_316(s: str) -> str:
        """
        url: https://leetcode.cn/problems/remove-duplicate-letters/
        tag: monotonic_stack|hash|counter
        """
        cnt = Counter(s)
        in_stack = defaultdict(int)
        stack = []
        for w in s:
            if not in_stack[w]:
                while stack and stack[-1] > w and cnt[stack[-1]]:
                    in_stack[stack.pop()] = 0
                stack.append(w)
                in_stack[w] = 1
            cnt[w] -= 1
        return "".join(stack)

    @staticmethod
    def lc_907(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/sum-of-subarray-minimums/
        tag: monotonic_stack|implemention|contribution_method|classical
        """
        mod = 10 ** 9 + 7
        n = len(nums)
        post = [n - 1] * n
        pre = [0] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                post[stack.pop()] = i - 1
            if stack:
                pre[i] = stack[-1] + 1
            stack.append(i)
        return sum(nums[i] * (i - pre[i] + 1) * (post[i] - i + 1) for i in range(n)) % mod

    @staticmethod
    def lc_1081(s: str) -> str:
        """
        url: https://leetcode.cn/problems/smallest-subsequence-of-distinct-characters/
        tag: monotonic_stack|hash|counter
        """
        cnt = Counter(s)
        in_stack = defaultdict(int)
        stack = []
        for w in s:
            if not in_stack[w]:
                while stack and stack[-1] > w and cnt[stack[-1]]:
                    in_stack[stack.pop()] = 0
                stack.append(w)
                in_stack[w] = 1
            cnt[w] -= 1
        return "".join(stack)

    @staticmethod
    def lc_1673(nums: List[int], k: int) -> List[int]:
        """
        url: https://leetcode.cn/problems/find-the-most-competitive-subsequence/
        tag: monotonic_stack|greedy
        """
        n = len(nums)
        rem = n - k
        stack = []
        for num in nums:
            while stack and stack[-1] > num and rem:
                rem -= 1
                stack.pop()
            stack.append(num)
        return stack[:k]

    @staticmethod
    def lc_1840(n: int, restrictions: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-building-height/
        tag: monotonic_stack|greedy|prefix_suffix|implemention
        """
        restrictions.sort()
        stack = [[1, 0]]
        for idx, height in restrictions:
            if height - stack[-1][1] >= idx - stack[-1][0]:
                continue
            while idx - stack[-1][0] <= stack[-1][1] - height:
                stack.pop()
            stack.append([idx, height])

        height = stack[-1][1] + n - stack[-1][0]
        for i in range(len(stack) - 1):
            tmp = (stack[i + 1][0] - stack[i][0] + stack[i][1] + stack[i + 1][1]) // 2
            height = height if height > tmp else tmp
        return height

    @staticmethod
    def lc_2262(s: str) -> int:
        """
        url: https://leetcode.cn/problems/total-appeal-of-a-string/
        tag: prefix_suffix|monotonic_stack
        """
        n = len(s)
        pre = defaultdict(lambda: -1)
        ans = 0
        for i in range(n):
            ans += (i - pre[s[i]]) * (n - i)
            pre[s[i]] = i
        return ans

    @staticmethod
    def lc_2355(books: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-number-of-books-you-can-take/
        tag: monotonic_stack|liner_dp
        """
        n = len(books)
        dp = [0] * n
        stack = []
        for i in range(n):

            while stack and stack[-1][0] >= books[i] - i:
                stack.pop()

            end = books[i]
            size = i + 1 if not stack else i - stack[-1][1]
            size = size if size < end else end
            cur = (end + end - size + 1) * size // 2

            dp[i] = dp[stack[-1][1]] + cur if stack else cur
            stack.append([books[i] - i, i])
        return max(dp)

    @staticmethod
    def cf_1313c2(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1313/C2
        tag: monotonic_stack|liner_dp|specific_plan
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        pre = [0] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] > nums[i]:
                stack.pop()
            if not stack:
                pre[i] = nums[i] * (i + 1)
            else:
                pre[i] = pre[stack[-1]] + nums[i] * (i - stack[-1])
            stack.append(i)

        post = [0] * n
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] > nums[i]:
                stack.pop()
            if not stack:
                post[i] = nums[i] * (n - i)
            else:
                post[i] = post[stack[-1]] + nums[i] * (stack[-1] - i)
            stack.append(i)

        ceil = max(pre[i] + post[i] - nums[i] for i in range(n))
        for i in range(n):
            if pre[i] + post[i] - nums[i] == ceil:
                for j in range(i + 1, n):
                    nums[j] = min(nums[j], nums[j - 1])
                for j in range(i - 1, -1, -1):
                    nums[j] = min(nums[j], nums[j + 1])
                ac.lst(nums)
                break
        return

    @staticmethod
    def cf_1795e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1795/E
        tag: monotonic_stack|liner_dp|greedy|counter|brute_force|prefix_suffix|dp
        """
        for _ in range(ac.read_int()):

            def check():
                res = [0] * n
                stack = []
                for i in range(n):
                    while stack and nums[stack[-1]] - stack[-1] > nums[i] - i:
                        stack.pop()
                    if not stack:
                        k = min(i, nums[i] - 1)
                        res[i] = k * (nums[i] - 1 + nums[i] - k) // 2
                    else:
                        k = min(i - stack[-1] - 1, nums[i] - 1)
                        res[i] = k * (nums[i] - 1 + nums[i] - k) // 2 + nums[stack[-1]] + res[stack[-1]]
                    stack.append(i)
                return res

            n = ac.read_int()
            nums = ac.read_list_ints()
            pre = check()
            nums.reverse()
            post = check()
            ans = sum(nums) - max(pre[i] + post[n - 1 - i] for i in range(n))
            ac.st(ans)
        return

    @staticmethod
    def lc_1130(arr: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-cost-tree-from-leaf-values/
        tag: monotonic_stack|interval_dp|classical
        """
        stack = [math.inf]
        res = 0
        for num in arr:
            while stack and stack[-1] <= num:
                cur = stack.pop(-1)
                res += min(stack[-1] * cur, cur * num)
            stack.append(num)
        m = len(stack)
        for i in range(m - 2, 0, -1):
            res += stack[i] * stack[i + 1]
        return res

    @staticmethod
    def lc_1504(mat: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/count-submatrices-with-all-ones/
        tag: brute_force|monotonic_stack|counter|sub_matrix
        """
        m, n = len(mat), len(mat[0])
        ans = 0
        rec = Rectangle()
        pre = [0] * n
        for j in range(m):
            for k in range(n):
                if mat[j][k]:
                    pre[k] += 1
                else:
                    pre[k] = 0
            ans += rec.compute_number(pre)
        return ans

    @staticmethod
    def ac_3780(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/3783/
        tag: monotonic_stack|greedy|linear_dp|construction|CF1313C2
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        pre = [0] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] > nums[i]:
                stack.pop()
            if not stack:
                pre[i] = nums[i] * (i + 1)
            else:
                pre[i] = pre[stack[-1]] + nums[i] * (i - stack[-1])
            stack.append(i)

        post = [0] * n
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] > nums[i]:
                stack.pop()
            if not stack:
                post[i] = nums[i] * (n - i)
            else:
                post[i] = post[stack[-1]] + nums[i] * (stack[-1] - i)
            stack.append(i)

        ceil = max(pre[i] + post[i] - nums[i] for i in range(n))
        for i in range(n):
            if pre[i] + post[i] - nums[i] == ceil:
                for j in range(i + 1, n):
                    nums[j] = min(nums[j], nums[j - 1])
                for j in range(i - 1, -1, -1):
                    nums[j] = min(nums[j], nums[j + 1])
                ac.lst(nums)
                break
        return

    @staticmethod
    def lc_1944(heights: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/number-of-visible-people-in-a-queue/
        tag: reverse_thinking|monotonic_stack
        """
        n = len(heights)
        ans = [0] * n
        stack = [heights[-1]]
        for i in range(n - 2, -1, -1):
            cur = 0
            while stack and stack[-1] < heights[i]:
                cur += 1
                stack.pop()
            if stack:
                cur += 1
            stack.append(heights[i])
            ans[i] = cur
        return ans

    @staticmethod
    def abc_336d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc336/tasks/abc336_d
        tag: monotonic_stack|linear_dp
        """

        def check():
            res = [0] * n
            stack = []
            for i in range(n):
                while stack and nums[stack[-1]] - stack[-1] >= nums[i] - i:
                    stack.pop()
                if not stack:
                    k = min(i + 1, nums[i])
                    res[i] = k
                else:
                    k = res[stack[-1]] + i - stack[-1]
                    res[i] = k
                stack.append(i)
            return res

        n = ac.read_int()
        nums = ac.read_list_ints()
        pre = check()
        nums.reverse()
        post = check()
        ans = max(min(pre[i], post[n - 1 - i]) for i in range(n))
        ac.st(ans)
        return

    @staticmethod
    def cf_1506g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1506/problem/G
        tag: monotonic_stack|greedy|classical
        """
        for _ in range(ac.read_int()):
            s = ac.read_str()
            cnt = Counter(s)
            stack = []
            in_stack = set()
            for w in s:
                while stack and stack[-1] <= w and cnt[stack[-1]] and w not in in_stack:
                    in_stack.discard(stack.pop())
                if w not in in_stack:
                    stack.append(w)
                    in_stack.add(w)
                cnt[w] -= 1
            ac.st("".join(stack))
        return

    @staticmethod
    def abc_311g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc311/tasks/abc311_g
        tag: brute_force|prefix_sum|monotonic_stack|classical
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        ans = 0
        lst = []
        for g in grid:
            lst.extend(g)
        pre = PreFixSumMatrix(grid)

        for num in set(lst):
            cur = [[int(x == num) for x in g] for g in grid]
            pre2 = PreFixSumMatrix(cur)
            height = [0] * n
            for i in range(m):
                for j in range(n):
                    if grid[i][j] >= num:
                        height[j] += 1
                    else:
                        height[j] = 0

                width = Rectangle().compute_width(height)
                for j in range(n):
                    if height[j]:
                        ll, rr = width[j]
                        xa, ya = i - height[j] + 1, ll
                        xb, yb = i, rr
                        if pre2.query(xa, ya, xb, yb):
                            ans = max(ans, pre.query(xa, ya, xb, yb) * num)
        ac.st(ans)
        return

    @staticmethod
    def abc_299g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc299/tasks/abc299_g
        tag: monotonic_stack_greedy|implemention|classical
        """
        ac.read_list_ints()
        nums = ac.read_list_ints()
        cnt = Counter(nums)
        in_stack = defaultdict(int)
        stack = []
        for w in nums:
            if not in_stack[w]:
                while stack and stack[-1] > w and cnt[stack[-1]]:
                    in_stack[stack.pop()] = 0
                stack.append(w)
                in_stack[w] = 1
            cnt[w] -= 1
        ac.lst(stack)
        return

    @staticmethod
    def lc_3113(nums: List[int]) -> int:
        """
        url: https://leetcode.com/problems/find-the-number-of-subarrays-where-boundary-elements-are-maximum/
        tag: brute_force|two_pointer|monotonic_stack|classical
        """

        n = len(nums)
        post = [n] * n
        stack = []
        for i in range(n):
            while stack and nums[i] > nums[stack[-1]]:
                post[stack.pop()] = i
            stack.append(i)
        ans = 0
        dct = defaultdict(list)
        for i, num in enumerate(nums):
            dct[num].append(i)
        for num in dct:
            lst = dct[num]
            j = 0
            m = len(lst)
            for i in range(m):
                if j < i:
                    j = i
                while j + 1 < m and post[lst[i]] >= lst[j + 1]:
                    j += 1
                ans += j - i + 1
        return ans

    @staticmethod
    def cf_1919d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1919/D
        tag: brain_teaser|monotonic_stack|construction
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            if nums.count(0) != 1:
                ac.st("NO")
                continue
            pre_smaller = [-1] * n
            stack = []
            for i in range(n):
                while stack and nums[i] <= nums[stack[-1]]:
                    stack.pop()
                if stack:
                    pre_smaller[i] = stack[-1]
                stack.append(i)

            post_smaller = [-1] * n
            stack = []
            for i in range(n - 1, -1, -1):
                while stack and nums[i] <= nums[stack[-1]]:
                    stack.pop()
                if stack:
                    post_smaller[i] = stack[-1]
                stack.append(i)
            for i in range(n):
                if nums[i]:
                    if pre_smaller[i] != -1 and nums[pre_smaller[i]] == nums[i] - 1:
                        continue
                    if post_smaller[i] != -1 and nums[post_smaller[i]] == nums[i] - 1:
                        continue
                    ac.st("NO")
                    break
            else:
                ac.st("YES")
        return

    @staticmethod
    def cf_1299c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1299/C
        tag: monotonic_stack|convex|brain_teaser|greedy|construction
        """
        n = ac.read_int()
        a = ac.read_list_ints()
        stack = [(a[0], 0, 0)]
        for i in range(1, n):
            stack.append((a[i], i, i))
            while len(stack) >= 2:
                s1, l1, r1 = stack[-2]
                s2, l2, r2 = stack[-1]
                if s1 * (r2 - l2 + 1) >= s2 * (r1 - l1 + 1):
                    stack.pop()
                    stack.pop()
                    stack.append((s1 + s2, l1, r2))
                else:
                    break
        for ss, ll, rr in stack:
            ans = ss / (rr - ll + 1)
            for _ in range(rr - ll + 1):
                ac.st(ans)
        return

    @staticmethod
    def cf_1407d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1407/D
        tag: array_implemention|monotonic_stack|linear_dp|classical
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        dp = [-1] * (n + 1)
        ceil = [-1] * (n + 1)
        floor = [-1] * (n + 1)
        j1 = j2 = 0
        for i in range(n):
            dp[i + 1] = dp[i] + 1
            while j1 > 0 and nums[ceil[j1]] <= nums[i]:
                x = nums[ceil[j1]]
                j1 -= 1
                if j1 and min(nums[ceil[j1]], nums[i]) > x:
                    dp[i + 1] = min(dp[i + 1], dp[ceil[j1] + 1] + 1)
            while j2 > 0 and nums[floor[j2]] >= nums[i]:
                x = nums[floor[j2]]
                j2 -= 1
                if j2 and max(nums[floor[j2]], nums[i]) < x:
                    dp[i + 1] = min(dp[i + 1], dp[floor[j2] + 1] + 1)
            j1 += 1
            j2 += 1
            ceil[j1] = i
            floor[j2] = i
        ac.st(dp[n])
        return

    @staticmethod
    def abc_379f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc379/tasks/abc379_f
        tag: sparse_table|monotonic_stack|binary_search
        """
        n, q = ac.read_list_ints()
        nums = ac.read_list_ints()
        queries = [[] for _ in range(n)]
        for i in range(q):
            ll, rr = ac.read_list_ints_minus_one()
            queries[rr].append(ll * q + i)
        st = SparseTable(nums, max)
        ans = [0] * q
        stack = []
        for i in range(n - 1, -1, -1):

            def check(xx):
                return nums[stack[xx]] >= ceil

            for val in queries[i]:
                ll, ind = val // q, val % q
                ceil = st.query(ll + 1, i)
                if stack and nums[stack[0]] >= ceil:
                    j = BinarySearch().find_int_right(0, len(stack) - 1, check)
                    ans[ind] = j + 1
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()

            stack.append(i)
        for x in ans:
            ac.st(x)
        return

    @staticmethod
    def cf_2035d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/2035/problem/D
        tag: monotonic_stack|data_range|classical
        """
        mod = 10 ** 9 + 7
        ceil = 32 * 2 * 10 ** 5
        p = [0] * (ceil + 1)
        p[0] = 1
        for i in range(1, ceil + 1):
            p[i] = p[i - 1] * 2 % mod

        def find(x):
            c = 0
            while not x & 1:
                c += 1
                x >>= 1
            return c, x

        for _ in range(ac.read_int()):
            ac.read_int()
            nums = ac.read_list_ints()
            stack = []
            ans = []
            res = 0
            for num in nums:
                cc, xx = find(num)
                while stack and stack[-1][1] <= xx * pow(2, min(32, cc)):
                    ccc, xxx = stack.pop()
                    res -= p[ccc] * xxx
                    res += xxx
                    res %= mod
                    cc += ccc
                res += p[cc] * xx
                res %= mod
                ans.append(res)
                stack.append((cc, xx))
            ac.lst(ans)
        return

    @staticmethod
    def cc_1(ac=FastIO()):
        """
        url: https://www.codechef.com/problems/CNTEMPTY?tab=statement
        tag: monotonic_stack|contribution_method|prefix_sum
        """

        for _ in range(ac.read_int()):
            n = ac.read_int()
            s = ac.read_str()
            lst = [2 * int(w == "A") - 1 for w in s]
            nums = ac.accumulate(lst)
            n = len(nums)

            post = [n - 1] * n  # initial can be n or n-1 or -1 dependent on usage
            pre = [0] * n  # initial can be 0 or -1 dependent on usage
            stack = []
            for i in range(n):  # can be also range(n-1, -1, -1) dependent on usage
                while stack and nums[stack[-1]] < nums[i]:  # can be < or > or <=  or >=  dependent on usage
                    post[stack.pop()] = i - 1  # can be i or i-1 dependent on usage
                if stack:  # which can be done only pre and post are no-repeat such as post bigger and pre not-bigger
                    pre[i] = stack[-1] + 1  # can be stack[-1] or stack[-1]-1 dependent on usage
                stack.append(i)
            ans = sum(nums[i] * (i - pre[i] + 1) * (post[i] - i + 1) for i in range(n))

            post = [n - 1] * n  # initial can be n or n-1 or -1 dependent on usage
            pre = [0] * n  # initial can be 0 or -1 dependent on usage
            stack = []
            for i in range(n):  # can be also range(n-1, -1, -1) dependent on usage
                while stack and nums[stack[-1]] > nums[i]:  # can be < or > or <=  or >=  dependent on usage
                    post[stack.pop()] = i - 1  # can be i or i-1 dependent on usage
                if stack:  # which can be done only pre and post are no-repeat such as post bigger and pre not-bigger
                    pre[i] = stack[-1] + 1  # can be stack[-1] or stack[-1]-1 dependent on usage
                stack.append(i)

            ans -= sum(nums[i] * (i - pre[i] + 1) * (post[i] - i + 1) for i in range(n))
            ac.st(ans)
        return
