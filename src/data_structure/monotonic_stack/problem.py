"""
Algorithm：monotonic_stack|contribution_method
Function：prefix_suffix|maximum|minimum|second_maximum

====================================LeetCode====================================
85（https://leetcode.com/problems/maximal-rectangle/）brute_force|monotonic_stack|matrix
316（https://leetcode.com/problems/remove-duplicate-letters/）monotonic_stack|hash|counter
321（https://leetcode.com/problems/create-maximum-number/）brute_force|monotonic_stack
1081（https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/）monotonic_stack|hash|counter
2334（https://leetcode.com/problems/subarray-with-elements-greater-than-varying-threshold/）sorting|brute_force|contribution_method
2262（https://leetcode.com/problems/total-appeal-of-a-string/）prefix_suffix|monotonic_stack
2355（https://leetcode.com/problems/maximum-number-of-books-you-can-take/）monotonic_stack|liner_dp
255（https://leetcode.com/problems/verify-preorder-sequence-in-binary-search-tree/）monotonic_stack|pre_order
654（https://leetcode.com/problems/maximum-binary-tree/）monotonic_stack
1130（https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/）monotonic_stack|interval_dp
1504（https://leetcode.com/problems/count-submatrices-with-all-ones/）brute_force|monotonic_stack|counter|sub_matrix
1673（https://leetcode.com/problems/find-the-most-competitive-subsequence/）monotonic_stack|greedy
1776（https://leetcode.com/problems/car-fleet-ii/）monotonic_stack|union_find|linked_list|implemention
1840（https://leetcode.com/problems/maximum-building-height/）monotonic_stack|greedy|prefix_suffix|implemention
1944（https://leetcode.com/problems/number-of-visible-people-in-a-queue/）reverse_thinking|monotonic_stack
1950（https://leetcode.com/problems/maximum-of-minimum-values-in-all-subarrays/）monotonic_stack
2030（https://leetcode.com/problems/smallest-k-length-subsequence-with-occurrences-of-a-letter/）monotonic_stack|greedy|lexicographical_order
2104（https://leetcode.com/problems/sum-of-subarray-ranges/）monotonic_stack|contribution_method
2282（https://leetcode.com/problems/number-of-people-that-can-be-seen-in-a-grid/）monotonic_stack
2289（https://leetcode.com/problems/steps-to-make-array-non-decreasing/）monotonic_stack|implemention
907（https://leetcode.com/problems/sum-of-subarray-minimums/）monotonic_stack|implemention
2454（https://leetcode.com/problems/next-greater-element-iv/description/）monotonic_stack|post_second_larger

=====================================LuoGu======================================
1950（https://www.luogu.com.cn/problem/P1950）brute_force|monotonic_stack|sub_matrix|counter
1901（https://www.luogu.com.cn/problem/P1901）monotonic_stack
2866（https://www.luogu.com.cn/problem/P2866）monotonic_stack
2947（https://www.luogu.com.cn/problem/P2947）monotonic_stack
4147（https://www.luogu.com.cn/problem/P4147）brute_force|monotonic_stack|sub_matrix|area
5788（https://www.luogu.com.cn/problem/P5788）monotonic_stack
7314（https://www.luogu.com.cn/problem/P7314）brute_force|monotonic_stack|pre_larger|post_larger
7399（https://www.luogu.com.cn/problem/P7399）monotonic_stack|greedy
7410（https://www.luogu.com.cn/problem/P7410）inclusion_exclusion|monotonic_stack|counter
7762（https://www.luogu.com.cn/problem/P7762）monotonic_stack|greedy|sorting|contribution_method|area
1578（https://www.luogu.com.cn/problem/P1578）monotonic_stack|discretization|brute_force|sub_matrix|area
3467（https://www.luogu.com.cn/problem/P3467）greedy|monotonic_stack
1191（https://www.luogu.com.cn/problem/P1191）monotonic_stack|sub_matrix|counter
1323（https://www.luogu.com.cn/problem/P1323）heapq|monotonic_stack|lexicographical_order|greedy
2422（https://www.luogu.com.cn/problem/P2422）monotonic_stack|prefix_sum
3467（https://www.luogu.com.cn/problem/P3467）monotonic_stack
6404（https://www.luogu.com.cn/problem/P6404）monotonic_stack|sub_matrix|counter
6503（https://www.luogu.com.cn/problem/P6503）monotonic_stack|counter|contribution_method
6510（https://www.luogu.com.cn/problem/P6510）monotonic_stack|sparse_table|hash|binary_search
6801（https://www.luogu.com.cn/problem/P6801）monotonic_stack|sub_matrix|counter
8094（https://www.luogu.com.cn/problem/P8094）monotonic_stack|pre_larger|post_larger

===================================CodeForces===================================
1795E（https://codeforces.com/problemset/problem/1795/E）monotonic_stack|liner_dp|greedy|counter|brute_force|prefix_suffix|dp
1313C2（https://codeforces.com/problemset/problem/1313/C2）monotonic_stack|liner_dp
1454F（https://codeforces.com/contest/1454/problem/F）monotonic_stack|brute_force

====================================AtCoder=====================================
E - Second Sum（https://atcoder.jp/contests/abc140/tasks/abc140_e）monotonic_stack|pre_pre_larger|post_post_larger

=====================================AcWing=====================================
131（https://www.acwing.com/problem/content/133/）monotonic_stack|sub_matrix
152（https://www.acwing.com/problem/content/description/154/）monotonic_stack|sub_matrix
3780（https://www.acwing.com/problem/content/description/3783/）monotonic_stack|greedy|linear_dp|construction

"""
import bisect
import heapq
from collections import defaultdict, Counter
from typing import List

from src.data_structure.monotonic_stack.template import Rectangle
from src.data_structure.sparse_table.template import SparseTable1
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def abc_140e(ac=FastIO()):
        # monotonic_stack|求下个与下下个严格更大元素与上个与上个个严格更大元素
        n = ac.read_int()
        nums = ac.read_list_ints()

        post = [-1] * n  # 这里可以是n/n-1/null，取决于用途
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
        # monotonic_stack|最大矩形
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
        # monotonic_stack|下下个更大元素
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
    def lg_p1191(ac=FastIO()):
        # brute_force下边界monotonic_stack|矩形个数
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
        # heapq|与monotonic_stack|，最大lexicographical_order数字
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
        # monotonic_stack|与prefix_sum
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
        # monotonic_stack|
        n = ac.read_int()
        nums = [ac.read_list_ints()[1] for _ in range(n)]
        stack = []
        ans = 0
        for i in range(n):
            while stack and nums[stack[-1]] >= nums[i]:
                j = stack.pop()
                if nums[j] == nums[i]:  # 同样高度的连续区域可以一次覆盖
                    ans += 1
            stack.append(i)
        ac.st(n - ans)
        return

    @staticmethod
    def lg_p1598(ac=FastIO()):

        # monotonic_stack|discretizationbrute_force障碍点的最大面积矩形
        def compute_area_obstacle(lst):
            nonlocal ans
            # monotonic_stack|根据高度最大矩形面积
            m = len(height)
            left = [0] * m
            right = [m - 1] * m
            stack = []
            for i in range(m):
                while stack and height[stack[-1]] > height[i]:
                    right[stack.pop()] = i  # 注意这里不减 1 了是边界
                if stack:  # 这里可以同时求得数组前后的下一个大于等于值
                    left[i] = stack[-1]  # 这里将相同的值视为右边的更大且并不会影响，注意这里不| 1 了是边界
                stack.append(i)

            for i in range(m):
                cur = height[i] * (lst[right[i]] - lst[left[i]])
                ans = ans if ans > cur else cur
            return ans

        length, n = ac.read_list_ints()
        q = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(q)]
        node_row = defaultdict(list)
        node_col = defaultdict(list)
        for x, y in nums:
            node_row[y].append(x)
            node_col[x].append(y)

        # brute_force矩形上下两行边界
        y_axis = sorted([y for _, y in nums] + [0, n], reverse=True)
        ans = 0
        col = defaultdict(lambda: n)
        x_axis = sorted([x for x, _ in nums] + [0, length])
        for y in y_axis:
            height = [col[x] - y for x in x_axis]
            compute_area_obstacle(x_axis)
            for x in node_row[y]:
                col[x] = y

        # brute_force矩形左右两列边界
        x_axis.reverse()
        y_axis.reverse()
        row = defaultdict(lambda: length)
        for x in x_axis:
            height = [row[y] - x for y in y_axis]
            compute_area_obstacle(y_axis)
            for y in node_col[x]:
                row[y] = x
        ac.st(ans)
        return

    @staticmethod
    def lc_255(preorder: List[int]) -> bool:
        # monotonic_stack|判断是否为前序序列

        pre_max = float("-inf")
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
        # monotonic_stack|最大矩形面积
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
        # monotonic_stack|最大矩形面积
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
            ans = ac.max(ans, Rectangle().compute_area(pre))
        ac.st(3 * ans)
        return

    @staticmethod
    def lg_p1950(ac=FastIO()):
        # monotonic_stack|矩形个数
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
        # monotonic_stack|具有相同数字的子矩形个数
        m, n = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        ans = 0
        rt = Rectangle()
        pre = [[0, 0] for _ in range(n)]
        # brute_force子矩形的下边界
        for i in range(m):
            for j in range(n):
                if pre[j][0] == grid[i][j]:
                    pre[j][1] += 1
                else:
                    pre[j] = [grid[i][j], 1]
            # 按照相同数字分段counter
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
        # monotonic_stack|连续子序列的最大值最小值贡献counter
        m = ac.read_int()
        nums = [ac.read_int() for _ in range(m)]
        left = [0] * m
        right = [m - 1] * m
        stack = []
        for i in range(m):
            while stack and nums[stack[-1]] < nums[i]:
                right[stack.pop()] = i - 1
            if stack:  # 这里可以同时求得数组前后的下一个大于等于值
                left[i] = stack[-1] + 1  # 这里将相同的值视为右边的更大且并不会影响
            stack.append(i)
        ans = sum((right[i] - i + 1) * nums[i] * (i - left[i] + 1) for i in range(m))

        left = [0] * m
        right = [m - 1] * m
        stack = []
        for i in range(m):
            while stack and nums[stack[-1]] > nums[i]:
                right[stack.pop()] = i - 1
            if stack:  # 这里可以同时求得数组前后的下一个大于等于值
                left[i] = stack[-1] + 1  # 这里将相同的值视为右边的更大且并不会影响
            stack.append(i)
        ans -= sum((right[i] - i + 1) * nums[i] * (i - left[i] + 1) for i in range(m))
        ac.st(ans)
        return

    @staticmethod
    def lg_p6510(ac=FastIO()):
        # monotonic_stack|sparse_table|hashbinary_search
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
        st = SparseTable1(nums)
        ans = 0
        for i in range(n):
            x = st.query(i + 1, post[i] + 1)
            if x == nums[i]:
                continue
            j = bisect.bisect_left(dct[x], i)
            ans = ac.max(ans, dct[x][j] - i + 1)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6801(ac=FastIO()):
        # monotonic_stack|矩形个数

        def compute(x, y):
            return x * (x + 1) * y * (y + 1) // 4

        ans = 0
        mod = 10 ** 9 + 7
        n = ac.read_int()
        h = ac.read_list_ints()
        w = ac.read_list_ints()
        # 削峰维持单调递增
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
        # 反向剩余
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
        # monotonic_stack|应用
        n = ac.read_int()
        nums = ac.read_list_ints()
        ans = 0
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                ans += i - stack.pop() + 1  # 当前作为较大值
            if stack:
                ans += i - stack[-1] + 1  # 当前作为较小值
            stack.append(i)
        ac.st(ans)
        return

    @staticmethod
    def lc_316(s: str) -> str:
        # monotonic_stack|结合hash与counter
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
        # monotonic_stack|implemention
        mod = 10 ** 9 + 7
        n = len(nums)
        post = [n - 1] * n  # 这里可以是n/n-1/null，取决于用途
        pre = [0] * n  # 这里可以是0/-1/null，取决于用途
        stack = []
        for i in range(n):  # 这里也可以是从n-1到0reverse_order|，取决于用途
            while stack and nums[stack[-1]] <= nums[i]:  # 这里可以是"<" ">" "<=" ">="，取决于需要判断的大小关系
                post[stack.pop()] = i - 1  # 这里可以是i或者i-1，取决于是否包含i作为右端点
            if stack:  # 这里不一定可以同时，比如前后都是大于等于时，只有前后所求范围互斥时，可以
                pre[i] = stack[-1] + 1  # 这里可以是stack[-1]或者stack[-1]+1，取决于是否包含stack[-1]作为左端点
            stack.append(i)
        return sum(nums[i] * (i - pre[i] + 1) * (post[i] - i + 1) for i in range(n)) % mod

    @staticmethod
    def lc_1081(s: str) -> str:
        # monotonic_stack|结合hash与counter
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
        # monotonic_stack|greedy删除选取
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
        # monotonic_stack|greedy，也可以prefix_suffix数组implemention
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
        # 下一个或者上一个不同字符的位置
        n = len(s)
        pre = defaultdict(lambda: -1)
        ans = 0
        for i in range(n):
            ans += (i - pre[s[i]]) * (n - i)
            pre[s[i]] = i
        return ans

    @staticmethod
    def lc_2355(books: List[int]) -> int:
        # monotonic_stack|优化liner_dp
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
        # monotonic_stack|优化liner_dp
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
                    nums[j] = ac.min(nums[j], nums[j - 1])
                for j in range(i - 1, -1, -1):
                    nums[j] = ac.min(nums[j], nums[j + 1])
                ac.lst(nums)
                break
        return

    @staticmethod
    def cf_1795e(ac=FastIO()):
        # monotonic_stack|优化liner_dp，greedycounterbrute_force，prefix_suffixDP转移
        for _ in range(ac.read_int()):

            def check():
                res = [0] * n
                stack = []
                for i in range(n):
                    while stack and nums[stack[-1]] - stack[-1] > nums[i] - i:
                        stack.pop()
                    if not stack:
                        k = ac.min(i, nums[i] - 1)
                        res[i] = k * (nums[i] - 1 + nums[i] - k) // 2
                    else:
                        k = ac.min(i - stack[-1] - 1, nums[i] - 1)
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
        # monotonic_stack|也可以区间DP
        stack = [float('inf')]
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
        # brute_force上下边界monotonic_stack|全为 1 的子矩形个数
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
        # monotonic_stack|线性greedyDPconstruction
        n = ac.read_int()
        nums = ac.read_list_ints()
        if n == 1:
            ac.lst(nums)
            return

        n = len(nums)

        # 后面更小的
        post = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] > nums[i]:
                post[stack.pop()] = i
            stack.append(i)

        # 前面更小的
        pre = [-1] * n
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] > nums[i]:
                pre[stack.pop()] = i
            stack.append(i)

        left = [0] * n
        left[0] = nums[0]
        for i in range(1, n):
            j = pre[i]
            if j != -1:
                left[i] += nums[i] * (i - j) + left[j]
            else:
                left[i] = nums[i] * (i + 1)

        right = [0] * n
        right[-1] = nums[-1]
        for i in range(n - 2, -1, -1):
            j = post[i]
            if j != -1:
                right[i] += nums[i] * (j - i) + right[j]
            else:
                right[i] = (n - i) * nums[i]

        # brute_force先上升后下降的最高点
        dp = [left[i] + right[i] - nums[i] for i in range(n)]
        x = dp.index(max(dp))
        for i in range(x + 1, n):
            nums[i] = ac.min(nums[i - 1], nums[i])
        for i in range(x - 1, -1, -1):
            nums[i] = ac.min(nums[i + 1], nums[i])
        ac.lst(nums)
        return