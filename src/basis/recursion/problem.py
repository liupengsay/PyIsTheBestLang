"""
Algorithm：divide_and_conquer|recursion|n-tree|pre_order|mid_order|post_order|iteration
Description：recursion|iteration

====================================LeetCode====================================
1545（https://leetcode.com/problems/find-kth-bit-in-nth-binary-string/）recursion|implemention
894（https://leetcode.com/problems/all-possible-full-binary-trees/）catalan_num|recursion|implemention
880（https://leetcode.com/problems/decoded-string-at-index/）recursion|implemention
932（https://leetcode.com/problems/beautiful-array/description/）recursion|divide_and_conquer|construction
889（https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/）recursion|divide_and_conquer|construction
1028（https://leetcode.com/problems/recover-a-tree-from-preorder-traversal/description/）pre_order|recursion|construction|2-tree

=====================================LuoGu======================================
1911（https://www.luogu.com.cn/problem/P1911）4-tree|recursion|matrix
5461（https://www.luogu.com.cn/problem/P5461）recursion|4-tree|matrix
5551（https://www.luogu.com.cn/problem/P5551）pre_order|2-tree|recursion
5626（https://www.luogu.com.cn/problem/P5626）divide_and_conquer|dp|merge_sort
2907（https://www.luogu.com.cn/problem/P2907）recursion|implemention
7673（https://www.luogu.com.cn/problem/P7673）mid_order|recursion|2-tree
1228（https://www.luogu.com.cn/problem/P1228）4-tree|divide_and_conquer|recursion|matrix
1185（https://www.luogu.com.cn/problem/P1185）2-tree|recursion

===================================CodeForces===================================
448C（https://codeforces.com/contest/448/problem/C）greedy|recursion|dp

98（https://www.acwing.com/problem/content/100/）4-tree|recursion|matrix_rotate
93（https://www.acwing.com/problem/content/95/）recursion|comb|iteration
118（https://www.acwing.com/problem/content/120/）recursion

"""
from functools import lru_cache
from typing import Optional, List

from src.basis.tree_node.template import TreeNode
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_880(t: str, m: int) -> str:
        # recursionimplemention

        def dfs(s, k):
            n = len(s)
            cur = 0
            for i in range(n):
                if s[i].isnumeric():
                    d = int(s[i])
                    if cur * d >= k:
                        return dfs(s[:i], k % cur + cur * int(k % cur == 0))
                    cur *= d
                else:
                    if cur + 1 == k:
                        return s[i]
                    cur += 1

        return dfs(t, m)

    def lc_889(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        # recursiondivide_and_conquerconstruction
        if not preorder:
            return

        n = len(preorder)
        root = preorder[0]
        ans = TreeNode(root)
        if n == 1:
            return ans
        val = preorder[1]
        i = postorder.index(val)  # 注意前序遍历与后序遍历的数组特点
        left_cnt = i + 1
        ans.left = self.lc_889(preorder[1:left_cnt + 1], postorder[:left_cnt])
        ans.right = self.lc_889(preorder[left_cnt + 1:], postorder[left_cnt:-1])
        return ans

    @lru_cache(None)
    def lc_894(self, n: int) -> List[Optional[TreeNode]]:

        # 类似catalan_number的recursionimplemention生成
        if n % 2 == 0:
            return []
        if n == 1:
            return [TreeNode(0)]

        ans = []
        for i in range(1, n - 1):
            for left in self.lc_894(i):
                for right in self.lc_894(n - i - 1):
                    node = TreeNode(0)
                    node.left = left
                    node.right = right
                    ans.append(node)
        return ans

    @lru_cache(None)
    def lc_932(self, n: int) -> List[int]:
        # recursiondivide_and_conquerconstruction
        if n == 1:
            return [1]
        return [2 * x - 1 for x in self.lc_932((n + 1) // 2)] + [2 * x for x in self.lc_932(n // 2)]

    @staticmethod
    def lc_1028(traversal: str) -> Optional[TreeNode]:

        # 根据先序遍历recursionconstruction二叉树
        ans = ""
        pre = 0
        for w in traversal:
            if w == "-":
                pre += 1
            else:
                if pre:
                    ans += "(" + str(pre) + ")"
                pre = 0
                ans += w

        def dfs(s, d):
            if not s:
                return
            c = "(" + str(d) + ")"
            lst = s.split(c)
            root = TreeNode(int(lst[0]))
            if len(lst) > 1:
                root.left = dfs(lst[1], d + 1)
            if len(lst) > 2:
                root.right = dfs(lst[2], d + 1)
            return root

        return dfs(ans, 1)

    @staticmethod
    def lc_1345(a: int, b: int) -> str:

        # recursionimplemention
        def dfs(n, k):

            if n == 1 and k == 1:
                return '0'
            if k == 2 ** (n - 1):
                return '1'
            if k < 2 ** (n - 1):
                return dfs(n - 1, k)
            k -= 2 ** (n - 1)
            ans = dfs(n - 1, 2 ** (n - 1) - k)
            return '1' if ans == '0' else '0'

        return dfs(a, b)

    @staticmethod
    def lg_p1911(n, x, y):

        # recursion处理四叉树

        def dfs(x1, y1, x2, y2, a, b):
            nonlocal ind
            if x1 == x2 and y1 == y2:
                return

            # 确定是哪一个角被占用了
            flag = find(x1, y1, x2, y2, a, b)
            x0 = x1 + (x2 - x1) // 2
            y0 = y1 + (y2 - y1) // 2

            # 四叉树中心邻居节点
            lst = [[x0, y0], [x0, y0 + 1], [x0 + 1, y0], [x0 + 1, y0 + 1]]
            nex = []
            for ii in range(4):
                if ii != flag:
                    ans[lst[ii][0]][lst[ii][1]] = ind
                    nex.append(lst[ii])
                else:
                    nex.append([a, b])
            ind += 1
            # 四叉树recursion坐标
            dfs(x1, y1, x0, y0, nex[0][0], nex[0][1])
            dfs(x1, y0 + 1, x0, y2, nex[1][0], nex[1][1])
            dfs(x0 + 1, y1, x2, y0, nex[2][0], nex[2][1])
            dfs(x0 + 1, y0 + 1, x2, y2, nex[3][0], nex[3][1])
            return

        def find(x1, y1, x2, y2, a, b):
            x0 = x1 + (x2 - x1) // 2
            y0 = y1 + (y2 - y1) // 2
            if x1 <= a <= x0 and y1 <= b <= y0:
                return 0
            if x1 <= a <= x0 and y0 + 1 <= b <= y2:
                return 1
            if x0 + 1 <= a <= x2 and y1 <= b <= y0:
                return 2
            return 3

        x -= 1
        y -= 1
        m = 1 << n
        ans = [[0] * m for _ in range(m)]
        ind = 1
        # recursion生成
        dfs(0, 0, m - 1, m - 1, x, y)

        # hash化处理
        dct = dict()
        dct[0] = 0
        for i in range(m):
            for j in range(m):
                x = ans[i][j]
                if x not in dct:
                    dct[x] = len(dct)
        return [[dct[i] for i in a] for a in ans]

    @staticmethod
    def cf_448c(ac=FastIO()):
        # greedyrecursionDP
        ac.read_int()
        nums = ac.read_list_ints()

        @ac.bootstrap
        def dfs(arr):
            m = len(arr)
            low = min(arr)
            cur = [num - low for num in arr]
            ans = low
            i = 0
            while i < m:
                if cur[i] == 0:
                    i += 1
                    continue
                j = i
                while j < m and cur[j] > 0:
                    j += 1
                ans += yield dfs(cur[i: j])
                i = j
            yield ac.min(ans, m)

        ac.st(dfs(nums))
        return

    @staticmethod
    def ac_98(ac=FastIO()):

        # 四叉树recursion与坐标旋转变换
        for _ in range(ac.read_int()):
            n, a, b = ac.read_list_ints()
            a -= 1
            b -= 1

            def check(nn, mm):
                # recursion改成迭代写法极大提升速度
                stack = [[nn, mm]]
                x = y = -1
                while stack:
                    if stack[-1][0] == 0:
                        x = y = 0
                        stack.pop()
                        continue
                    else:
                        nn, mm = stack[-1]
                        cc = 2 ** (2 * nn - 2)
                        if x != -1:
                            stack.pop()
                            z = mm // cc
                            length = 2 ** (nn - 1)
                            if z == 0:
                                x, y = y, x
                            elif z == 1:
                                x, y = x, y + length
                            elif z == 2:
                                x, y = x + length, y + length
                            else:
                                x, y = 2 * length - y - 1, length - x - 1
                        else:
                            stack.append([nn - 1, mm % cc])
                return x, y

            x1, y1 = check(n, a)
            x2, y2 = check(n, b)
            ans = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 * 10
            ans = int(ans) + int(ans - int(ans) >= 0.5)
            ac.st(ans)
        return

    @staticmethod
    def ac_93_1(ac=FastIO()):
        n, m = ac.read_list_ints()

        # recursion实现选取

        def dfs(i):
            if len(pre) == m:
                ac.lst(pre)
                return
            if i == n:
                return

            dfs(i + 1)
            pre.append(i + 1)
            dfs(i + 1)
            pre.pop()
            return

        pre = []
        dfs(0)
        return

    @staticmethod
    def ac_93_2(ac=FastIO()):
        n, m = ac.read_list_ints()

        # 迭代实现选取
        pre = []
        stack = [[0, 0]]
        while stack:
            i, state = stack.pop()
            if i >= 0:
                stack.append([~i, state])
                if len(pre) == m:
                    ac.lst(pre)
                    continue
                if i == n:
                    continue
                stack.append([i + 1, 0])
                pre.append(i + 1)
                stack.append([i + 1, 1])
            else:
                if state:
                    pre.pop()
        return

    @staticmethod
    def ac_118(ac=FastIO()):
        # 迭代方式recursion
        dp = []
        for i in range(1, 8):
            n = 3 ** (i - 1)
            ans = [[" "] * n for _ in range(n)]
            stack = [[0, 0, n - 1, n - 1]]
            while stack:
                x1, y1, x2, y2 = stack.pop()
                if x1 == x2 and y1 == y2:
                    ans[x1][y1] = "X"
                    continue
                m = x2 - x1 + 1
                b = m // 3
                col = {0: 0, 2: 2, 4: 1, 6: 0, 8: 2}
                for x in [0, 2, 4, 6, 8]:
                    start = [x1 + b * (x // 3), y1 + b * col[x]]
                    end = [x1 + b * (x // 3) + b - 1, y1 + b * col[x] + b - 1]
                    stack.append([start[0], start[1], end[0], end[1]])
            dp.append(["".join(a) for a in ans])

        while True:
            n = ac.read_int()
            if n == -1:
                break
            for a in dp[n - 1]:
                ac.st(a)
            ac.st("-")

        return