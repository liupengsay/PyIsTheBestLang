"""
Algorithm：bst|binary_search_tree|array_to_bst|implemention
Description：build a binary_search_tree by the order of array


====================================LeetCode====================================
1569（https://leetcode.com/problems/number-of-ways-to-reorder-array-to-get-same-bst/）array_to_bst|dp|comb|counter|specific_plan
1902（https://leetcode.com/problems/depth-of-bst-given-insertion-order/）array_to_bst|tree_depth|implemention

=====================================LuoGu======================================
2171（https://www.luogu.com.cn/problem/P2171）array_to_bst|reverse_order|union_find|implemention

"""
from typing import List

from src.graph.bst.template import BST, ans1, BinarySearchTreeByArray
from src.mathmatics.comb_perm.template import Combinatorics
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p2171_1(ac=FastIO()):
        # 模板: bst 标准插入 O(n^2)
        ac.read_int()
        nums = ac.read_list_ints()
        bst = BST(nums[0])
        for num in nums[1:]:
            bst.insert(num)
        ac.st(f"deep={ans1}")
        bst.post_order()
        return

    @staticmethod
    def lg_p2171_2(ac=FastIO()):

        # 模板: bst linked_list|与二叉树implemention插入 O(nlogn)
        @ac.bootstrap
        def dfs(rt):
            if ls[rt]:
                yield dfs(ls[rt])
            if rs[rt]:
                yield dfs(rs[rt])
            ac.st(a[rt])
            yield

        n = ac.read_int()
        m = n + 10

        # sorting后discretization
        a = [0] + ac.read_list_ints()
        b = a[:]
        a.sort()
        ind = {a[i]: i for i in range(n + 1)}
        b = [ind[x] for x in b]
        del ind

        # 初始化序号
        pre = [i - 1 for i in range(m)]
        nxt = [i + 1 for i in range(m)]
        dep = [0] * m
        u = [0] * m
        d = [0] * m

        for i in range(n, 0, -1):
            t = b[i]
            u[t] = pre[t]
            d[t] = nxt[t]
            nxt[pre[t]] = nxt[t]
            pre[nxt[t]] = pre[t]

        ls = [0] * (n + 1)
        rs = [0] * (n + 1)
        root = b[1]
        dep[b[1]] = 1
        deep = 1
        for i in range(2, n + 1):
            f = 0
            t = b[i]
            if n >= u[t] >= 1 and dep[u[t]] + 1 > dep[t]:
                dep[t] = dep[u[t]] + 1
                f = u[t]
            if 1 <= d[t] <= n and dep[d[t]] + 1 > dep[t]:
                dep[t] = dep[d[t]] + 1
                f = d[t]
            if f < t:
                rs[f] = t
            else:
                ls[f] = t
            deep = ac.max(deep, dep[t])
        ac.st(f"deep={deep}")
        dfs(root)
        return

    @staticmethod
    def lg_p2171_3(ac=FastIO()):
        ac.read_int()
        nums = ac.read_list_ints()
        dct = BinarySearchTreeByArray().build_with_unionfind(nums)  # 或者是 build_with_stack
        # 迭代的方式后序遍历
        ans = []
        depth = 0
        stack = [[0, 1]]
        while stack:
            i, d = stack.pop()
            if i >= 0:
                stack.append([~i, d])
                dct[i].sort(key=lambda it: -nums[it])
                for j in dct[i]:
                    stack.append([j, d + 1])
            else:
                i = ~i
                depth = depth if depth > d else d
                ans.append(nums[i])
        ac.st(f"deep={depth}")
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lc_1569(nums: List[int]) -> int:
        # array_to_bst，DP与组合counter求specific_plan数
        dct = BinarySearchTreeByArray().build_with_unionfind(nums)
        mod = 10 ** 9 + 7
        cb = Combinatorics(100000, mod)  # preprocess
        stack = [0]
        n = len(nums)
        ans = [0] * n
        sub = [0] * n
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in dct[i]:
                    stack.append(j)
            else:
                i = ~i
                cur_ans = 1
                cur_sub = sum(sub[j] for j in dct[i])
                sub[i] = cur_sub + 1
                for j in dct[i]:
                    cur_ans *= cb.comb(cur_sub, sub[j]) * ans[j]
                    cur_sub -= sub[j]
                    cur_ans %= mod
                ans[i] = cur_ans
        return (ans[0] - 1) % mod

    @staticmethod
    def lc_1902(order: List[int]) -> int:
        # array_to_bst求深度
        dct = BinarySearchTreeByArray().build_with_unionfind(order)  # 也可以 build_with_stack
        stack = [[0, 1]]
        ans = 1
        while stack:
            i, d = stack.pop()
            for j in dct[i]:
                stack.append([j, d + 1])
                ans = ans if ans > d + 1 else d + 1
        return ans