"""
算法：BST二叉搜索树
功能：根据数字顺序建立二叉搜索树、实时维护
题目：


===================================力扣===================================
1569. 将子数组重新排序得到同一个二叉搜索树的方案数（https://leetcode.com/problems/number-of-ways-to-reorder-array-to-get-same-bst/）按照顺序建立二叉树，使用DP与组合计数求方案数
1902. 给定二叉搜索树的插入顺序求深度（https://leetcode.com/problems/depth-of-bst-given-insertion-order/）按照顺序建立二叉树求深度

===================================洛谷===================================
P2171 Hz吐泡泡（https://www.luogu.com.cn/problem/P2171）依次输入数据生成二叉搜索树，可使用逆序并查集

参考：OI WiKi（xx）
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

        # 模板: bst 链表与二叉树模拟插入 O(nlogn)
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

        # 排序后离散化
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
        # 使用迭代的方式计算后序遍历
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
        # 模板：按照顺序建立二叉树，使用DP与组合计数求方案数
        dct = BinarySearchTreeByArray().build_with_unionfind(nums)
        mod = 10 ** 9 + 7
        cb = Combinatorics(100000, mod)  # 预处理计算
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
        # 模板：按照顺序建立二叉树求深度
        dct = BinarySearchTreeByArray().build_with_unionfind(order)  # 也可以使用 build_with_stack
        stack = [[0, 1]]
        ans = 1
        while stack:
            i, d = stack.pop()
            for j in dct[i]:
                stack.append([j, d + 1])
                ans = ans if ans > d + 1 else d + 1
        return ans
