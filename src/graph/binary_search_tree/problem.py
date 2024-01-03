"""
Algorithm：binary_search_tree|binary_search_tree|array_to_bst|implemention
Description：build a binary_search_tree by the order of array


====================================LeetCode====================================
1569（https://leetcode.cn/problems/number-of-ways-to-reorder-array-to-get-same-bst/）array_to_bst|dp|comb|counter|specific_plan
1902（https://leetcode.cn/problems/depth-of-bst-given-insertion-order/）array_to_bst|tree_depth|implemention

=====================================LuoGu======================================
P2171（https://www.luogu.com.cn/problem/P2171）array_to_bst|reverse_order|union_find|implemention

"""
from typing import List

from src.graph.binary_search_tree.template import BinarySearchTree
from src.mathmatics.comb_perm.template import Combinatorics
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p2171(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2171
        tag: array_to_bst|reverse_order|union_find|implemention
        """
        ac.read_int()
        nums = ac.read_list_ints()
        dct = BinarySearchTree().build_with_unionfind(nums)  # or build_with_stack
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
        """
        url: https://leetcode.cn/problems/number-of-ways-to-reorder-array-to-get-same-bst/
        tag: array_to_bst|dp|comb|counter|specific_plan
        """
        mod = 10 ** 9 + 7
        cb = Combinatorics(1000, mod)
        dct = BinarySearchTree().build_with_unionfind(nums)  # build_with_stack is also ok
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
        """
        url: https://leetcode.cn/problems/depth-of-bst-given-insertion-order/
        tag: array_to_bst|tree_depth|implemention
        """
        dct = BinarySearchTree().build_with_stack(order)  # or build_with_unionfind
        stack = [[0, 1]]
        ans = 1
        while stack:
            i, d = stack.pop()
            for j in dct[i]:
                stack.append([j, d + 1])
                ans = ans if ans > d + 1 else d + 1
        return ans
