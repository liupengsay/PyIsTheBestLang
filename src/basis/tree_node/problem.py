"""
Algorithm：n-tree|pre_order|post_order|mid_order
Description：

====================================LeetCode====================================
94（https://leetcode.cn/problems/binary-tree-inorder-traversal/description/）mid_order
144（https://leetcode.cn/problems/binary-tree-preorder-traversal/description/）pre_order
145（https://leetcode.cn/problems/binary-tree-postorder-traversal/）post_order
105（https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/）pre_order|in_order|construction|classical
106（https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/）in_order|post_order|construction|classical

=====================================AcWing=====================================
19（https://www.acwing.com/problem/content/description/31/）mid_order

=====================================AtCoder=====================================
ABC255F（https://atcoder.jp/contests/abc255/tasks/abc255_f）pre_order|in_order|construction|classical


"""
from typing import List, Optional

from src.basis.tree_node.template import TreeNode, TreeOrder
from src.util.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_145(root: Optional[TreeNode]) -> List[int]:
        """
        url: https://leetcode.cn/problems/binary-tree-postorder-traversal/
        tag: post_order
        """
        return TreeOrder().post_order(root)

    @staticmethod
    def lc_94(root: Optional[TreeNode]) -> List[int]:
        """
        url: https://leetcode.cn/problems/binary-tree-inorder-traversal/description/
        tag: mid_order
        """
        return TreeOrder().in_order(root)

    @staticmethod
    def lc_144(root: Optional[TreeNode]) -> List[int]:
        """
        url: https://leetcode.cn/problems/binary-tree-preorder-traversal/description/
        tag: pre_order
        """
        return TreeOrder().pre_order(root)

    @staticmethod
    def ac_19(q):
        """
        url: https://www.acwing.com/problem/content/description/31/
        tag: mid_order
        """
        x = q
        while q.father:
            q = q.father

        ans = []
        stack = [[q, 1]] if q else []
        while stack:
            node, state = stack.pop()
            if state:
                if node.right:
                    stack.append([node.right, 1])
                stack.append([node, 0])
                if node.left:
                    stack.append([node.left, 1])
            else:
                ans.append(node)

        i = ans.index(x)
        if i == len(ans) - 1:
            return
        return ans[i + 1]


    @staticmethod
    def abc_255f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc255/tasks/abc255_f
        tag: pre_order|in_order|construction|classical
        """
        n = ac.read_int()
        pre_order = ac.read_list_ints_minus_one()

        in_order = ac.read_list_ints_minus_one()
        if pre_order[0] != 0:
            ac.st(-1)
            return

        ind = {num: i for i, num in enumerate(in_order)}
        stack = [(0, n - 1, 0, n - 1, 0)]
        sub = [[0, 0] for _ in range(n)]
        while stack:
            x1, y1, x2, y2, fa = stack.pop()
            if y1 - x1 != y2 - x2:
                ac.st(-1)
                return
            if x1 == y1:
                if pre_order[x1] != in_order[x2]:
                    ac.st(-1)
                    return
                continue
            if not x2 <= ind[pre_order[x1]] <= y2:
                ac.st(-1)
                return
            x0 = ind[pre_order[x1]]
            left_cnt = x0 - x2
            right_cnt = y2 - x0
            if left_cnt:
                sub[pre_order[fa]][0] = pre_order[x1 + 1] + 1
                stack.append((x1 + 1, x1 + left_cnt, x2, x0 - 1, x1 + 1))
            if right_cnt:
                sub[pre_order[fa]][1] = pre_order[x1 + left_cnt + 1] + 1
                stack.append((x1 + left_cnt + 1, y1, x0 + 1, y2, x1 + left_cnt + 1))
        for w in sub:
            ac.lst(w)
        return