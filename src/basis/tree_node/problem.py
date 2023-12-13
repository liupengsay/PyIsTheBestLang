"""
Algorithm：n-tree|pre_order|post_order|mid_order
Description：

====================================LeetCode====================================
94（https://leetcode.cn/problems/binary-tree-inorder-traversal/description/）mid_order
144（https://leetcode.cn/problems/binary-tree-preorder-traversal/description/）pre_order
145（https://leetcode.cn/problems/binary-tree-postorder-traversal/）post_order

=====================================AcWing=====================================
19（https://www.acwing.com/problem/content/description/31/）mid_order


"""
from typing import List, Optional

from src.basis.tree_node.template import TreeNode, TreeOrder


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
                # 中序
                stack.append([node, 0])
                if node.left:
                    stack.append([node.left, 1])
            else:
                ans.append(node)

        i = ans.index(x)
        if i == len(ans) - 1:
            return
        return ans[i + 1]
