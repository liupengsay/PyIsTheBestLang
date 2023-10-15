import unittest
from typing import List, Optional

"""
算法：二叉树、N叉树、先序遍历、中序遍历、后序遍历、迭代实现、前序遍历
功能：
参考：
题目：

===================================力扣===================================
94. 二叉树的中序遍历（https://leetcode.cn/problems/binary-tree-inorder-traversal/description/）中序遍历迭代写法
144. 二叉树的前序遍历（https://leetcode.cn/problems/binary-tree-preorder-traversal/description/）前序遍历迭代写法
145. 二叉树的后序遍历（https://leetcode.cn/problems/binary-tree-postorder-traversal/）后序遍历迭代写法

===================================AcWing===================================
19. 二叉树的下一个节点（https://www.acwing.com/problem/content/description/31/）使用中序遍历找到其节点后序下一个


"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TreeOrder:
    def __init__(self):
        # 二叉树或者N叉树的前、中、后序写法
        return

    @staticmethod
    def post_order(root: Optional[TreeNode]) -> List[int]:
        ans = []
        stack = [[root, 1]] if root else []
        while stack:
            node, state = stack.pop()
            if state:
                # 后序
                stack.append([node, 0])
                if node.right:
                    stack.append([node.right, 1])
                if node.left:
                    stack.append([node.left, 1])
            else:
                ans.append(node.val)
        return ans

    @staticmethod
    def pre_order(root: Optional[TreeNode]) -> List[int]:
        ans = []
        stack = [[root, 1]] if root else []
        while stack:
            node, state = stack.pop()
            if state:
                if node.right:
                    stack.append([node.right, 1])
                if node.left:
                    stack.append([node.left, 1])
                # 前序
                stack.append([node, 0])
            else:
                ans.append(node.val)
        return ans

    @staticmethod
    def in_order(root: Optional[TreeNode]) -> List[int]:
        ans = []
        stack = [[root, 1]] if root else []
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
                ans.append(node.val)
        return ans

    @staticmethod
    def ac_19(q):

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


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_145(root: Optional[TreeNode]) -> List[int]:
        return TreeOrder().post_order(root)
    
    @staticmethod
    def lc_94(root: Optional[TreeNode]) -> List[int]:
        return TreeOrder().in_order(root)

    @staticmethod
    def lc_144(root: Optional[TreeNode]) -> List[int]:
        return TreeOrder().pre_order(root)


class TestGeneral(unittest.TestCase):

    def test_xxxx(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
