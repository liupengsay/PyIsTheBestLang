import unittest
from collections import deque
from typing import Optional

from src.basis.tree_node import TreeNode
from src.fast_io import FastIO

"""
算法：序列化与反序列化
功能：将二叉树、N叉树等数据结构序列化为字符串，并通过字符串反序列化恢复数据
参考：
题目：

===================================力扣===================================
428. 序列化和反序列化 N 叉树（https://leetcode.cn/problems/serialize-and-deserialize-n-ary-tree/）序列化模板题
297. 二叉树的序列化与反序列化（https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/）序列化模板题
449. 序列化和反序列化二叉搜索树（https://leetcode.cn/problems/serialize-and-deserialize-bst/）序列化模板题

===================================洛谷===================================
xx（xxx）xxx

================================CodeForces================================
xx（xxx）xxx

"""


class CodecBFS:

    @staticmethod
    def serialize(root: Optional[TreeNode]) -> str:
        """Encodes a tree to a single string.
        """
        stack = deque([root]) if root else deque()
        res = []
        while stack:
            node = stack.popleft()
            if not node:
                res.append("n")
                continue
            else:
                res.append(str(node.val))
                stack.append(node.left)
                stack.append(node.right)
        return ",".join(res)

    @staticmethod
    def deserialize(data: str) -> Optional[TreeNode]:
        """Decodes your encoded data to tree.
        """
        if not data:
            return
        lst = deque(data.split(","))
        ans = TreeNode(int(lst.popleft()))
        stack = deque([ans])
        while lst:
            left, right = lst.popleft(), lst.popleft()
            pre = stack.popleft()
            if left != "n":
                pre.left = TreeNode(int(left))
                stack.append(pre.left)
            if right != "n":
                pre.right = TreeNode(int(right))
                stack.append(pre.right)
        return ans


class CodecDFS:

    @staticmethod
    def serialize(root: TreeNode) -> str:
        """Encodes a tree to a single string.
        """
        def dfs(node):
            if not node:
                return "n"
            return dfs(node.right) + "," + dfs(node.left) + "," + str(node.val)
        return dfs(root)

    @staticmethod
    def deserialize(data: str) -> TreeNode:
        """Decodes your encoded data to tree.
        """
        lst = data.split(",")

        def dfs():
            if not lst:
                return
            val = lst.pop()
            if val == "n":
                return
            root = TreeNode(int(val))
            root.left = dfs()
            root.right = dfs()
            return root
        return dfs()


class Solution:
    def __int__(self):
        return

    @staticmethod
    def xx_xxxx(ac=FastIO()):
        pass
        return


class TestGeneral(unittest.TestCase):

    def test_xxxx(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
