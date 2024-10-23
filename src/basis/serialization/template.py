from collections import deque
from typing import Optional

from src.basis.tree_node.template import TreeNode


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
