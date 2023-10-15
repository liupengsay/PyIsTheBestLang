


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



