

import math
from typing import List

MOD = 10**9 + 7


class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def numOfWays(self, nums: List[int]) -> int:

        def insert(node):
            if num > node.val:
                if node.right:
                    insert(node.right)
                else:
                    node.right = TreeNode(num)
            else:
                if node.left:
                    insert(node.left)
                else:
                    node.left = TreeNode(num)
            return

        def dfs(node):
            if not node:
                return [1, 0]
            if not node.left and not node.right:
                return [1, 1]
            res = [1, 1]
            left = dfs(node.left)
            right = dfs(node.right)
            res[1] += left[1] + right[1]
            res[0] = left[0] * right[0] * math.comb(res[1] - 1, left[1])
            res[0] %= MOD
            return res

        # 建立二叉搜索树
        root = TreeNode(nums[0])
        for num in nums[1:]:
            insert(root)
        # 使用树形DP进行方案数计算
        ans = (dfs(root)[0] - 1) % MOD
        return ans
