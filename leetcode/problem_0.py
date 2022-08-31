import bisect

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:

        def dfs(node):
            nonlocal ans, pre
            if not node:
                return
            pre += node.val
            ans += dct[pre - targetSum]
            dct[pre] += 1
            dfs(node.left)
            dfs(node.right)
            dct[pre] -= 1
            return

        dct = defaultdict(int)
        dct[0] = 1
        pre = 0
        ans = 0
        dfs(root)
        return ans
