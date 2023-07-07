
import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
from functools import reduce
from operator import xor
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy
import time
from algorithm.src.fast_io import FastIO, inf
import sys
sys.setrecursionlimit(10000000)

"""
算法：BST二叉搜索树
功能：根据数字顺序建立二叉搜索树、实时维护
题目：


===================================力扣===================================
1569. 将子数组重新排序得到同一个二叉搜索树的方案数（https://leetcode.cn/problems/number-of-ways-to-reorder-array-to-get-same-bst/）逆序思维，倒序利用并查集建立二叉搜索树，排列组合加并查集
1902. 给定二叉搜索树的插入顺序求深度（https://leetcode.cn/problems/depth-of-bst-given-insertion-order/）按照顺序建立二叉树求深度

===================================洛谷===================================
P2171 Hz吐泡泡（https://www.luogu.com.cn/problem/P2171）依次输入数据生成二叉搜索树，可使用逆序并查集

参考：OI WiKi（xx）
"""


class UnionFindSpecial:
    def __init__(self, n: int) -> None:
        self.root = [i for i in range(n)]
        return

    def find(self, x):
        lst = []
        while x != self.root[x]:
            lst.append(x)
            # 在查询的时候合并到顺带直接根节点
            x = self.root[x]
        for w in lst:
            self.root[w] = x
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x < root_y:
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        return


class BinarySearchTreeByArray:
    # 模板：根据数组生成二叉树的有向图结构
    def __init__(self):
        return

    @staticmethod
    def build(nums: List[int]):
        n = len(nums)
        ind = list(range(n))
        ind.sort(key=lambda it: nums[it])  # 索引
        rank = {idx: i for i, idx in enumerate(ind)}  # 排序

        dct = [[] for _ in range(n)]  # 二叉树按照索引的有向图结构
        uf = UnionFindSpecial(n)
        post = {}
        for i in range(n - 1, -1, -1):
            x = rank[i]
            if x + 1 in post:
                r = uf.find(post[x + 1])
                dct[i].append(r)
                uf.union(i, r)
            if x - 1 in post:
                r = uf.find(post[x - 1])
                dct[i].append(r)
                uf.union(i, r)
            post[x] = i
        return dct

    @staticmethod
    def build_with_stack(nums: List[int]):
        # 模板：按顺序生成二叉树，返回二叉树的索引父子信息
        n = len(nums)
        # 先按照大小关系编码成 1..n
        lst = sorted(nums)
        dct = {num: i + 1 for i, num in enumerate(lst)}
        ind = {num: i for i, num in enumerate(nums)}
        order = [dct[i] for i in nums]
        father, occur, stack = [0] * (n + 1), [0] * (n + 1), []
        deep = [0] * (n + 1)
        for i, x in enumerate(order, 1):
            occur[x] = i
        # 记录原数组索引的父子关系
        for x, i in enumerate(occur):
            while stack and occur[stack[-1]] > i:
                if occur[father[stack[-1]]] < i:
                    father[stack[-1]] = x
                stack.pop()
            if stack:
                father[x] = stack[-1]
            stack.append(x)

        for x in order:
            deep[x] = 1 + deep[father[x]]

        dct = [[] for _ in range(n)]
        for i in range(1, n + 1):
            if father[i]:
                u, v = father[i]-1, i-1
                x, y = ind[lst[u]], ind[lst[v]]
                dct[x].append(y)

        return dct


ans = 1  # 标记是第几层
ans1 = 1  # 最高层数


class BST(object):
    def __init__(self, data, left=None, right=None):  # BST的三个，值，左子树，右子树
        # 这里遇到链条形状的树会有性能问题（在线）
        super(BST, self).__init__()
        self.data = data
        self.left = left
        self.right = right

    def insert(self, val):  # 插入函数
        global ans, ans1
        if val < self.data:  # 如果小于就放到左子树
            if self.left:  # 如果有左子树
                ans += 1  # 层数+1
                self.left.insert(val)  # 左子树调用递归
            else:  # 没有左子树
                ans += 1  # 层数+1
                self.left = BST(val)  # 把值放在这个点的左子树上
                if ans > ans1:  # 如果层数比之前最高层数高
                    ans1 = ans  # 替换
                ans = 1  # 重新开始
        else:  # 比节点的值大
            if self.right:  # 有右子树
                ans += 1  # 层数+1
                self.right.insert(val)  # 右子树调用递归
            else:  # 没有右子树
                ans += 1  # 层数+1
                self.right = BST(val)  # 将值放在右子树上
                if ans > ans1:  # 如果层数大于最高层数
                    ans1 = ans  # 覆盖
                ans = 1  # 重新开始
        return

    def post_order(self):  # 后序遍历：左，右，根
        if self.left:  # 有左子树
            self.left.post_order()  # 先遍历它的左子树
        if self.right:  # 有右子树
            self.right.post_order()  # 遍历右子树
        # print(self.data)  # 都没有或已经遍历完就输出值
        return


class Node:
    # Constructor assigns the given key, with left and right
    # children assigned with None.
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class BinarySearchTree:
    # Constructor just assigns an empty root.
    def __init__(self):
        self.root = None

    # Search for a node containing a matching key. Returns the
    # Node object that has the matching key if found, None if
    # not found.
    def search(self, desired_key):
        current_node = self.root
        while current_node is not None:
            # Return the node if the key matches.
            if current_node.key == desired_key:
                return current_node

            # Navigate to the left if the search key is
            # less than the node's key.
            elif desired_key < current_node.key:
                current_node = current_node.left

            # Navigate to the right if the search key is
            # greater than the node's key.
            else:
                current_node = current_node.right

        # The key was not found in the tree.
        return None

    # Inserts the new node into the tree.
    def insert(self, node):

        # Check if the tree is empty
        if self.root is None:
            self.root = node
        else:
            current_node = self.root
            while current_node is not None:
                if node.key < current_node.key:
                    # If there is no left child, add the new
                    # node here; otherwise repeat from the
                    # left child.
                    if current_node.left is None:
                        current_node.left = node
                        current_node = None
                    else:
                        current_node = current_node.left
                else:
                    # If there is no right child, add the new
                    # node here; otherwise repeat from the
                    # right child.
                    if current_node.right is None:
                        current_node.right = node
                        current_node = None
                    else:
                        current_node = current_node.right

                        # Removes the node with the matching key from the tree.

    def remove(self, key):
        parent = None
        current_node = self.root

        # Search for the node.
        while current_node is not None:

            # Check if current_node has a matching key.
            if current_node.key == key:
                if current_node.left is None and current_node.right is None:  # Case 1
                    if parent is None:  # Node is root
                        self.root = None
                    elif parent.left is current_node:
                        parent.left = None
                    else:
                        parent.right = None
                    return  # Node found and removed
                elif current_node.left is not None and current_node.right is None:  # Case 2
                    if parent is None:  # Node is root
                        self.root = current_node.left
                    elif parent.left is current_node:
                        parent.left = current_node.left
                    else:
                        parent.right = current_node.left
                    return  # Node found and removed
                elif current_node.left is None and current_node.right is not None:  # Case 2
                    if parent is None:  # Node is root
                        self.root = current_node.right
                    elif parent.left is current_node:
                        parent.left = current_node.right
                    else:
                        parent.right = current_node.right
                    return  # Node found and removed
                else:  # Case 3
                    # Find successor (leftmost child of right subtree)
                    successor = current_node.right
                    while successor.left is not None:
                        successor = successor.left
                    current_node.key = successor.key  # Copy successor to current node
                    parent = current_node
                    current_node = current_node.right  # Remove successor from right subtree
                    key = parent.key  # Loop continues with new key
            elif current_node.key < key:  # Search right
                parent = current_node
                current_node = current_node.right
            else:  # Search left
                parent = current_node
                current_node = current_node.left

        return  # Node not found


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p2171_1(ac=FastIO()):
        # 模板: bst 标准插入 O(n^2)
        n = ac.read_int()
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
        dct = BinarySearchTreeByArray().build(nums)  # 或者是 build_with_stack
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
                    stack.append([j, d+1])
            else:
                i = ~i
                depth = depth if depth > d else d
                ans.append(nums[i])
        ac.st(f"deep={depth}")
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lc_1902(order: List[int]) -> int:
        # 模板：按照顺序建立二叉树求深度
        dct = BinarySearchTreeByArray().build(order)
        stack = [[0, 1]]
        ans = 1
        while stack:
            i, d = stack.pop()
            for j in dct[i]:
                stack.append([j, d+1])
                ans = ans if ans > d + 1 else d + 1
        return ans


class TestGeneral(unittest.TestCase):

    @staticmethod
    def lg_2171_1_input(n, nums):
        # 模板: bst 标准插入 O(n^2)（在线）
        bst = BST(nums[0])
        for num in nums[1:]:
            bst.insert(num)
        bst.post_order()  # 实现是这里要将 print 从内部注释恢复打印
        return

    @staticmethod
    def lg_2171_2_input(n, nums, ac=FastIO):
        # 模板: bst 链表与二叉树模拟插入 O(nlogn)（离线）

        @ac.bootstrap
        def dfs(rt):
            if ls[rt]:
                yield dfs(ls[rt])
            if rs[rt]:
                yield dfs(rs[rt])
            yield

        m = n + 10
        # 排序后离散化
        a = [0] + nums
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
        dfs(root)
        return

    @staticmethod
    def lg_2171_3_input(n, nums):
        dct = BinarySearchTreeByArray().build(nums)
        # 使用迭代的方式计算后序遍历（离线）
        ans = []
        depth = 0
        stack = [[0, 1]]
        while stack:
            i, d = stack.pop()
            if i >= 0:
                stack.append([~i, d])
                dct[i].sort(key=lambda it: -nums[it])
                for j in dct[i]:
                    stack.append([j, d+1])
            else:
                i = ~i
                depth = depth if depth > d else d
                ans.append(nums[i])
        return

    @staticmethod
    def lg_2171_3_input_2(n, nums):
        dct = BinarySearchTreeByArray().build_with_stack(nums)
        # 使用迭代的方式计算后序遍历（离线）
        ans = []
        depth = 0
        stack = [[0, 1]]
        while stack:
            i, d = stack.pop()
            if i >= 0:
                stack.append([~i, d])
                dct[i].sort(key=lambda it: -nums[it])
                for j in dct[i]:
                    stack.append([j, d+1])
            else:
                i = ~i
                depth = depth if depth > d else d
                ans.append(nums[i])
        return

    def test_solution(self):
        n = 1000000
        nums = [random.randint(1, n) for _ in range(n)]
        nums = list(set(nums))
        n = len(nums)
        random.shuffle(nums)
        t1 = time.time()
        self.lg_2171_1_input(n, nums[:])
        t2 = time.time()
        self.lg_2171_2_input(n, nums[:])
        t3 = time.time()
        self.lg_2171_3_input(n, nums[:])
        t4 = time.time()
        self.lg_2171_3_input_2(n, nums[:])
        t5 = time.time()
        print(n, t2-t1, t3 - t2, t4 - t3, t5-t4)

        t1 = time.time()
        nums = list(range(1, n+1))
        t2 = time.time()
        self.lg_2171_2_input(n, nums[:])
        t3 = time.time()
        self.lg_2171_3_input(n, nums[:])
        t4 = time.time()
        self.lg_2171_3_input_2(n, nums[:])
        t5 = time.time()
        print(n, t2-t1, t3 - t2, t4 - t3, t5-t4)

        return


if __name__ == '__main__':
    unittest.main()
