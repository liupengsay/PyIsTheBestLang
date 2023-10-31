import random
import time
import unittest

from src.graph.bst.template import BST, BinarySearchTreeByArray
from src.utils.fast_io import FastIO


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
        dct = BinarySearchTreeByArray().build_with_unionfind(nums)
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

        nums = list(range(1, n+1))
        t2 = time.time()
        self.lg_2171_2_input(n, nums[:])
        t3 = time.time()
        self.lg_2171_3_input(n, nums[:])
        t4 = time.time()
        self.lg_2171_3_input_2(n, nums[:])
        t5 = time.time()
        print(n, t3 - t2, t4 - t3, t5-t4)

        return


if __name__ == '__main__':
    unittest.main()
