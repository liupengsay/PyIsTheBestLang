import unittest
from collections import defaultdict, Counter
from itertools import accumulate
from operator import xor

from src.fast_io import FastIO

"""
算法：分块查询、双指针
功能：将查询区间进行分块排序，交替移动双指针进行动态维护查询值
题目：

===================================力扣===================================

===================================洛谷===================================

================================CodeForces================================
B. Little Elephant and Array（https://codeforces.com/contest/220/problem/B）分块矩阵计数模板题
D. Powerful array（https://codeforces.com/contest/86/problem/D）分块矩阵求函数值模板题
E. XOR and Favorite Number（https://codeforces.com/contest/617/problem/E）分块矩阵求异或对计数模板题

参考：OI WiKi（https://oi-wiki.org/ds/fenwick/）
"""


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_220b(ac=FastIO()):
        # 模板：查询区间内符合条件的元素个数
        n, m = ac.read_ints()
        nums = ac.read_list_ints()
        size = int(n ** 0.5) + 1  # 分块的大小

        queries = [[] for _ in range(size)]
        # 将查询分段
        for i in range(m):
            a, b = ac.read_list_ints_minus_one()
            queries[b // size].append([a, b, i])

        # 更新区间信息
        def update(num, p):
            nonlocal cur, cnt
            if num == cnt[num]:
                cur -= 1
            cnt[num] += p
            if num == cnt[num]:
                cur += 1
            return

        cur = 0
        ans = [0] * m
        x = y = 0
        cnt = defaultdict(int)
        cnt[nums[0]] = 1
        if nums[0] == 1:
            cur += 1
        for i in range(size):
            # 按照分块后单独排序
            if i % 2:
                queries[i].sort(key=lambda it: -it[0])
            else:
                queries[i].sort(key=lambda it: it[0])
            # 移动双指针
            for a, b, j in queries[i]:
                while y > b:
                    update(nums[y], -1)
                    y -= 1
                while y < b:
                    y += 1
                    update(nums[y], 1)
                while x > a:
                    x -= 1
                    update(nums[x], 1)
                while x < a:
                    update(nums[x], -1)
                    x += 1
                ans[j] = cur
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def cf_86d(ac=FastIO()):
        # 模板：查询区间内的函数值
        n, t = ac.read_ints()
        nums = ac.read_list_ints()
        size = int(n**0.5) + 1

        queries = [[] for _ in range(t)]
        # 将查询分段
        for i in range(t):
            a, b = ac.read_list_ints_minus_one()
            queries[b // size].append([a, b, i])

        def update(p, z):
            nonlocal cur
            num = nums[p]
            cur -= cnt[num] * cnt[num] * num
            cnt[num] += z
            cur += cnt[num] * cnt[num] * num
            return

        cnt = [0] * (10**6 + 1)
        x = y = 0
        ans = [0] * t
        cnt = Counter()
        cur = nums[0]
        cnt[nums[0]] = 1
        for i in range(size):
            # 按照分块后单独排序
            if i % 2:
                queries[i].sort(key=lambda it: -it[0])
            else:
                queries[i].sort(key=lambda it: it[0])

            for a, b, j in queries[i]:
                while y > b:
                    update(y, -1)
                    y -= 1
                while y < b:
                    y += 1
                    update(y, 1)

                while x > a:
                    x -= 1
                    update(x, 1)
                while x < a:
                    update(x, -1)
                    x += 1
                ans[j] = cur
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def cf_617e(ac=FastIO()):
        # 模板：查询区间内的异或对数
        n, m, k = ac.read_ints()
        nums = ac.read_list_ints()
        pre = list(accumulate(nums, xor, initial=0))

        size = int(n ** 0.5) + 1  # 分块的大小
        queries = [[] for _ in range(size)]
        # 将查询分段
        for i in range(m):
            a, b = ac.read_list_ints()
            queries[b // size].append([a, b, i])

        def update(num, p):
            nonlocal cur
            if p == 1:
                cur += dct[num ^ k]
                dct[num] += 1
            else:
                dct[num] -= 1
                cur -= dct[num ^ k]
            return

        dct = [0]*(2*10**6+1)
        x = y = 0
        ans = [0] * m
        dct[pre[0]] += 1
        cur = 0
        for i in range(size):
            # 按照分块后单独排序
            if i % 2:
                queries[i].sort(key=lambda it: -it[0])
            else:
                queries[i].sort(key=lambda it: it[0])
            # 移动双指针
            for a, b, j in queries[i]:
                a -= 1
                while y > b:
                    update(pre[y], -1)
                    y -= 1
                while y < b:
                    y += 1
                    update(pre[y], 1)
                while x > a:
                    x -= 1
                    update(pre[x], 1)
                while x < a:
                    update(pre[x], -1)
                    x += 1
                ans[j] = cur
        for a in ans:
            ac.st(a)
        return


class TestGeneral(unittest.TestCase):

    def test_tree_array_range_sum(self):
        pass


if __name__ == '__main__':
    unittest.main()
