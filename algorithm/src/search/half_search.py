import unittest
import bisect

from algorithm.src.fast_io import FastIO

"""
算法：折半搜索、meet in middle
功能：常见于 1<<n 较大的情况，对半分开进行枚举 
题目：

===================================洛谷===================================
P5194 [USACO05DEC]Scales S（https://www.luogu.com.cn/problem/P5194）利用Fibonacci数列的长度特点进行折半搜索枚举，与二分查找确定可行的最大值
Anya and Cubes（https://www.luogu.com.cn/problem/CF525E）折半搜索计算长度
P5691 [NOI2001] 方程的解数（https://www.luogu.com.cn/problem/P5691）折半搜索与枚举

===================================AcWing======================================
171. 送礼物（https://www.acwing.com/problem/content/173/）经典折半搜索查找最接近目标值的子数组和

参考：OI WiKi（xx）
"""


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p5194(ac=FastIO()):
        # 模板：折半搜索枚举后使用二分寻找最接近目标值的数
        n, c = ac.read_ints()
        val = [ac.read_int() for _ in range(n)]

        def check(lst):
            s = len(lst)
            pre = 0
            res = set()

            def dfs(i):
                nonlocal pre
                if pre > c:
                    return
                if i == s:
                    res.add(pre)
                    return
                pre += lst[i]
                dfs(i + 1)
                pre -= lst[i]
                dfs(i + 1)
                return

            dfs(0)
            return sorted(list(res))

        res1 = check(val[:n // 2])
        res2 = check(val[n // 2:])
        ans = max(max(res1), max(res2))
        for num in res2:
            i = bisect.bisect_right(res1, c - num) - 1
            if i >= 0:
                ans = ans if ans > num + res1[i] else num + res1[i]
        ac.st(ans)
        return

    @staticmethod
    def ac_171(ac=FastIO()):
        # 模板：经典折半搜索查找最接近目标值的子数组和

        w, n = ac.read_ints()
        lst = [ac.read_int() for _ in range(n)]
        lst.sort()

        def check(tmp):
            m = len(tmp)
            cur = set()
            stack = [[0, 0]]
            # 使用迭代方式枚举
            while stack:
                x, i = stack.pop()
                if x > w:
                    continue
                if i == m:
                    cur.add(x)
                    continue
                stack.append([x+tmp[i], i+1])
                stack.append([x, i+1])
            return cur

        pre = sorted(list(check(lst[:n//2])))
        post = sorted(list(check(lst[n//2:])))
        if len(pre) > len(post):
            pre, post = post, pre
        ans = 0
        for num in pre:
            j = bisect.bisect_right(post, w-num)-1
            ans = max(ans, num)
            if 0 <= j < len(post):
                ans = max(ans, num+post[j])
        ac.st(ans)
        return

    @staticmethod
    def lg_p5691(ac=FastIO()):
        # 模板：折半搜索与枚举
        n = ac.read_int()
        m = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        ans = 0
        if n == 1:
            for x1 in range(1, m + 1):
                if nums[0][0] * x1 ** nums[0][1] == 0:
                    ans += 1
            ac.st(ans)
            return
        if n == 2:
            for x1 in range(1, m + 1):
                for x2 in range(1, m + 1):
                    if nums[0][0] * x1 ** nums[0][1] + nums[1][0] * x2 ** nums[1][1] == 0:
                        ans += 1
            ac.st(ans)
            return
        if n == 3:
            for x1 in range(1, m + 1):
                for x2 in range(1, m + 1):
                    for x3 in range(1, m + 1):
                        cur = nums[0][0] * x1 ** nums[0][1] + nums[1][0] * x2 ** nums[1][1] + nums[2][0] * x3 ** \
                              nums[2][1]
                        if cur == 0:
                            ans += 1
            ac.st(ans)
            return

        # 枚举前半部分
        dct = dict()
        for x1 in range(1, m + 1):
            for x2 in range(1, m + 1):
                for x3 in range(1, m + 1):
                    cur = nums[0][0] * x1 ** nums[0][1] + nums[1][0] * x2 ** nums[1][1] + nums[2][0] * x3 ** nums[2][1]
                    dct[cur] = dct.get(cur, 0) + 1

        # 计算后半部分
        nums = nums[3:]
        n = len(nums)
        if n == 1:
            for x1 in range(1, m + 1):
                cur = nums[0][0] * x1 ** nums[0][1]
                ans += dct.get(-cur, 0)
            ac.st(ans)
            return
        if n == 2:
            for x1 in range(1, m + 1):
                for x2 in range(1, m + 1):
                    cur = nums[0][0] * x1 ** nums[0][1] + nums[1][0] * x2 ** nums[1][1]
                    ans += dct.get(-cur, 0)
            ac.st(ans)
            return
        if n == 3:
            for x1 in range(1, m + 1):
                for x2 in range(1, m + 1):
                    for x3 in range(1, m + 1):
                        cur = nums[0][0] * x1 ** nums[0][1] + nums[1][0] * x2 ** nums[1][1] + nums[2][0] * x3 ** \
                              nums[2][1]
                        ans += dct.get(-cur, 0)
            ac.st(ans)
            return
        return


class TestGeneral(unittest.TestCase):

    def test_xxx(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
