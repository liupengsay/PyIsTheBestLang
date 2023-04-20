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


class TestGeneral(unittest.TestCase):

    def test_xxx(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
