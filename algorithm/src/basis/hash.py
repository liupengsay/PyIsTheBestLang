import unittest
from collections import defaultdict
from typing import List

from algorithm.src.fast_io import FastIO

"""
算法：哈希
功能：前后缀计数、索引、加和
题目：

===================================力扣===================================
2143. 在两个数组的区间中选取数字（https://leetcode.cn/problems/choose-numbers-from-two-arrays-in-range/）前缀和哈希计数转换为求连续子数组和为 0 的个数
面试题 17.05.  字母与数字（https://leetcode.cn/problems/find-longest-subarray-lcci/）和为0的最长连续子数组，使用前缀和哈希求解
1590. 使数组和能被 P 整除（https://leetcode.cn/problems/make-sum-divisible-by-p/）求和模p与整个子数组模p相等的最短子数组，使用前缀和哈希求解
6317. 统计美丽子数组数目（https://leetcode.cn/contest/weekly-contest-336/problems/count-the-number-of-beautiful-subarrays/）前缀和哈希计数
题目-02. 销售出色区间（https://leetcode.cn/contest/hhrc2022/problems/0Wx4Pc/）前缀和哈希，加脑筋急转弯贪心
题目-03. 重复的彩灯树（https://leetcode.cn/contest/hhrc2022/problems/VAc7h3/）二叉树序列化

===================================洛谷===================================
P2697 宝石串（https://www.luogu.com.cn/problem/P2697）哈希记录前缀和与对应索引
P1114 “非常男女”计划（https://www.luogu.com.cn/record/list?user=739032&status=12&page=13）哈希记录前缀和与对应索引


137. 雪花雪花雪花（https://www.acwing.com/problem/content/139/）哈希找相同雪花

参考：OI WiKi（xx）
"""

class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_2143(nums1: List[int], nums2: List[int]) -> int:
        # 模板：经典使用哈希计数模拟线性 DP 转移
        n = len(nums1)
        mod = 10**9 + 7
        pre = defaultdict(int)
        pre[-nums1[0]] += 1
        pre[nums2[0]] += 1
        ans = pre[0]
        for i in range(1, n):
            cur = defaultdict(int)
            cur[-nums1[i]] += 1
            cur[nums2[i]] += 1
            for p in pre:
                cur[p - nums1[i]] += pre[p]
                cur[p + nums2[i]] += pre[p]
            ans += cur[0]
            ans %= mod
            pre = cur
        return ans

    @staticmethod
    def ac_137(ac=FastIO()):


        p1 = random.randint(26, 100)
        p2 = random.randint(26, 100)
        mod1 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)
        mod2 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)

        def compute(ls):
            res1 = 0
            for num in ls:
                res1 *= p1
                res1 += num
                res1 %= mod1
            res2 = 0
            for num in ls:
                res2 *= p2
                res2 += num
                res2 %= mod2
            return (res1, res2)

        def check():
            res = []
            for ii in range(6):
                cu = tuple(lst[ii:]+lst[:ii])
                res.append(compute(cu))
                cu = tuple(lst[:ii+1][::-1]+lst[ii+1:][::-1])
                res.append(compute(cu))
            return res

        n = ac.read_int()
        pre = set()
        ans = False
        for _ in range(n):
            if ans:
                break
            lst = ac.read_list_ints()
            now = check()
            if any(cur in pre for cur in now):
                ans = True
                break
            for cur in now:
                pre.add(cur)

        if ans:
            ac.st("Twin snowflakes found.")
        else:
            ac.st("No two snowflakes are alike.")
        return


class HashMap:
    def __init__(self):
        return

    def gen_result(self):
        return


class TestGeneral(unittest.TestCase):

    def test_hash_map(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
