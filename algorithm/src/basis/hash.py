import unittest
from collections import defaultdict, Counter
import random
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
P4889 kls与flag（https://www.luogu.com.cn/problem/P4889）经典公式变换使用哈希计数
P6273 [eJOI2017] 魔法（https://www.luogu.com.cn/problem/P6273）经典哈希前缀计数

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
            return res1, res2

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

    @staticmethod
    def lg_p4889(ac=FastIO()):
        # 模板：枚举计数
        n, m = ac.read_ints()
        height = ac.read_list_ints()
        cnt = defaultdict(int)
        ans = 0
        for i in range(n):
            # hj - hi = j - i
            ans += cnt[height[i] - i]
            cnt[height[i] - i] += 1

        cnt = defaultdict(int)
        for i in range(n):
            # hi - hj = j - i
            ans += cnt[height[i] + i]
            # hj + hi = j - i
            ans += cnt[i - height[i]]
            cnt[height[i] + i] += 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p6273(ac=FastIO()):
        # 模板：经典哈希前缀计数
        ac.read_int()
        s = ac.read_str()
        # 选择最少出现的字符作为减数
        ct = Counter(s)
        st = list(ct.keys())
        ind = {w: i for i, w in enumerate(st)}
        m = len(ind)
        x = 0
        for i in range(1, m):
            if ct[st[i]] < ct[st[x]]:
                x = i
        # 记录状态
        cnt = [0] * m
        pre = defaultdict(int)
        pre[tuple(cnt)] = 1
        ans = 0
        mod = 10**9 + 7
        for w in s:
            if w == st[x]:
                # 其余所有字符减 1
                for i in range(m):
                    if i != ind[w]:
                        cnt[i] -= 1
            else:
                # 减数字符加 1
                cnt[ind[w]] += 1
            tp = tuple(cnt)
            # sa-ta = sb-tb 则有 sa-sb = ta-tb 因此这样计数
            ans += pre[tp]
            pre[tp] += 1
            ans %= mod
        ac.st(ans)
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
