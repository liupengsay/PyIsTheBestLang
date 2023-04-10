import unittest
from typing import List

from algorithm.src.fast_io import FastIO
from collections import deque

"""

算法：构造
功能：通常需要根据题目要求，进行贪心排序之后的构造出合理可行的答案
题目：xx（xx）

===================================力扣===================================
280. 摆动排序（https://leetcode.cn/problems/wiggle-sort/）按照降序先赋给奇数索引再给偶数索引

===================================洛谷===================================
P8846 『JROI-7』PMK 配匹串符字（https://www.luogu.com.cn/problem/P8846）厘清题意进行贪心构造
P2902 [USACO08MAR]Pearl Pairing G（https://www.luogu.com.cn/problem/P2902）构造不同颜色对
P5823 【L&K R-03】课表的排列（https://www.luogu.com.cn/problem/P5823）构造相同元素间隔为等差1的数组
P7383 「EZEC-6」加减（https://www.luogu.com.cn/problem/P7383）贪心构造

================================CodeForces================================
https://codeforces.com/problemset/problem/1396/A（贪心构造）
https://codeforces.com/problemset/problem/1118/E（模拟贪心构造）
https://codeforces.com/problemset/problem/960/C（贪心构造）
B. Fedya and Array（https://codeforces.com/contest/1793/problem/B）思维题贪心构造
D. Replace by MEX（https://codeforces.com/problemset/problem/1375/D）经典MEX的构造题，使得数组非降，即最小的未出现在数组中的数
D. Phoenix and Science（https://codeforces.com/problemset/problem/1348/D）利用二进制优化将数分割，并利用分裂特点进行计算
D. Diane（https://codeforces.com/problemset/problem/1554/D）构造，将n取一半向下取整
C. Matching Numbers（https://codeforces.com/problemset/problem/1788/C）构造，选取 1 到 2n 的数两两配对且和为连续数
D. Task On The Board（https://codeforces.com/problemset/problem/1367/D）逆向思维模拟构造
D. Multiples and Power Differences（https://codeforces.com/problemset/problem/1485/D）按照数据范围进行构造
G. Even-Odd XOR（https://codeforces.com/problemset/problem/1722/G）根据奇偶性和异或和特点进行构造

==================================AtCoder=================================
https://atcoder.jp/contests/agc007/tasks/agc007_b（脑筋急转弯思维构造公式）


参考：OI WiKi（xx）
"""


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1367d(ac=FastIO()):
        # 模板：逆向思维进行生成模拟构造
        for _ in range(ac.read_int()):
            s = ac.read_str()
            m = ac.read_int()
            nums = ac.read_list_ints()
            ans = [""] * m
            lst = deque(sorted(list(s), reverse=True))
            while max(nums) >= 0:
                zero = [i for i in range(m) if nums[i] == 0]
                k = len(zero)
                while len(set(list(lst)[:k])) != 1:
                    lst.popleft()
                for i in zero:
                    nums[i] = -1
                    ans[i] = lst.popleft()
                while lst and lst[0] == ans[zero[0]]:
                    lst.popleft()
                for i in range(m):
                    if nums[i] != -1:
                        nums[i] -= sum(abs(i - j) for j in zero)
            ac.st("".join(ans))
        return

    @staticmethod
    def cf_1788c(ac=FastIO()):
        # 模板：选取 1 到 2n 的数两两配对且和为连续数
        for _ in range(ac.read_int()):
            n = ac.read_int()
            if n % 2:
                ac.st("YES")
                x = n // 2
                for i in range(1, n + 1):
                    if i <= x:
                        ac.lst([i, i + n + x + 1])
                    else:
                        ac.lst([i, i - x + n])
            else:
                ac.st("NO")
        return

    @staticmethod
    def lc_280(nums: List[int]) -> None:
        # 模板：摆动排序数组构造
        nums.sort()
        n = len(nums)
        ans = [0]*n
        j = n-1
        for i in range(1, n, 2):
            ans[i] = nums[j]
            j -= 1
        j = 0
        for i in range(0, n, 2):
            ans[i] = nums[j]
            j += 1
        for i in range(n):
            nums[i] = ans[i]
        return


class TestGeneral(unittest.TestCase):

    def test_construction(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
