import unittest
from collections import defaultdict
from functools import lru_cache
from functools import reduce
from operator import xor
from typing import List

from algorithm.src.fast_io import FastIO

"""

算法：位运算相关技巧（也叫bitmasks）
功能：进行二进制上的位操作，包括与、异或、或、取反，通常使用按位思考与举例的方式寻找规律
题目：

===================================力扣===================================
2354. 优质数对的数目（https://leetcode.cn/problems/number-of-excellent-pairs/）需要脑筋急转弯确定位 1 的规律进行哈希计数枚举即可
260. 只出现一次的数字 III（https://leetcode.cn/problems/single-number-iii/）利用位运算两个相同元素异或和为0的特点，以及lowbit进行分组确定两个只出现一次的元素
6365. 将整数减少到零需要的最少操作数（https://leetcode.cn/problems/minimum-operations-to-reduce-an-integer-to-0/）n 加上或减去 2 的某个幂使得 n 变为 0 的最少操作数
6360. 最小无法得到的或值（https://leetcode.cn/problems/minimum-impossible-or/）利用贪心思想，类似硬币凑金额推理得出结论
2564. 子字符串异或查询（https://leetcode.cn/problems/substring-xor-queries/）利用二进制字符串无前置0时长度不超过10的特点进行查询
1238. 循环码排列（https://leetcode.cn/problems/circular-permutation-in-binary-representation/）生成格雷码，使得循环数组相邻数字二进制位只有一位不同
89. 格雷编码（https://leetcode.cn/problems/gray-code/）生成 0 开头的 n 位格雷码序列
137. 只出现一次的数字 II（https://leetcode.cn/problems/single-number-ii/）使用位运算按位计数
剑指 Offer 56 - I. 数组中数字出现的次数（https://leetcode.cn/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/）使用位运算按位计数
260. 只出现一次的数字 III（https://leetcode.cn/problems/single-number-iii/）使用位运算按位计数

===================================洛谷===================================
P5657 格雷码（https://www.luogu.com.cn/problem/P5657）计算编号为 k 的二进制符，并补前缀 0 为 n 位
P6102 [EER2]谔运算（https://www.luogu.com.cn/problem/P6102）经典位运算加和题目，按位计算，按照位0与位1的数量进行讨论
P7442 「EZEC-7」维护序列（https://www.luogu.com.cn/problem/P7442）观察操作规律，使用位运算模拟操作
P7617 [COCI2011-2012#2] KOMPIĆI（https://www.luogu.com.cn/problem/P7617）使用位运算枚举
P7627 [COCI2011-2012#1] X3（https://www.luogu.com.cn/problem/P7627）经典按位操作枚举计算个数
P7649 [BalticOI 2004 Day 1] SCALES（https://www.luogu.com.cn/problem/P7649）三进制计算，贪心模拟砝码放置

================================CodeForces================================
https://codeforces.com/problemset/problem/305/C（利用二进制加减的思想进行解题）
https://codeforces.com/problemset/problem/878/A（位运算的操作理解）
http://codeforces.com/problemset/problem/282/C（利用位运算的特性进行判断）
C. Mikasa（https://codeforces.com/problemset/problem/1554/C）经典位运算操作贪心计算

参考：OI WiKi（xx）
https://blog.csdn.net/qq_35473473/article/details/106320878
"""


class BitOperation:
    def __init__(self):
        return

    @staticmethod
    def graycode_to_integer(graycode):
        # 格雷码转二进制
        graycode_len = len(graycode)
        binary = list()
        binary.append(graycode[0])
        for i in range(1, graycode_len):
            if graycode[i] == binary[i - 1]:
                b = 0
            else:
                b = 1
            binary.append(str(b))
        return int("0b" + ''.join(binary), 2)

    @staticmethod
    def integer_to_graycode(integer):
        # 二进制转格雷码
        binary = bin(integer).replace('0b', '')
        graycode = list()
        binay_len = len(binary)
        graycode.append(binary[0])
        for i in range(1, binay_len):
            if binary[i - 1] == binary[i]:
                g = 0
            else:
                g = 1
            graycode.append(str(g))
        return ''.join(graycode)

    @staticmethod
    def get_graycode(n):
        # n位数格雷码
        code = [0, 1]
        for i in range(1, n):
            code.extend([(1 << i) + num for num in code[::-1]])
        return code


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_260(nums: List[int]) -> List[int]:

        # 模板：将整数换算成二进制进行计数
        s = reduce(xor, nums)
        last = s & (-s)
        one = two = 0
        for num in nums:
            if num & last:
                one ^= num
            else:
                two ^= num
        return [one, two]

    @staticmethod
    def lc_137(nums: List[int]) -> int:

        # 模板：将整数换算成二进制进行计数
        floor = (1 << 31) + 1
        dp = [0] * 33
        for num in nums:
            num += floor
            for i in range(33):
                if num & (1 << i):
                    dp[i] += 1
        ans = 0
        for i in range(33):
            if dp[i] % 3:
                ans |= (1 << i)
        return ans - floor

    @staticmethod
    def cf_1554c(ac=FastIO()):
        # 模板：涉及到 MEX 转换为求 n^ans>=m+1 的最小值ans
        for _ in range(ac.read_int()):
            n, m = ac.read_ints()
            assert 0 <= n <= 10**9
            assert 0 <= m <= 10**9
            p = m + 1
            ans = 0
            for i in range(30, -1, -1):
                if ans ^ n >= p:
                    break
                if n & (1 << i) == p & (1 << i):
                    continue
                if p & (1 << i):
                    ans |= (1 << i)
            ac.st(ans)
        return

    @staticmethod
    def lc_6360(nums):
        # 模板：最小的无法由子数组的或运算得到的数（异或则可以使用线性基求解判断）
        dct = set(nums)
        ans = 1
        while ans in dct:
            ans *= 2
        return ans

    @staticmethod
    def lc_6365(num):
        # 模板：n 加上或减去 2 的某个幂使得 n 变为 0 的最少操作数
        @lru_cache(None)
        def dfs(n):
            if not n:
                return 0
            if bin(n).count("1") == 1:
                return 1
            lowbit = n & (-n)
            return 1 + min(dfs(n - lowbit), dfs(n + lowbit))

        def greedy(n):
            # 对应有 O(logn) 贪心解法
            s = bin(n)[2:][::-1]
            ans = cnt = 0
            m = len(s)
            for i in range(m):
                if s[i] == "1":
                    cnt += 1
                else:
                    # 中心思想是连续的 111 可以通过加 1 变成 1000 再减去其中的 1 即操作两次
                    if cnt == 1:
                        ans += 1
                        cnt = 0
                    elif cnt >= 2:
                        if i + 1 < m and s[i + 1] == "1":
                            ans += 1
                            cnt = 1
                        else:
                            ans += 2
                            cnt = 0
            if cnt:
                ans += 1 if cnt == 1 else 2
            return ans

        # 更优解法 bin(n ^ (3 * n)).count("1")
        return dfs(num)

    @staticmethod
    def lc_260(nums):
        # 模板：找出数组当中两个只出现一次的数（其余数保证出现两次）
        s = reduce(xor, nums)
        low = s & (-s)
        ans = [0, 0]
        for num in nums:
            if num & low:
                ans[0] ^= num
            else:
                ans[1] ^= num
        return ans

    @staticmethod
    def lc_2564(s, queries):
        # 预处理相同异或值的索引
        dct = defaultdict(set)
        m = len(queries)
        for i in range(m):
            a, b = queries[i]
            x = bin(a ^ b)[2:]
            dct[x].add(i)
        ceil = max(len(x) for x in dct)
        ans = [[-1, -1] for _ in range(m)]
        # 遍历往前回溯查找个数
        n = len(s)
        for i in range(n):
            for j in range(max(i - ceil + 1, 0), i+1):
                st = s[j:i + 1]
                if dct[st]:
                    for k in dct[st]:
                        ans[k] = [j, i]
                    dct[st] = set()
        return ans

    @staticmethod
    def lc_1238(n: int, start: int) -> List[int]:
        # 模板：生成 n 位数的格雷码
        ans = BitOperation().get_graycode(n)
        i = ans.index(start)
        return ans[i:] + ans[:i]

    @staticmethod
    def lc_89(n: int) -> List[int]:
        # 模板：生成 n 位数的格雷码
        ans = BitOperation().get_graycode(n)
        return ans


class TestGeneral(unittest.TestCase):

    def test_bit_operation(self):
        bo = BitOperation()

        lst = [bo.integer_to_graycode(i) for i in range(11)]
        print(lst)

        assert bo.integer_to_graycode(0) == "0"
        assert bo.integer_to_graycode(22) == "11101"
        assert bo.graycode_to_integer("10110") == 27

        n = 8
        code = bo.get_graycode(n)
        m = len(code)
        for i in range(m):
            assert bo.graycode_to_integer(bin(code[i])[2:]) == i
            assert bo.integer_to_graycode(i) == bin(code[i])[2:]
        return


if __name__ == '__main__':
    unittest.main()
