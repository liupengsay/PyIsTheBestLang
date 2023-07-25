import unittest
from collections import defaultdict, Counter
from functools import lru_cache
from functools import reduce
from operator import xor, or_
from typing import List

from src.fast_io import FastIO

"""

算法：位运算相关技巧（也叫bitmasks）
功能：进行二进制上的位操作，包括与、异或、或、取反，通常使用按位思考与举例的方式寻找规律
题目：
异或经典性质：(4*i)^(4*i+1)^(4*i+2)^(4*i+3)=0
异或经典性质：(a&b)^(a&c) = a&(b^c)
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
2546. 执行逐位运算使字符串相等（https://leetcode.cn/problems/apply-bitwise-operations-to-make-strings-equal/）按照异或特点脑筋急转弯
1486. 数组异或操作（https://leetcode.cn/problems/xor-operation-in-an-array/）经典异或公式计算
1734. 解码异或后的排列（https://leetcode.cn/problems/decode-xored-permutation/）经典变换公式，解码相邻异或值编码，并利用奇数排列的异或性质
1787. 使所有区间的异或结果为零（https://leetcode.cn/problems/make-the-xor-of-all-segments-equal-to-zero/）经典按照异或特性分组并利用值域枚举DP
1835. 所有数对按位与结果的异或和（https://leetcode.cn/problems/find-xor-sum-of-all-pairs-bitwise-and/）按位操作模拟
1611. 使整数变为 0 的最少操作次数（https://leetcode.cn/problems/minimum-one-bit-operations-to-make-integers-zero/）格雷码的操作，直接计算格雷码对应的二进制数字

===================================洛谷===================================
P5657 格雷码（https://www.luogu.com.cn/problem/P5657）计算编号为 k 的二进制符，并补前缀 0 为 n 位
P6102 [EER2]谔运算（https://www.luogu.com.cn/problem/P6102）经典位运算加和题目，按位计算，按照位0与位1的数量进行讨论
P7442 「EZEC-7」维护序列（https://www.luogu.com.cn/problem/P7442）观察操作规律，使用位运算模拟操作
P7617 [COCI2011-2012#2] KOMPIĆI（https://www.luogu.com.cn/problem/P7617）使用位运算枚举
P7627 [COCI2011-2012#1] X3（https://www.luogu.com.cn/problem/P7627）经典按位操作枚举计算个数
P7649 [BalticOI 2004 Day 1] SCALES（https://www.luogu.com.cn/problem/P7649）三进制计算，贪心模拟砝码放置
P1582 倒水（https://www.luogu.com.cn/problem/P1582）进制题脑经急转弯
P2114 [NOI2014] 起床困难综合症（https://www.luogu.com.cn/problem/P2114）按位操作计算模拟，贪心选取最大结果
P2326 AKN’s PPAP（https://www.luogu.com.cn/problem/P2326）按位模拟贪心选取与值最大的数值对，最大与值对
P4144 大河的序列（https://www.luogu.com.cn/problem/P4144）按位思考贪心脑筋急转弯
P4310 绝世好题（https://www.luogu.com.cn/problem/P4310）线性 DP 使用按位转移
P5390 [Cnoi2019]数学作业（https://www.luogu.com.cn/problem/P5390）按位操作
P6824 「EZEC-4」可乐（https://www.luogu.com.cn/problem/P6824）经典按位操作计算异或不等式在使用差分作用域计数
P8842 [传智杯 #4 初赛] 小卡与质数2（https://www.luogu.com.cn/problem/P8842）经典质数个数前缀和与异或不等式区间计数
P8965 坠梦 | Falling into Dream（https://www.luogu.com.cn/problem/P8965）树形 DP 与异或计算

================================CodeForces================================
https://codeforces.com/problemset/problem/305/C（利用二进制加减的思想进行解题）
https://codeforces.com/problemset/problem/878/A（位运算的操作理解）
http://codeforces.com/problemset/problem/282/C（利用位运算的特性进行判断）
C. Mikasa（https://codeforces.com/problemset/problem/1554/C）经典位运算操作贪心计算
F. Dasha and Nightmares（https://codeforces.com/contest/1800/problem/F）位运算枚举计数
D. Little Girl and Maximum XOR（https://codeforces.com/problemset/problem/276/D）范围[l,r]区间的最大异或和
G. Orray（https://codeforces.com/contest/1742/problem/G）重排数组使得前缀或值的字典序最大

================================AcWing===================================
998. 起床困难综合症（https://www.acwing.com/problem/content/1000/）按位进行或、异或、与操作后贪心选取最大值

参考：OI WiKi（xx）
https://blog.csdn.net/qq_35473473/article/details/106320878
"""


class BitOperation:
    def __init__(self):
        return

    @staticmethod
    def sum_xor(n):
        # 模板：计算 [0, x] 区间证书的异或和
        if n % 4 == 0:
            return n
        # (4*i)^(4*i+1)^(4*i+2)^(4*i+3)=0
        elif n % 4 == 1:
            return 1  # n^(n-1)
        elif n % 4 == 2:
            return n + 1  # n^(n-1)^(n-2)
        return 0  # n^(n-1)^(n-2)^(n-3)

    @staticmethod
    def sum_xor_2(n):
        # 模板：计算 [0, x] 区间证书的异或和
        if n % 4 == 0:
            return n
        # (4*i)^(4*i+1)^(4*i+2)^(4*i+3)=0
        elif n % 4 == 1:
            return n ^ (n - 1)
        elif n % 4 == 2:
            return n ^ (n - 1) ^ (n - 2)
        return n ^ (n - 1) ^ (n - 2) ^ (n - 3)

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
    def cf_1742g(ac=FastIO()):

        # 模板：重排数组使得前缀或值的字典序最大
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            total = reduce(or_, nums)
            ind = set(list(range(n)))
            rest = nums[:]
            ans = []
            pre = 0
            while pre != total:
                low = -1
                x = 0
                for i in ind:
                    if rest[i] > low:
                        low = rest[i]
                        x = i
                pre |= nums[x]
                ans.append(nums[x])
                ind.discard(x)
                for i in ind:
                    rest[i] = (rest[i] ^ (rest[i] & pre))
            for i in ind:
                ans.append(nums[i])
            ac.lst(ans)
        return

    @staticmethod
    def cf_276d(ac=FastIO()):

        # 模板：区间[l,r]的最大异或和
        a, b = ac.read_ints()
        n = len(bin(b)) - 2
        ans = 0
        for i in range(n - 1, -1, -1):
            if b - a >= (1 << i):
                ans |= (1 << i)
            else:
                if (a ^ b) & (1 << i):
                    ans |= (1 << i)
        ac.st(ans)
        return

    @staticmethod
    def cf_1800f(ac=FastIO()):

        # 模板：位运算枚举计数
        n = ac.read_int()
        strings = [ac.read_str() for _ in range(n)]
        states = []
        for s in strings:
            cnt = Counter(s)
            a = b = 0
            for i in range(26):
                x = chr(i + ord("a"))
                if cnt[x] % 2:
                    a |= (1 << i)
                if cnt[x]:
                    b |= (1 << i)
            states.append([a, b])

        ans = 0
        for i in range(26):
            pre = defaultdict(int)
            target = ((1 << 26) - 1) ^ (1 << i)
            for j in range(n):
                if not states[j][1] & (1 << i):
                    ans += pre[target ^ states[j][0]]
                    pre[states[j][0]] += 1
        ac.st(ans)
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
            for j in range(max(i - ceil + 1, 0), i + 1):
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

    @staticmethod
    def ac_998(ac=FastIO()):
        # 模板：按照二进制每个位进行操作，计算贪心结果
        n, m = ac.read_ints()
        ans = [[0, 1 << i] for i in range(32)]
        for _ in range(n):
            op, t = ac.read_list_strs()
            t = int(t)
            if op == "AND":
                for i in range(32):
                    ans[i][0] &= t & (1 << i)
                    ans[i][1] &= t & (1 << i)
            elif op == "OR":
                for i in range(32):
                    ans[i][0] |= t & (1 << i)
                    ans[i][1] |= t & (1 << i)
            else:
                for i in range(32):
                    ans[i][0] ^= t & (1 << i)
                    ans[i][1] ^= t & (1 << i)
        res = x = 0
        for i in range(31, -1, -1):
            if ans[i][1] > ans[i][0] and (x | (1 << i)) <= m:
                x |= (1 << i)
                res += ans[i][1]
            else:
                res += ans[i][0]
        ac.st(res)
        return

    @staticmethod
    def lg_p1582(ac=FastIO()):
        # 模板：进制题脑筋急转弯
        n, k = ac.read_ints()
        ans = 0
        # 每次选末尾的 1 进行增加合并
        while bin(n).count("1") > k:
            ans += n & (-n)
            n += n & (-n)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2114(ac=FastIO()):
        # 模板：按位操作计算模拟，贪心选取最大结果
        n, m = ac.read_ints()
        one = [1 << i for i in range(32)]
        zero = [0] * 32
        for _ in range(n):
            op, t = ac.read_list_strs()
            t = int(t)
            for i in range(32):
                if op == "AND":
                    one[i] &= (t & (1 << i))
                    zero[i] &= (t & (1 << i))
                elif op == "OR":
                    one[i] |= (t & (1 << i))
                    zero[i] |= (t & (1 << i))
                else:
                    one[i] ^= (t & (1 << i))
                    zero[i] ^= (t & (1 << i))
        # 倒序选取最大值
        ans = 0
        for i in range(31, -1, -1):
            if one[i] > zero[i] and m >= (1 << i):
                m -= (1 << i)
                ans += one[i]
            else:
                ans += zero[i]
        ac.st(ans)
        return

    @staticmethod
    def lg_p2326(ac=FastIO()):
        # 模板：按位模拟贪心选取与值最大的数值对
        for case in range(ac.read_int()):
            ac.read_int()
            nums = ac.read_list_ints()
            ans = 0
            while nums:
                cnt = [0] * 32
                for num in nums:
                    for i in range(21):
                        if num & (1 << i):
                            cnt[i] += 1
                for i in range(20, -1, -1):
                    if cnt[i] >= 2:
                        ans |= (1 << i)
                        nums = [
                            num ^ (
                                1 << i) for num in nums if num & (
                                1 << i) and num ^ (
                                1 << i)]
                        break
                else:
                    nums = []
            ac.st(f"Case #{case + 1}: {ans}")
        return

    @staticmethod
    def lg_p4144(ac=FastIO()):
        # 模板：按位思考脑筋急转弯贪心
        n, b, p = ac.read_ints()
        nums = ac.read_list_ints()
        ans = max(nums) * 2
        ac.st(pow(ans + 233, b, p))
        return

    @staticmethod
    def lg_p4310(ac=FastIO()):
        # 模板：线性 DP 使用按位转移
        ac.read_int()
        nums = ac.read_list_ints()
        cnt = [0] * 32
        for num in nums:
            # 根据按位与与的特点
            pre = 0
            lst = []
            for j in range(32):
                if num & (1 << j):
                    lst.append(j)
                    pre = ac.max(pre, cnt[j])
            pre += 1
            for j in lst:
                cnt[j] = pre
        ac.st(max(cnt))
        return

    @staticmethod
    def lg_p5390(ac=FastIO()):
        # 模板：按位操作统计枚举
        mod = 998244353
        for _ in range(ac.read_int()):
            nums = ac.read_list_ints()
            n = nums[0]
            num = reduce(or_, nums[1:])
            pp = pow(2, n - 1, mod)
            ans = 0
            for i in range(32):
                if num & (1 << i):
                    ans += (1 << i) * pp
            ac.st(ans % mod)
        return

    @staticmethod
    def lg_p6824(ac=FastIO()):
        # 模板：经典按位操作计算异或不等式在使用差分作用域计数
        n, k = ac.read_ints()
        nums = [ac.read_int() for _ in range(n)]
        m = len(bin(max(k, max(nums))))
        diff = [0] * ((1 << m) + 1)
        for a in nums:
            i, pre = m - 1, 0
            while True:
                if i == -1:
                    diff[pre] += 1
                    diff[pre + 1] -= 1
                    break
                if k & (1 << i):
                    if a & (1 << i):
                        low = pre ^ (1 << i)
                        high = low ^ ((1 << i) - 1)
                        diff[low] += 1
                        diff[high + 1] -= 1
                        i -= 1
                    else:
                        low, high = pre, pre ^ ((1 << i) - 1)
                        diff[low] += 1
                        diff[high + 1] -= 1
                        i, pre = i - 1, pre ^ (1 << i)
                else:
                    if a & (1 << i):
                        i, pre = i - 1, pre ^ (1 << i)
                    else:
                        i -= 1
        ac.st(max(ac.accumulate(diff)))
        return

    @staticmethod
    def lg_p8842(ac=FastIO()):
        # 模板：经典质数个数前缀和与异或不等式区间计数（也可考虑 01 Trie）
        n = 1 << 21
        prime = [0] * (n + 1)
        prime[0] = 0
        prime[1] = 1
        for i in range(2, n + 1):
            if not prime[i]:
                for j in range(i * i, n + 1, i):
                    prime[j] = 1
            prime[i] += prime[i - 1]
        for _ in range(ac.read_int()):
            x = ac.read_int()
            ans = 0
            for k in range(21):
                if x & (1 << k):
                    ans += (1 << (k + 1)) - (1 << k) - \
                        (prime[(1 << (k + 1)) - 1] - prime[(1 << k) - 1])
            ac.st(ans)
        return

    @staticmethod
    def lc_1486(n: int, start: int) -> int:
        # 模板：经典异或公式计算
        s = start // 2
        bo = BitOperation()
        e = n & start & 1
        # (start+0)^(start+2)^..^(start+2*n-2)
        return (bo.sum_xor(s - 1) ^ bo.sum_xor(s + n - 1)) * 2 + e

    @staticmethod
    def lc_1734(encoded: List[int]) -> List[int]:
        # 模板：经典变换公式，解码相邻异或值编码，并利用奇数排列的异或性质
        n = len(encoded) + 1
        total = 1 if n % 4 == 1 else 0  # n=4*k+1 与 n=4*k+3
        odd = reduce(xor, encoded[1::2])
        ans = [total ^ odd]
        for num in encoded:
            ans.append(ans[-1] ^ num)
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

        pre = 0
        for i in range(100000):
            pre ^= i
            assert bo.sum_xor(i) == pre
            assert bo.sum_xor_2(i) == pre
        return


if __name__ == '__main__':
    unittest.main()