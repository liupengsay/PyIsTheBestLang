"""

Algorithm：bit_operation相关技巧（也叫bitmasks）
Description：二进制上的位操作，包括与、异或、或、取反，通常按位思考与举例的方式寻找规律
异或性质：(4*i)^(4*i+1)^(4*i+2)^(4*i+3)=0  (2*n)^(2*n+1)=1
异或性质：(a&b)^(a&c) = a&(b^c)
====================================LeetCode====================================

2354（https://leetcode.com/problems/number-of-excellent-pairs/）需要brain_teaser确定位 1 的规律hashcounterbrute_force即可
260（https://leetcode.com/problems/single-number-iii/）利用bit_operation两个相同元素异或和为0的特点，以及lowbit分组确定两个只出现一次的元素
6365（https://leetcode.com/problems/minimum-operations-to-reduce-an-integer-to-0/）n |上或减去 2 的某个幂使得 n 变为 0 的最少操作数
6360（https://leetcode.com/problems/minimum-impossible-or/）利用greedy思想，类似硬币凑金额推理得出结论
2564（https://leetcode.com/problems/substring-xor-queries/）利用二进制字符串无前置0时长度不超过10的特点查询
1238（https://leetcode.com/problems/circular-permutation-in-binary-representation/）生成格雷码，使得循环数组相邻数字二进制位只有一位不同
89（https://leetcode.com/problems/gray-code/）生成 0 开头的 n 位格雷码序列
137（https://leetcode.com/problems/single-number-ii/）bit_operation按位counter
剑指 Offer 56 - I（https://leetcode.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/）bit_operation按位counter
260（https://leetcode.com/problems/single-number-iii/）bit_operation按位counter
2546（https://leetcode.com/problems/apply-bitwise-operations-to-make-strings-equal/）按照异或特点brain_teaser
1486（https://leetcode.com/problems/xor-operation-in-an-array/）异或公式
1734（https://leetcode.com/problems/decode-xored-permutation/）变换公式，解码相邻异或值编码，并利用奇数排列的异或性质
1787（https://leetcode.com/problems/make-the-xor-of-all-segments-equal-to-zero/）按照异或特性分组并利用data_rangebrute_forceDP
1835（https://leetcode.com/problems/find-xor-sum-of-all-pairs-bitwise-and/）按位操作implemention
1611（https://leetcode.com/problems/minimum-one-bit-operations-to-make-integers-zero/）格雷码的操作，直接格雷码对应的二进制数字
2275（https://leetcode.com/problems/largest-combination-with-bitwise-and-greater-than-zero/）求按位与不为0的最长子序列，不要求连续
2527（https://leetcode.com/problems/find-xor-beauty-of-array/description/）按位brute_forcebrain_teaser
2680（https://leetcode.com/problems/maximum-or/description/）greedybrute_force，prefix_suffix或分解
100087（https://leetcode.com/problems/apply-operations-on-array-to-maximize-sum-of-squares/description/）按位greedy

=====================================LuoGu======================================
5657（https://www.luogu.com.cn/problem/P5657）编号为 k 的二进制符，并补前缀 0 为 n 位
6102（https://www.luogu.com.cn/problem/P6102）bit_operation|和题目，按位，按照位0与位1的数量讨论
7442（https://www.luogu.com.cn/problem/P7442）观察操作规律，bit_operationimplemention操作
7617（https://www.luogu.com.cn/problem/P7617）bit_operationbrute_force
7627（https://www.luogu.com.cn/problem/P7627）按位操作brute_force个数
7649（https://www.luogu.com.cn/problem/P7649）三进制，greedyimplemention砝码放置
1582（https://www.luogu.com.cn/problem/P1582）进制题脑经急转弯
2114（https://www.luogu.com.cn/problem/P2114）按位操作implemention，greedy选取最大结果
2326（https://www.luogu.com.cn/problem/P2326）按位implementiongreedy选取与值最大的数值对，最大与值对
4144（https://www.luogu.com.cn/problem/P4144）按位思考greedybrain_teaser
4310（https://www.luogu.com.cn/problem/P4310）linear_dp 按位转移
5390（https://www.luogu.com.cn/problem/P5390）按位操作
6824（https://www.luogu.com.cn/problem/P6824）按位操作异或不等式在差分action_scopecounter
8842（https://www.luogu.com.cn/problem/P8842）质数个数prefix_sum与异或不等式区间counter
8965（https://www.luogu.com.cn/problem/P8965）树形 DP 与异或

===================================CodeForces===================================
305C（https://codeforces.com/problemset/problem/305/C）利用二进制|减的思想解题
878A（https://codeforces.com/problemset/problem/878/A）bit_operation的操作理解
C. XOR and OR（http://codeforces.com/problemset/problem/282/C）利用bit_operation的特性判断
1554C（https://codeforces.com/problemset/problem/1554/C）bit_operation操作greedy
1800F（https://codeforces.com/contest/1800/problem/F）bit_operationbrute_forcecounter
276D（https://codeforces.com/problemset/problem/276/D）范围[l,r]区间的最大异或和
1742G（https://codeforces.com/contest/1742/problem/G）重排数组使得前缀或值的lexicographical_order最大
1851F（https://codeforces.com/contest/1851/problem/F）数组的最小异或对，一定是sorting后相邻的数
1879D（https://codeforces.com/contest/1879/problem/D）按位贡献与prefix_sumcounter与前缀下标|和
1368D（https://codeforces.com/problemset/problem/1368/D）按位implementiongreedy
1802C（https://codeforces.com/contest/1802/problem/C）construction with the property of xor

====================================AtCoder=====================================
D - XXOR（https://atcoder.jp/contests/abc117/tasks/abc117_d）从高位到低位按位greedy，brain_teaser|
D - Xor Sum 4（https://atcoder.jp/contests/abc147/tasks/abc147_d）classical按位异或和贡献

=====================================AcWing=====================================
998（https://www.acwing.com/problem/content/1000/）按位或、异或、与操作后greedy选取最大值
4614（https://www.acwing.com/problem/content/4617/）bit_operationbrute_force与prefix_sum预处理


https://blog.csdn.net/qq_35473473/article/details/106320878
"""
from collections import defaultdict, Counter
from functools import lru_cache
from functools import reduce
from math import inf
from operator import xor, or_
from typing import List

from src.mathmatics.bit_operation.template import BitOperation
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1742g(ac=FastIO()):

        # 重排数组使得前缀或值的lexicographical_order最大
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

        # 区间[l,r]的最大异或和
        a, b = ac.read_list_ints()
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

        # bit_operationbrute_forcecounter
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

        # 将整数换算成二进制counter
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

        # 将整数换算成二进制counter
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
        # 涉及到 MEX 转换为求 n^ans>=m+1 的最小值ans
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            assert 0 <= n <= 10 ** 9
            assert 0 <= m <= 10 ** 9
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
    def lc_1787(nums: List[int], k: int) -> int:
        # 按照异或特性分组并利用data_rangebrute_forceDP
        m = max(len(bin(num)) - 2 for num in nums)
        pre = [inf] * (1 << m)
        pre[0] = 0
        for i in range(k):
            lst = nums[i::k]
            n = len(lst)
            cnt = Counter(lst)
            low = min(pre)
            cur = [low + n for x in pre]
            for j in range(1 << m):
                for num in cnt:
                    a, b = cur[j], pre[j ^ num] + n - cnt[num]
                    cur[j] = a if a < b else b
            pre = cur[:]
        return pre[0]

    @staticmethod
    def lc_6360(nums):
        # 最小的无法由子数组的或运算得到的数（异或则可以线性基求解判断）
        dct = set(nums)
        ans = 1
        while ans in dct:
            ans *= 2
        return ans

    @staticmethod
    def lc_6365(num):
        # n |上或减去 2 的某个幂使得 n 变为 0 的最少操作数
        @lru_cache(None)
        def dfs(n):
            if not n:
                return 0
            if bin(n).count("1") == 1:
                return 1
            low = n & (-n)
            return 1 + min(dfs(n - low), dfs(n + low))

        # 更优解法 bin(n ^ (3 * n)).count("1")
        return dfs(num)

    @staticmethod
    def lc_6365_2(num):
        # 对应有 O(logn) greedy解法
        s = bin(num)[2:][::-1]
        ans = cnt = 0
        m = len(s)
        for i in range(m):
            if s[i] == "1":
                cnt += 1
            else:
                # 中心思想是连续的 111 可以通过| 1 变成 1000 再减去其中的 1 即操作两次
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

    @staticmethod
    def lc_2275(candidates: List[int]) -> int:
        # 求按位与不为0的最长子序列，不要求连续
        count = [0] * 32
        for num in candidates:
            st = bin(num)[2:]
            n = len(st)
            for i in range(1, n + 1, 1):  # 也可要求连续的情况
                if st[-i] == '1':
                    count[i] += 1
        return max(count)

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
        # 遍历往前back_track查找个数
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
        # 生成 n 位数的格雷码
        ans = BitOperation().get_graycode(n)
        i = ans.index(start)
        return ans[i:] + ans[:i]

    @staticmethod
    def lc_89(n: int) -> List[int]:
        # 生成 n 位数的格雷码
        ans = BitOperation().get_graycode(n)
        return ans

    @staticmethod
    def abc_117d(ac=FastIO()):
        # 从高位到低位按位greedy，brain_teaser|
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        ans = pre = 0
        for i in range(40, -1, -1):
            cnt = Counter(int(num & (1 << i) > 0) for num in nums)
            if cnt[1] >= cnt[0] or pre + (1 << i) > k:
                ans += cnt[1] * (1 << i)  # 由于多于一半因此必然最优
            else:
                pre += (1 << i)
                ans += cnt[0] * (1 << i)
        ac.st(ans)
        return

    @staticmethod
    def abc_121d(ac=FastIO()):
        def count(x):
            if x <= 0:
                return 0
            m = (x + 1) // 2
            ans = m % 2
            if (x + 1) % 2:
                ans ^= x
            return ans

        a, b = ac.read_list_ints()
        ac.st(count(b) ^ count(a - 1))
        return

    @staticmethod
    def ac_998(ac=FastIO()):
        # 按照二进制每个位操作，greedy结果
        n, m = ac.read_list_ints()
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
        # 进制题brain_teaser
        n, k = ac.read_list_ints()
        ans = 0
        # 每次选末尾的 1 增|合并
        while bin(n).count("1") > k:
            ans += n & (-n)
            n += n & (-n)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2114(ac=FastIO()):
        # 按位操作implemention，greedy选取最大结果
        n, m = ac.read_list_ints()
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
        # reverse_order|选取最大值
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
        # 按位implementiongreedy选取与值最大的数值对
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
        # 按位思考brain_teasergreedy
        n, b, p = ac.read_list_ints()
        nums = ac.read_list_ints()
        ans = max(nums) * 2
        ac.st(pow(ans + 233, b, p))
        return

    @staticmethod
    def lg_p4310(ac=FastIO()):
        # linear_dp 按位转移
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
        # 按位操作统计brute_force
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
        # 按位操作异或不等式在差分action_scopecounter
        n, k = ac.read_list_ints()
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
        # 质数个数prefix_sum与异或不等式区间counter（也可考虑 01 Trie）
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
        # 异或公式
        s = start // 2
        bo = BitOperation()
        e = n & start & 1
        # (start+0)^(start+2)^..^(start+2*n-2)
        return (bo.sum_xor(s - 1) ^ bo.sum_xor(s + n - 1)) * 2 + e

    @staticmethod
    def lc_1734(encoded: List[int]) -> List[int]:
        # 变换公式，解码相邻异或值编码，并利用奇数排列的异或性质
        n = len(encoded) + 1
        total = 1 if n % 4 == 1 else 0  # n=4*k+1 与 n=4*k+3
        odd = reduce(xor, encoded[1::2])
        ans = [total ^ odd]
        for num in encoded:
            ans.append(ans[-1] ^ num)
        return ans

    @staticmethod
    def ac_4614(ac=FastIO()):
        # bit_operationbrute_force与prefix_sum预处理
        n, m, q = ac.read_list_ints()
        nums = ac.read_list_ints()
        lst = [ac.read_str() for _ in range(m)]
        cnt = Counter(lst)
        dct = [0] * (1 << n)
        for va in cnt:
            dct[int("0b" + va, 2)] = cnt[va]

        weight = [0] * (1 << n)
        for i in range(1 << n):
            weight[i] = sum(nums[j] for j in range(n) if not i & (1 << (n - 1 - j)))

        res = [[0] * 101 for _ in range(1 << n)]
        for i in range(1 << n):
            for j in range(1 << n):
                s = weight[i ^ j]
                if s <= 100:
                    res[i][s] += dct[j]

        for i in range(1 << n):
            for j in range(1, 101):
                res[i][j] += res[i][j - 1]

        for _ in range(q):
            t, k = ac.read_list_strs()
            k = int(k)

            ac.st(res[int("0b" + t, 2)][k])
        return