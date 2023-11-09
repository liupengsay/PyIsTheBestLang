"""
算法：数论、欧拉筛、线性筛、素数、欧拉函数、因子分解、素因子分解、进制转换、因数分解
功能：有时候数位DP类型题目可以使用N进制来求取，质因数分解、因数分解、素数筛、线性筛、欧拉函数、pollard_rho、Meissel–Lehmer 算法（计算范围内素数个数）
题目：

===================================力扣===================================
264. 丑数 II（https://leetcode.cn/problems/ugly-number-ii/）只含2、3、5质因数的第 n 个丑数
1201. 丑数 III（https://leetcode.cn/problems/ugly-number-iii/）只含特定因子数即能被其中一个数整除的第 n 个丑数
313. 超级丑数（https://leetcode.cn/problems/super-ugly-number/）只含某些特定质因数的第 n 个丑数
12. 整数转罗马数字（https://leetcode.cn/problems/integer-to-roman/）整数转罗马数字
13. 罗马数字转整数（https://leetcode.cn/problems/roman-to-integer/）罗马数字转整数
264. 丑数 II（https://leetcode.cn/problems/ugly-number-ii/）只含2、3、5质因数的第 n 个丑数
1201. 丑数 III（https://leetcode.cn/problems/ugly-number-iii/）只含特定因子数即能被其中一个数整除的第 n 个丑数
313. 超级丑数（https://leetcode.cn/problems/super-ugly-number/）只含某些特定质因数的第 n 个丑数
6364. 无平方子集计数（https://leetcode.cn/problems/count-the-number-of-square-free-subsets/）非空子集乘积不含除 1 之外任何平方整除数，即乘积质数因子的幂次均为 1（背包DP计数）
1994. 好子集的数目（https://leetcode.cn/problems/the-number-of-good-subsets/）非空子集乘积不含除 1 之外任何平方整除数，即乘积质数因子的幂次均为 1（背包DP计数）
6309. 分割数组使乘积互质（https://leetcode.cn/contest/weekly-contest-335/problems/split-the-array-to-make-coprime-products/）计算 1 到 n 的每个数所有的质因子，并使用差分进行影响因子计数
2464. 有效分割中的最少子数组数目（https://leetcode.cn/problems/minimum-subarrays-in-a-valid-split/）计算 1 到 n 的每个数所有的质因子，并使用动态规划计数
LCP 14. 切分数组（https://leetcode.cn/problems/qie-fen-shu-zu/）计算 1 到 n 的每个数所有的质因子，并使用动态规划计数
279. 完全平方数（https://leetcode.cn/problems/perfect-squares/）四平方数定理
650. 只有两个键的键盘（https://leetcode.cn/problems/2-keys-keyboard/）经典分解质因数
1735. 生成乘积数组的方案数（https://leetcode.cn/problems/count-ways-to-make-array-with-product/）经典质数分解与隔板法应用
1390. 四因数（https://leetcode.cn/contest/weekly-contest-181/problems/four-divisors/）预处理所有数的所有因子
1819. 序列中不同最大公约数的数目（https://leetcode.cn/problems/number-of-different-subsequences-gcds/）预处理所有整数的所有因子，再枚举gcd计算
1017. 负二进制转换（https://leetcode.cn/contest/weekly-contest-130/problems/convert-to-base-2/）负进制转换模板题
1073. 负二进制数相加（https://leetcode.cn/problems/adding-two-negabinary-numbers/）经典负进制计算题
8041. 完全子集的最大元素和（https://leetcode.cn/problems/maximum-element-sum-of-a-complete-subset-of-indices/description/）经典质因数分解，奇数幂次的质因子组合哈希
2183. 统计可以被 K 整除的下标对数目（https://leetcode.cn/problems/count-array-pairs-divisible-by-k/description/）可以使用所有因子遍历枚举计数解决，正解为按照 k 的最大公因数分组

===================================洛谷===================================
P1865 A % B Problem（https://www.luogu.com.cn/problem/P1865）通过线性筛素数后进行二分查询区间素数个数
P1748 H数（https://www.luogu.com.cn/problem/P1748）丑数可以使用堆模拟可以使用指针递增也可以使用容斥原理与二分进行计算
P2723 [USACO3.1]丑数 Humble Numbers（https://www.luogu.com.cn/problem/P2723）第n小的只含给定素因子的丑数
P1952 火星上的加法运算（https://www.luogu.com.cn/problem/P1952）N进制加法
P1555 尴尬的数字（https://www.luogu.com.cn/problem/P1555）二进制与三进制
P1465 [USACO2.2]序言页码 Preface Numbering（https://www.luogu.com.cn/problem/P1465）整数转罗马数字
P1112 波浪数（https://www.luogu.com.cn/problem/P1112）枚举波浪数计算其不同进制下是否满足条件
P2926 [USACO08DEC]Patting Heads S（https://www.luogu.com.cn/problem/P2926）素数筛或者因数分解计数统计可被数列其他数整除的个数
P5535 【XR-3】小道消息（https://www.luogu.com.cn/problem/P5535）素数is_prime5判断加贪心脑筋急转弯
P1876 开灯（https://www.luogu.com.cn/problem/P1876）经典好题，理解完全平方数的因子个数为奇数，其余为偶数
P1887 乘积最大3（https://www.luogu.com.cn/problem/P1887）在和一定的情况下，数组分散越平均，其乘积越大
P2043 质因子分解（https://www.luogu.com.cn/problem/P2043）使用素数筛法的思想，计算阶乘n!的质因子与对应的个数
P2192 HXY玩卡片（https://www.luogu.com.cn/problem/P2192）一个数能整除9当且仅当其数位和能整除9
P7191 [COCI2007-2008#6] GRANICA（https://www.luogu.com.cn/problem/P7191）取模公式变换，转换为计算最大公约数，与所有因数分解计算
P7517 [省选联考 2021 B 卷] 数对（https://www.luogu.com.cn/problem/P7517）利用埃氏筛的思想，从小到大，进行因数枚举计数
P7588 双重素数（2021 CoE-II A）（https://www.luogu.com.cn/problem/P7588）素数枚举计算，优先使用is_prime4
P7696 [COCI2009-2010#4] IKS（https://www.luogu.com.cn/problem/P7696）数组，每个数进行质因数分解，然后均匀分配质因子
P4718 【模板】Pollard's rho 算法（https://www.luogu.com.cn/problem/P4718）使用pollard_rho进行质因数分解与素数判断
P1865 A % B Problem（https://www.luogu.com.cn/problem/P1865）通过线性筛素数后进行二分查询区间素数个数
P1748 H数（https://www.luogu.com.cn/problem/P1748）丑数可以使用堆模拟可以使用指针递增也可以使用容斥原理与二分进行计算
P2723 [USACO3.1]丑数 Humble Numbers（https://www.luogu.com.cn/problem/P2723）第n小的只含给定素因子的丑数
P2429 制杖题（https://www.luogu.com.cn/problem/P2429）枚举质因数组合加容斥原理计数
P2926 [USACO08DEC]Patting Heads S（https://www.luogu.com.cn/problem/P2926）素数筛或者因数分解计数统计可被数列其他数整除的个数
P5535 【XR-3】小道消息（https://www.luogu.com.cn/problem/P5535）素数is_prime5判断加贪心脑筋急转弯
P1876 开灯（https://www.luogu.com.cn/problem/P1876）经典好题，理解完全平方数的因子个数为奇数，其余为偶数
P7588 双重素数（2021 CoE-II A）（https://www.luogu.com.cn/problem/P7588）素数枚举计算，优先使用is_prime4
P7696 [COCI2009-2010#4] IKS（https://www.luogu.com.cn/problem/P7696）数组，每个数进行质因数分解，然后均匀分配质因子
P4718 【模板】Pollard's rho 算法（https://www.luogu.com.cn/problem/P4718）使用pollard_rho进行质因数分解与素数判断
P1069 [NOIP2009 普及组] 细胞分裂（https://www.luogu.com.cn/problem/P1069）质因数分解，转换为因子计数翻倍整除
P1072 [NOIP2009 提高组] Hankson 的趣味题（https://www.luogu.com.cn/problem/P1072）枚举所有因数，需要计算所有因数
P1593 因子和（https://www.luogu.com.cn/problem/P1593）使用质因数分解与快速幂计算a^b的所有因子之和
P2527 [SHOI2001]Panda的烦恼（https://www.luogu.com.cn/problem/P2527）丑数即只含特定质因子的数
P2557 [AHOI2002]芝麻开门（https://www.luogu.com.cn/problem/P2557）使用质因数分解计算a^b的所有因子之和
P4446 [AHOI2018初中组]根式化简（https://www.luogu.com.cn/problem/P4446）预先处理出素数然后计算最大的完全立方数因子
P4752 Divided Prime（https://www.luogu.com.cn/problem/P4752）判断除数是否为质数
P5248 [LnOI2019SP]快速多项式变换(FPT)（https://www.luogu.com.cn/problem/P5248）经典进制题目
P5253 [JSOI2013]丢番图（https://www.luogu.com.cn/problem/P5253）经典方程变换计算 (x-n)*(y-n)=n^2 的对数
P7960 [NOIP2021] 报数（https://www.luogu.com.cn/problem/P7960）类似埃氏筛的思路进行预处理
P8319 『JROI-4』分数（https://www.luogu.com.cn/problem/P8319）质因数分解与因子计数
P8646 [蓝桥杯 2017 省 AB] 包子凑数（https://www.luogu.com.cn/problem/P8646）经典裴蜀定理与背包 DP
P8762 [蓝桥杯 2021 国 ABC] 123（https://www.luogu.com.cn/problem/P8762）容斥原理加前缀和计数
P8778 [蓝桥杯 2022 省 A] 数的拆分（https://www.luogu.com.cn/problem/P8778）经典枚举素因子后O(n^0.25)计算是否为完全平方数与立方数
P8782 [蓝桥杯 2022 省 B] X 进制减法（https://www.luogu.com.cn/problem/P8782）多种进制结合贪心计算，经典好题

================================CodeForces================================
C. Hossam and Trainees（https://codeforces.com/problemset/problem/1771/C）使用pollard_rho进行质因数分解
A. Enlarge GCD（https://codeforces.com/problemset/problem/1034/A）经典求 1 到 n 所有数字的质因子个数总和
C. Hossam and Trainees（https://codeforces.com/problemset/problem/1771/C）使用pollard_rho进行质因数分解
D. Two Divisors（https://codeforces.com/problemset/problem/1366/D）计算最小的质因子，使用构造判断是否符合条件
A. Orac and LCM（https://codeforces.com/contest/1349/problem/A）质因数分解，枚举最终结果当中质因子的幂次
D. Same GCDs（https://codeforces.com/problemset/problem/1295/D）利用最大公因数的特性转换为欧拉函数求解，即比 n 小且与 n 互质的数个数
D. Another Problem About Dividing Numbers（https://codeforces.com/problemset/problem/1538/D）使用pollard_rho进行质因数分解
A. Row GCD（https://codeforces.com/problemset/problem/1458/A）gcd公式变换求解
A. Division（https://codeforces.com/problemset/problem/1444/A）贪心枚举质数因子
C. Strongly Composite（https://codeforces.com/contest/1823/problem/C）质因数分解进行贪心计算
D. Recover it!（https://codeforces.com/contest/1176/problem/D）经典构造题，贪心模拟，记录合数最大不等于自身的因子，以及质数列表的顺序
D. Counting Rhyme（https://codeforces.com/contest/1884/problem/D）factor dp and cnt

================================AtCoder================================
D - 756（https://atcoder.jp/contests/abc114/tasks/abc114_d）质因数分解计数

================================AcWing================================
97. 约数之和（https://www.acwing.com/problem/content/99/）计算a^b的所有约数之和
124. 数的进制转换（https://www.acwing.com/problem/content/126/）不同进制的转换，注意0的处理
197. 阶乘分解（https://www.acwing.com/problem/content/199/）计算n!阶乘的质因数分解即因子与因子的个数
196. 质数距离（https://www.acwing.com/problem/content/198/）经典计算质数距离对
198. 反素数（https://www.acwing.com/problem/content/200/）经典计算最大的反质数（反素数，即约数或者说因数个数大于任何小于它的数的因数个数）
199. 余数之和（https://www.acwing.com/problem/content/description/201/）经典枚举因数计算之和
3727. 乘方相加（https://www.acwing.com/solution/content/54479/）脑筋急转弯转换成进制表达问题
3999. 最大公约数（https://www.acwing.com/problem/content/description/4002/）同CF1295D
4319. 合适数对（https://www.acwing.com/problem/content/4322/）质因数分解后前缀哈希计数
4484. 有限小数（https://www.acwing.com/problem/content/4487/）分数在某个进制下是否为有限小数问题
4486. 数字操作（https://www.acwing.com/problem/content/description/4489/）经典质数分解贪心题
4622. 整数拆分（https://www.acwing.com/problem/content/description/4625/）思维题贪心构造
5049. 选人（https://www.acwing.com/problem/content/description/5052/）使用质因数分解计算组合数


参考：OI WiKi（xx）
"""
import math
from collections import Counter
from collections import defaultdict
from functools import reduce
from itertools import permutations
from math import inf
from typing import List

from src.mathmatics.number_theory.template import NumberTheory
from src.mathmatics.prime_factor.template import PrimeFactor
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1034a(ac=FastIO()):

        n = ac.read_int()
        nums = ac.read_list_ints()
        ceil = max(nums)

        # 模板：快速计算 1~ceil 的质数因子数
        p = [0] * (ceil + 1)
        for i in range(2, ceil + 1):
            if p[i] == 0:
                p[i] = i
                # 从 i*i 开始作为 p[j] 的最小质数因子
                for j in range(i * i, ceil + 1, i):
                    p[j] = i

        # 计算gcd
        g = reduce(math.gcd, nums)
        cnt = [0] * (ceil + 1)
        for i in range(n):
            b = nums[i] // g
            while b > 1:
                # 计算 num[i] 除掉 g 以后的质数因子数
                fac = p[b]
                # 计数加 1 也可以记录由多少个因子
                cnt[fac] += 1
                while b % fac == 0:
                    b //= fac
        res = max(cnt)
        if res == 0:
            ac.st(-1)
        else:
            ac.st(n - res)
        return

    @staticmethod
    def lc_6334(nums: List[int]) -> int:
        # 模板：非空子集乘积不含除 1 之外任何平方整除数，即乘积质数因子的幂次均为 1（背包DP计数）
        dct = {2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30}
        # 集合为质数因子幂次均为 1
        mod = 10 ** 9 + 7
        cnt = Counter(nums)
        pre = defaultdict(int)
        for num in cnt:
            if num in dct:
                cur = pre.copy()
                for p in pre:
                    if math.gcd(p, num) == 1:
                        cur[p * num] += pre[p] * cnt[num]
                        cur[p * num] %= mod
                cur[num] += cnt[num]
                pre = cur.copy()
        # 1 需要特殊处理
        p = pow(2, cnt[1], mod)
        ans = sum(pre.values()) * p
        ans += p - 1
        return ans % mod

    @staticmethod
    def cf_1366d(ac=FastIO()):
        ac.read_int()
        nums = ac.read_list_ints()
        ceil = max(nums)

        # 模板：利用线性筛的思想计算最小的质因数
        min_div = [i for i in range(ceil + 1)]
        for i in range(2, len(min_div)):
            if min_div[i] != i:
                continue
            if i * i >= len(min_div):
                break
            for j in range(i, len(min_div)):
                if i * j >= len(min_div):
                    break
                if min_div[i * j] == i * j:
                    min_div[i * j] = i

        # 构造结果
        ans1 = []
        ans2 = []
        for num in nums:
            p = min_div[num]
            v = num
            while v % p == 0:
                v //= p
            if v == 1:
                # 只有一个质因子
                ans1.append(-1)
                ans2.append(-1)
            else:
                ans1.append(v)
                ans2.append(num // v)
        ac.lst(ans1)
        ac.lst(ans2)
        return

    @staticmethod
    def lc_2183(nums: List[int], k: int) -> int:
        # 模板：可以使用所有因子遍历枚举计数解决，正解为按照 k 的最大公因数分组
        nt = PrimeFactor(10 ** 5)
        ans = 0
        dct = defaultdict(int)
        for i, num in enumerate(nums):
            w = k // math.gcd(num, k)
            ans += dct[w]
            for w in nt.all_factor[num]:
                dct[w] += 1
        return ans

    @staticmethod
    def lc_2464(nums: List[int]) -> int:
        # 模板：计算 1 到 n 的数所有的质因子并使用动态规划计数
        nt = PrimeFactor(max(nums))
        ind = dict()
        n = len(nums)
        dp = [inf] * (n + 1)
        dp[0] = 0
        for i, num in enumerate(nums):
            while num > 1:
                p = nt.min_prime[num]
                while num % p == 0:
                    num //= p
                if p not in ind or dp[i] < dp[ind[p]]:
                    ind[p] = i
                if dp[ind[p]] + 1 < dp[i + 1]:
                    dp[i + 1] = dp[ind[p]] + 1
                if dp[i] + 1 < dp[i + 1]:
                    dp[i + 1] = dp[i] + 1
        return dp[-1] if dp[-1] < inf else -1

    @staticmethod
    def lc_8041(nums: List[int]) -> int:
        # 模板：经典预处理幂次为奇数的质因子哈希分组计数
        n = len(nums)
        nt = PrimeFactor(n)
        dct = defaultdict(int)
        for i in range(1, n + 1):
            cur = nt.prime_factor[i]
            cur = [p for p, c in cur if c % 2]
            dct[tuple(cur)] += nums[i - 1]
        return max(dct.values())

    @staticmethod
    def lc_lcp14(nums: List[int]) -> int:
        # 模板：计算 1 到 n 的数所有的质因子并使用动态规划计数
        nt = PrimeFactor(max(nums))
        ind = dict()
        n = len(nums)
        dp = [inf] * (n + 1)
        dp[0] = 0
        for i, num in enumerate(nums):
            while num > 1:
                p = nt.min_prime[num]
                while num % p == 0:
                    num //= p
                if p not in ind or dp[i] < dp[ind[p]]:
                    ind[p] = i
                if dp[ind[p]] + 1 < dp[i + 1]:
                    dp[i + 1] = dp[ind[p]] + 1
                if dp[i] + 1 < dp[i + 1]:
                    dp[i + 1] = dp[i] + 1
        return dp[-1] if dp[-1] < inf else -1

    @staticmethod
    def cf_1176d(ac=FastIO()):
        # 模板：经典构造题，贪心模拟，记录合数最大不等于自身的因子，以及质数列表的顺序
        ac.read_int()
        nt = PrimeFactor(2 * 10 ** 5)
        prime_numbers = NumberTheory().euler_flag_prime(3 * 10 ** 6)
        dct = {num: i + 1 for i, num in enumerate(prime_numbers)}
        nums = ac.read_list_ints()
        nums.sort(reverse=True)
        cnt = Counter(nums)
        ans = []
        for num in nums:
            if not cnt[num]:
                continue
            if num in dct:
                fa = dct[num]
                cnt[num] -= 1
                cnt[fa] -= 1
                ans.append(fa)
            else:
                cnt[num] -= 1
                x = nt.all_factor[num][-2]
                cnt[x] -= 1
                ans.append(num)
        ac.lst(ans)
        return

    @staticmethod
    def cf_1349a(ac=FastIO()):
        # 模板：质因数分解，枚举最终结果当中质因子的幂次
        n = ac.read_int()
        nums = ac.read_list_ints()
        nmp = PrimeFactor(max(nums))
        dct = defaultdict(list)

        for num in nums:
            for p, c in nmp.prime_factor[num]:
                dct[p].append(c)

        ans = 1
        for p in dct:
            if len(dct[p]) >= n - 1:
                dct[p].sort()
                ans *= p ** dct[p][-n + 1]
        ac.st(ans)
        return

    @staticmethod
    def cf_1458a(ac=FastIO()):
        # 模板：gcd公式变换求解gcd(x,y)=gcd(x-y,y)
        m, n = ac.read_list_ints()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        g = 0
        # 推广到n维
        for i in range(1, m):
            g = math.gcd(g, a[i] - a[i - 1])
        ans = [math.gcd(g, a[0] + num) for num in b]
        ac.lst(ans)
        return

    @staticmethod
    def abc_114d(ac=FastIO()):
        # 模板：质因数分解计数
        n = ac.read_int()
        nt = PrimeFactor(n + 10)
        cnt = Counter()
        for x in range(1, n + 1):
            for p, c in nt.prime_factor[x]:
                cnt[p] += c
        ans = set()
        for item in permutations(list(cnt.keys()), 3):
            x, y, z = item
            if cnt[x] >= 2 and cnt[y] >= 4 and cnt[z] >= 4:
                if y > z:
                    y, z = z, y
                ans.add((x, y, z))

        for item in permutations(list(cnt.keys()), 2):
            x, y = item
            if cnt[x] >= 2 and cnt[y] >= 24:
                ans.add((x, y, 325))
            if cnt[x] >= 4 and cnt[y] >= 14:
                ans.add((x, y, 515))
        for x in cnt:
            if cnt[x] >= 74:
                ans.add(x)
        ac.st(len(ans))
        return

    @staticmethod
    def ac_124(ac=FastIO()):
        # 模板：不同进制之间的转换
        st = "0123456789"
        for i in range(26):
            st += chr(i + ord("A"))
        for i in range(26):
            st += chr(i + ord("a"))
        ind = {w: i for i, w in enumerate(st)}
        for _ in range(ac.read_int()):
            a, b, word = ac.read_list_strs()
            a = int(a)
            b = int(b)
            num = 0
            for w in word:
                num *= a
                num += ind[w]
            ac.lst([a, word])
            ans = ""
            while num:
                ans += st[num % b]
                num //= b
            if not ans:
                ans = "0"
            ac.lst([b, ans[::-1]])
            ac.st("")
        return

    @staticmethod
    def ac_197(ac=FastIO()):
        # 模板：计算n!阶乘的质因数分解即因子与因子的个数
        ceil = ac.read_int()
        min_prime = [0] * (ceil + 1)
        # 模板：计算 1 到 ceil 所有数字的最小质数因子
        for i in range(2, ceil + 1):
            if not min_prime[i]:
                min_prime[i] = i
                for j in range(i * i, ceil + 1, i):
                    min_prime[j] = i

        # 模板：计算 1 到 ceil 所有数字的质数分解结果
        dct = defaultdict(int)
        for num in range(2, ceil + 1):
            while num > 1:
                p = min_prime[num]
                cnt = 0
                while num % p == 0:
                    num //= p
                    cnt += 1
                dct[p] += cnt
        for p in sorted(dct):
            ac.lst([p, dct[p]])
        return

    @staticmethod
    def ac_199(ac=FastIO()):
        # 模板：计算 sum(k%i for i in range(n))
        n, k = ac.read_list_ints()
        ans = n * k
        left = 1
        while left <= min(n, k):
            right = min(k // (k // left), n)
            ans -= (k // left) * (left + right) * (right - left + 1) // 2
            left = right + 1
        ac.st(ans)
        return

    @staticmethod
    def lc_p2429(ac=FastIO()):
        # 模板：枚举质因数组合加容斥原理计数
        n, m = ac.read_list_ints()
        primes = sorted(ac.read_list_ints())

        def dfs(i):
            nonlocal ans, value, cnt
            if value > m:
                return
            if i == n:
                if cnt:
                    num = m // value
                    ans += value * (num * (num + 1) // 2) * (-1) ** (cnt + 1)
                    ans %= mod
                return

            value *= primes[i]
            cnt += 1
            dfs(i + 1)
            cnt -= 1
            value //= primes[i]
            dfs(i + 1)
            return

        cnt = ans = 0
        value = 1
        mod = 376544743
        dfs(0)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2527(ac=FastIO()):
        # 模板：丑数即只含特定质因子的数
        n, k = ac.read_list_ints()
        primes = ac.read_list_ints()
        dp = [1] * (k + 1)
        pointer = [0] * n
        for i in range(k):
            num = min(dp[pointer[i]] * primes[i] for i in range(n))
            for x in range(n):
                if dp[pointer[x]] * primes[x] == num:
                    pointer[x] += 1
            dp[i + 1] = num
        ac.st(dp[-1])
        return

    @staticmethod
    def lg_p5248(ac=FastIO()):
        # 模板：经典进制题目
        m, fm = ac.read_list_ints()
        lst = []
        while fm:
            lst.append(fm % m)
            fm //= m
        ac.st(len(lst))
        ac.lst(lst)
        return

    @staticmethod
    def lg_p7960(ac=FastIO()):
        # 模板：类似埃氏筛的思路进行预处理
        n = 10 ** 7
        dp = [0] * (n + 1)
        for x in range(1, n + 1):
            if "7" in str(x):
                y = 1
                while x * y <= n:
                    dp[x * y] = 1
                    y += 1
        post = 10 ** 7 + 1
        for i in range(n, -1, -1):
            if dp[i] == 1:
                dp[i] = -1
            else:
                dp[i] = post
                post = i

        for _ in range(ac.read_int()):
            ac.st(dp[ac.read_int()])
        return

    @staticmethod
    def lg_p8319(ac=FastIO()):
        # 模板：质因数分解进行贪心计算
        n = 2 * 10 ** 6
        f = [1] * (n + 1)
        prime = [0] * (n + 1)
        for x in range(2, n + 1):
            if not prime[x]:
                # 计算当前值作为质因子的花费次数
                t = 1
                while t * x <= n:
                    c = 1
                    xx = t
                    while xx % x == 0:
                        xx //= x
                        c += 1
                    f[t * x] += (x - 1) * c
                    prime[t * x] = 1
                    t += 1

        # 进行前缀最大值计算处理
        for i in range(1, n + 1):
            f[i] = ac.max(f[i - 1], f[i])
        for _ in range(ac.read_int()):
            ac.st(f[ac.read_int()])
        return

    @staticmethod
    def lg_p8646(ac=FastIO()):
        # 模板：经典裴蜀定理与背包 DP
        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]
        s = 10000
        dp = [0] * (s + 1)
        dp[0] = 1
        for num in nums:
            for i in range(num, s + 1):
                if dp[i - num]:
                    dp[i] = 1
        ans = s + 1 - sum(dp)
        if reduce(math.gcd, nums) != 1:
            ac.st("INF")
        else:
            ac.st(ans)
        return

    @staticmethod
    def lc_1390(nums: List[int]) -> int:
        # 模板：预处理所有数的所有因子
        nt = PrimeFactor(10 ** 5)
        ans = 0
        for num in nums:
            if len(nt.all_factor[num]) == 4:
                ans += sum(nt.all_factor[num])
        return ans

    @staticmethod
    def lc_1819(nums: List[int]) -> int:
        # 模板：预处理所有整数的所有因子，再枚举gcd计算
        nt = PrimeFactor(2 * 10 ** 5 + 10)
        dct = defaultdict(list)
        for num in set(nums):
            for x in nt.all_factor[num]:
                dct[x].append(num)
        ans = 0
        for num in dct:
            if reduce(math.gcd, dct[num]) == num:
                ans += 1
        return ans

    @staticmethod
    def ac_3727(ac=FastIO()):
        # 模板：脑筋急转弯转换成进制表达问题

        for _ in range(ac.read_int()):
            def check():
                n, k = ac.read_list_ints()
                cnt = Counter()
                for num in ac.read_list_ints():
                    lst = []
                    while num:
                        lst.append(num % k)
                        num //= k
                    for i, va in enumerate(lst):
                        cnt[i] += va
                        if cnt[i] > 1:
                            ac.st("NO")
                            return
                ac.st("YES")
                return

            check()

        return

    @staticmethod
    def ac_4319(ac=FastIO()):
        # 模板：质因数分解后前缀哈希计数
        n, k = ac.read_list_ints()
        a = ac.read_list_ints()
        nt = PrimeFactor(max(a))
        pre = defaultdict(int)
        ans = 0
        for num in a:
            cur = []
            lst = []
            for p, c in nt.prime_factor[num]:
                c %= k
                if c:
                    cur.append((p, c))
                    lst.append((p, k - c))
            ans += pre[tuple(lst)]
            pre[tuple(cur)] += 1
        ac.st(ans)
        return

    @staticmethod
    def ac_4484(ac=FastIO()):
        # 模板：分数在某个进制下是否为有限小数问题
        for _ in range(ac.read_int()):

            def check():
                nonlocal q
                while q > 1:
                    gg = math.gcd(q, b)
                    if gg == 1:
                        break
                    q //= gg

                return q == 1

            p, q, b = ac.read_list_ints()
            g = math.gcd(p, q)
            p //= g
            q //= g

            ac.st("YES" if check() else "NO")
        return

    @staticmethod
    def ac_5049(ac=FastIO()):
        # 模板：使用质因数分解计算组合数
        n, m, h = ac.read_list_ints()
        a = ac.read_list_ints()
        h -= 1
        s = sum(a)
        if s < n:
            ac.st(-1)
            return
        if s - a[h] < n - 1:
            ac.st(1)
            return
        nt = PrimeFactor(s)
        total = nt.comb(s - 1, n - 1)
        part = nt.comb(s - a[h], n - 1)
        ac.st(1 - part / total)
        return
