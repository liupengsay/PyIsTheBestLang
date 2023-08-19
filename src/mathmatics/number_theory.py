import math
import random
import time
import unittest
from collections import Counter
from collections import defaultdict
from functools import reduce
from itertools import combinations
from math import inf
from operator import mul
from typing import List

from src.fast_io import FastIO

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
        
===================================洛谷===================================
P1865 A % B Problem（https://www.luogu.com.cn/problem/P1865）通过线性筛素数后进行二分查询区间素数个数
P1748 H数（https://www.luogu.com.cn/problem/P1748）丑数可以使用堆模拟可以使用指针递增也可以使用容斥原理与二分进行计算
P2723 [USACO3.1]丑数 Humble Numbers（https://www.luogu.com.cn/problem/P2723）第n小的只含给定素因子的丑数
P1952 火星上的加法运算（https://www.luogu.com.cn/problem/P1952）N进制加法
P1555 尴尬的数字（https://www.luogu.com.cn/problem/P1555）二进制与三进制
P1592 互质（https://www.luogu.com.cn/problem/P1592）使用二分与容斥原理计算与 n 互质的第 k 个正整数
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
P1592 互质（https://www.luogu.com.cn/problem/P1592）使用二分与容斥原理计算与 n 互质的第 k 个正整数
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

参考：OI WiKi（xx）
"""


class NumberTheoryPrimeFactor:
    def __init__(self, ceil):
        self.ceil = ceil
        self.prime_factor = [[] for _ in range(self.ceil + 1)]
        self.min_prime = [0] * (self.ceil + 1)
        self.get_min_prime_and_prime_factor()
        return

    def get_min_prime_and_prime_factor(self):
        # 模板：计算 1 到 self.ceil 所有数字的最小质数因子
        for i in range(2, self.ceil + 1):
            if not self.min_prime[i]:
                self.min_prime[i] = i
                for j in range(i * i, self.ceil + 1, i):
                    if not self.min_prime[j]:
                        self.min_prime[j] = i

        # 模板：计算 1 到 self.ceil 所有数字的质数分解（可选）
        for num in range(2, self.ceil + 1):
            i = num
            while num > 1:
                p = self.min_prime[num]
                cnt = 0
                while num % p == 0:
                    num //= p
                    cnt += 1
                self.prime_factor[i].append([p, cnt])
        return


class NumberTheoryAllFactor:
    def __init__(self, ceil):
        self.ceil = ceil+10
        self.factor = [[1] for _ in range(self.ceil+1)]
        self.get_all_factor()
        return

    def get_all_factor(self):
        # 模板：计算 1 到 self.ceil 所有数字的所有因子
        for i in range(2, self.ceil + 1):
            x = 1
            while x*i <= self.ceil:
                self.factor[x*i].append(i)
                x += 1
        return


class NumberTheory:
    def __init__(self):
        return

    @staticmethod
    def get_prime_factor2(x):
        # 模板：质因数分解最多支持 1**10
        ans = []
        j = 2
        while j * j <= x:
            if x % j == 0:
                c = 0
                while x % j == 0:
                    x //= j
                    c += 1
                ans.append([j, c])
            j += 1
        if x > 1:
            ans.append([x, 1])
        return ans

    @staticmethod
    def least_square_sum(n: int) -> int:
        # 模板：四平方数定理（每个数最多用四个数的完全平方和就可以表示）
        while n % 4 == 0:
            n //= 4
        if n % 8 == 7:
            return 4
        for i in range(n + 1):
            temp = i * i
            if temp <= n:
                if int((n - temp) ** 0.5) ** 2 + temp == n:
                    return 1 + (0 if temp == 0 else 1)
            else:
                break
        return 3

    @staticmethod
    def get_min_prime_and_prime_factor():
        
        # 模板：计算 1 到 ceil 所有数字的最小质数因子
        ceil = 10**6
        min_prime = [0]*(ceil+1)
        for i in range(2, ceil+1):
            if not min_prime[i]:
                min_prime[i] = i
                for j in range(i*i, ceil+1, i):
                    min_prime[j] = i
        
        # 模板：计算 1 到 ceil 所有数字的质数分解
        prime_factor = [[] for _ in range(ceil+1)]
        for num in range(2, ceil+1):
            i = num
            while num > 1:
                p = min_prime[num]
                cnt = 0
                while num % p == 0:
                    num //= p
                    cnt += 1
                prime_factor[i].append([p, cnt])
        return
        
    @staticmethod
    def get_num_prime_factor(ceil):
        # 模板：快速计算 1~ceil 的所有质数因子
        prime = [[] for _ in range(ceil + 1)]
        for i in range(2, ceil + 1):
            if not prime[i]:
                prime[i].append(i)
                # 从 i*i 开始作为 prime[j] 的最小质数因子
                for j in range(i * 2, ceil + 1, i):
                    prime[j].append(i)
        return prime

    @staticmethod
    def int_to_roman(num: int) -> str:

        # 模板: 罗马数字转整数
        lst = [['I', 1], ['IV', 4], ['V', 5], ['IX', 9], ['X', 10], ['XL', 40], ['L', 50], ['XC', 90], ['C', 100], ['CD', 400], ['D', 500], ['CM', 900], ['M', 1000]]
        n = len(lst)
        i = n - 1
        ans = ''
        while i >= 0:
            if num >= lst[i][1]:
                k = num // lst[i][1]
                ans += k * lst[i][0]
                num -= k * lst[i][1]
                if num == 0:
                    return ans
            else:
                i -= 1
        return ans

    @staticmethod
    def roman_to_int(s: str) -> int:
        dct = {'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900, 'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        ans = i = 0
        n = len(s)
        while i < n:
            if i + 1 < n and s[i:i + 2] in dct:
                ans += dct[s[i:i + 2]]
                i += 2
            else:
                ans += dct[s[i]]
                i += 1
        return ans

    @staticmethod
    def nth_super_ugly_number(n: int, primes) -> int:

        # 计算只含 primes 中的质因数的第 n 个丑数，注意这里包含了 1
        dp = [1] * n
        m = len(primes)
        points = [0] * m
        for i in range(1, n):
            nex = float('inf')
            for j in range(m):
                if primes[j] * dp[points[j]] < nex:
                    nex = primes[j] * dp[points[j]]
            dp[i] = nex
            for j in range(m):
                if primes[j] * dp[points[j]] == nex:
                    points[j] += 1
        return dp[n - 1]

    @staticmethod
    def gcd(x, y):
        # 模板: 迭代法求最大公约数
        while y:
            x, y = y, x % y
        return x

    def lcm(self, x, y):
        # 最小公倍数
        return x * y // self.gcd(x, y)

    @staticmethod
    def get_factor_upper(n):
        # 模板：使用素数筛类似的方法获取小于等于 n 的所有数除 1 与自身之外的所有因数（倍数法）
        factor = [[] for _ in range(n+1)]
        for i in range(2, n+1):
            x = 2
            while i*x <= n:
                factor[i*x].append(i)
                x += 1
        return factor

    @staticmethod
    def factorial_zero_count(num):
        # 阶乘后的后缀零个数
        cnt = 0
        while num > 0:
            cnt += num // 5
            num //= 5
        return cnt

    @staticmethod
    def get_k_bin_of_n(n, k):
        # 整数n的k进制计算（支持正数进制与负数进制）二进制、三进制、十六进制、负进制
        if n == 0:
            return [0]
        if k == 0:
            return []
        # 支持正负数
        pos = 1 if k > 0 else -1
        k = abs(k)
        lst = []
        while n:
            lst.append(n % k)
            n //= k
            n *= pos
        lst.reverse()
        # 最高支持三十六进制的表达
        # "0123456789" + "".join(chr(i+ord("A")) for i in range(26))
        return lst

    @staticmethod
    def k_bin_to_ten(k, st: str) -> str:
        # k进制字符转为 10 进制字符
        order = "0123456789" + "".join(chr(i + ord("a")) for i in range(26))
        ind = {w: i for i, w in enumerate(order)}
        m = len(st)
        ans = 0
        for i in range(m):
            ans *= k
            ans += ind[st[i]]
        return str(ans)

    def ten_to_k_bin(self, k, st: str) -> str:
        # 10 进制字符转为 k 进制字符
        order = "0123456789" + "".join(chr(i + ord("a")) for i in range(26))
        lst = self.get_k_bin_of_n(int(st), k)
        return "".join(order[i] for i in lst)

    @staticmethod
    def is_prime(num):
        # 判断数是否为质数
        if num <= 1:
            return False
        for i in range(2, min(int(math.sqrt(num)) + 2, num)):
            if num % i == 0:
                return False
        return True

    @staticmethod
    def is_prime1(x):
        if x == 1:
            return False
        for i in range(2, x):
            if x % i == 0:
                return False
        return True

    @staticmethod
    def is_prime2(x):
        if x == 1:
            return False
        for i in range(2, int(x ** 0.5) + 1):
            if x % i == 0:
                return False
        return True

    @staticmethod
    def is_prime3(x):
        if x == 1:
            return False
        if x == 2:
            return True
        elif x % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True

    @staticmethod
    def is_prime4(x):
        """https://zhuanlan.zhihu.com/p/107300262
        任何一个自然数，总可以表示成以下六种形式之一：6n，6n+1，6n+2，6n+3，6n+4，6n+5（n=0,1,2...）
        我们可以发现，除了2和3，只有形如6n+1和6n+5的数有可能是质数。
        且形如6n+1和6n+5的数如果不是质数，它们的因数也会含有形如6n+1或者6n+5的数，因此可以得到如下算法：
        """
        if x == 1:
            return False
        if (x == 2) or (x == 3):
            return True
        if (x % 6 != 1) and (x % 6 != 5):
            return False
        for i in range(5, int(math.sqrt(x)) + 1, 6):
            if (x % i == 0) or (x % (i + 2) == 0):
                return False
        return True

    @staticmethod
    def is_prime5(n):
        if n == 2:
            return True
        if n <= 1 or not n ^ 1:
            return False

        for i in range(15):
            rand = random.randint(2, n - 1)
            if pow(rand, n - 1, n) != 1:
                return False

        return True

    @staticmethod
    def rational_number_to_fraction(st):
        """
        # 有理数循环小数化为分数
        1.2727... = (27 / 99) + 1
        1.571428571428... = (571428 / 999999) + 1
        有n位循环 = (循环部分 / n位9) + 整数部分
        最后约简
        """
        n = len(st)
        a = int(st)
        b = int("9" * n)
        c = math.gcd(a, b)
        a //= c
        b //= c
        return [a, b]

    @staticmethod
    def euler_phi(n):
        # 欧拉函数返回小于等于n的与n互质的个数
        # 注意1和1互质，而大于1的质数与1不互质
        ans = n
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                ans = ans // i * (i - 1)
                while n % i == 0:
                    n = n // i
        if n > 1:
            ans = ans // n * (n - 1)
        return int(ans)

    @staticmethod
    def euler_flag_prime(n):
        # 欧拉线性筛素数，欧拉筛
        # 说明：返回小于等于 n 的所有素数
        flag = [False for _ in range(n + 1)]
        prime_numbers = []
        for num in range(2, n + 1):
            if not flag[num]:
                prime_numbers.append(num)
            for prime in prime_numbers:
                if num * prime > n:
                    break
                flag[num * prime] = True
                if num % prime == 0:  # 这句是最有意思的地方  下面解释
                    break
        return prime_numbers

    @staticmethod
    def sieve_of_eratosthenes(n):  # 埃拉托色尼筛选法，返回小于等于n的素数，质数筛
        primes = [True] * (n + 1)  # 范围0到n的列表
        p = 2  # 这是最小的素数
        while p * p <= n:  # 一直筛到sqrt(n)就行了
            if primes[p]:  # 如果没被筛，一定是素数
                for i in range(p * 2, n + 1, p):  # 筛掉它的倍数即可
                    primes[i] = False
            p += 1
        primes = [
            element for element in range(
                2, n + 1) if primes[element]]  # 得到所有小于等于n的素数
        return primes

    @staticmethod
    def linear_sieve(n):
        # 模板：线性筛素数并计算出所有的质因子
        is_prime = [True] * (n + 1)
        primes = []
        min_prime = [0] * (n + 1)
        for i in range(2, n + 1):
            if is_prime[i]:
                primes.append(i)
            for p in primes:
                if i * p > n:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    break
        return primes, min_prime

    @staticmethod
    def get_all_factor(num):
        # 获取整数所有的因子包括1和它自己
        factor = set()
        for i in range(1, int(math.sqrt(num)) + 1):
            if num % i == 0:
                factor.add(i)
                factor.add(num // i)
        return sorted(list(factor))

    @staticmethod
    def get_prime_factor(num):
        # 质因数分解
        res = []
        for i in range(2, int(math.sqrt(num)) + 1):
            cnt = 0
            while num % i == 0:
                num //= i
                cnt += 1
            if cnt:
                res.append([i, cnt])
            if i > num:
                break
        if num != 1 or not res:
            res.append([num, 1])
        return res

    def get_prime_cnt(self, x, y):
        # P1592 互质
        # P2429 制杖题

        # 使用容斥原理计算 [1, y] 内与 x 互质的个数
        if x == 1:
            return y

        lst = self.get_prime_factor(x)
        prime = [p for p, _ in lst]
        m = len(lst)
        # 求与 x 不互质的数，再减去这部分数
        res = 0
        for i in range(1, m+1):
            for item in combinations(prime, i):
                cur = 1
                for num in item:
                    cur *= num
                res += (y//cur)*(-1)**(i+1)
        return y-res

    def pollard_rho(self, n):
        # 随机返回一个 n 的因数 [1, 10**9]
        if n & 1 == 0:
            return 2
        if n % 3 == 0:
            return 3

        s = ((n - 1) & (1 - n)).bit_length() - 1
        d = n >> s
        for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
            p = pow(a, d, n)
            if p == 1 or p == n - 1 or a % n == 0:
                continue
            for _ in range(s):
                prev = p
                p = (p * p) % n
                if p == 1:
                    return self.gcd(prev - 1, n)
                if p == n - 1:
                    break
            else:
                for i in range(2, n):
                    x, y = i, (i * i + 1) % n
                    f = self.gcd(abs(x - y), n)
                    while f == 1:
                        x, y = (x * x + 1) % n, (y * y + 1) % n
                        y = (y * y + 1) % n
                        f = self.gcd(abs(x - y), n)
                    if f != n:
                        return f
        return n

    def get_prime_factors_with_pollard_rho(self, n):
        # 返回 n 的质因数分解与对应因子个数
        if n <= 1:
            return Counter()  # 注意特例返回
        f = self.pollard_rho(n)
        return Counter([n]) if f == n else self.get_prime_factors_with_pollard_rho(f) + self.get_prime_factors_with_pollard_rho(n // f)


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
        n = ac.read_int()
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
    def lc_6309(nums: List[int]) -> int:
        # 模板：计算 1 到 n 的数所有的质因子并使用差分确定作用范围
        prime = NumberTheory().get_num_prime_factor(10**6)
        n = len(nums)
        dct = defaultdict(list)
        for i, num in enumerate(nums):
            for p in prime[num]:
                dct[p].append(i)

        # 确定作用域
        diff = [0] * (n + 1)
        for p in dct:
            i, j = dct[p][0], dct[p][-1]
            a, b = i, j - 1
            if a <= b:
                diff[a] += 1
                diff[b + 1] -= 1
        for i in range(1, n + 1):
            diff[i] += diff[i - 1]
        for i in range(n - 1):
            if not diff[i]:
                return i
        return -1
    
    @staticmethod
    def lc_2464(nums: List[int]) -> int:
        # 模板：计算 1 到 n 的数所有的质因子并使用动态规划计数
        nt = NumberTheoryPrimeFactor(max(nums))
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
    def lc_lcp14(nums: List[int]) -> int:
        # 模板：计算 1 到 n 的数所有的质因子并使用动态规划计数
        nt = NumberTheoryPrimeFactor(max(nums))
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
    def cf_1349a(ac=FastIO()):
        # 模板：质因数分解，枚举最终结果当中质因子的幂次
        n = ac.read_int()
        nums = ac.read_list_ints()
        nmp = NumberTheoryPrimeFactor(max(nums))
        dct = defaultdict(list)

        for num in nums:
            for p, c in nmp.prime_factor[num]:
                dct[p].append(c)

        ans = 1
        for p in dct:
            if len(dct[p]) >= n - 1:
                dct[p].sort()
                ans *= p**dct[p][-n + 1]
        ac.st(ans)
        return

    @staticmethod
    def cf_1295d(ac=FastIO()):
        # 模板：欧拉函数求解
        for _ in range(ac.read_int()):
            a, m = ac.read_ints()
            g = math.gcd(a, m)
            mm = m // g
            ans = NumberTheory().euler_phi(mm)
            ac.st(ans)
        return

    @staticmethod
    def cf_1458a(ac=FastIO()):
        # 模板：gcd公式变换求解gcd(x,y)=gcd(x-y,y)
        m, n = ac.read_ints()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        g = 0
        # 推广到n维
        for i in range(1, m):
            g = math.gcd(g, a[i]-a[i-1])
        ans = [math.gcd(g, a[0]+num) for num in b]
        ac.lst(ans)
        return

    @staticmethod
    def main(ac=FastIO()):
        # 模板：预先枚举质因子，再进行质因数分解
        primes = NumberTheory().euler_flag_prime((4 * 10 ** 3))
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            cnt = defaultdict(int)
            for num in nums:
                for x in primes:
                    if x > num:
                        break
                    y = 0
                    while num % x == 0:
                        num //= x
                        y += 1
                    if y:
                        cnt[x] += y
                if num != 1:
                    cnt[num] += 1
            lst = list(cnt.values())
            even = sum(x // 2 for x in lst)
            odd = sum(x % 2 for x in lst)
            ans = odd // 3
            odd %= 3
            if odd:
                if ans or even:
                    ac.st(ans + even)
                else:
                    ac.st(0)
            else:
                ac.st(ans + even)
        return

    @staticmethod
    def ac_97_1(ac=FastIO()):
        # 模板：a^b的所有约数之和
        a, b = ac.read_ints()
        lst = NumberTheory().get_prime_factor2(a)
        mod = 9901
        ans = 1
        for p, c in lst:
            ans *= (pow(p, b*c+1, mod)-1) * pow(p-1, -1, mod)
            ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def ac_97_2(ac=FastIO()):
        # 模板：a^b的所有约数之和

        def check(pp, cc):

            # 等比数列求和递归分治计算
            if cc == 0:
                return 1
            if cc % 2 == 1:
                return (1 + pow(pp, (cc + 1) // 2, mod)) * check(pp, (cc - 1) // 2)
            return (1 + pow(pp, (cc + 0) // 2, mod)) * check(pp, (cc - 1) // 2) + pow(pp, cc, mod)

        a, b = ac.read_ints()
        if a == 0:
            ac.st(0)
            return
        lst = NumberTheory().get_prime_factor2(a)
        mod = 9901
        ans = 1
        for p, c in lst:
            ans *= check(p, c * b)
            ans %= mod
        ac.st(ans)
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
    def ac_196(ac=FastIO()):

        # 模板：经典计算质数距离对
        primes = NumberTheory().sieve_of_eratosthenes(2 ** 16)
        while True:
            lst = ac.read_list_ints()
            if not lst:
                break

            left, right = lst
            dp = [0] * (right - left + 1)
            for p in primes:
                x = max(math.ceil(left / p), 2) * p
                while left <= x <= right:
                    dp[x - left] = 1
                    x += p

            rest = [x + left for x in range(right - left + 1) if not dp[x] and x + left > 1]
            if len(rest) < 2:
                ac.st("There are no adjacent primes.")
            else:
                ans1 = [rest[0], rest[1]]
                ans2 = [rest[0], rest[1]]
                m = len(rest)
                for i in range(2, m):
                    a, b = rest[i - 1], rest[i]
                    if b - a < ans1[1] - ans1[0]:
                        ans1 = [a, b]
                    if b - a > ans2[1] - ans2[0]:
                        ans2 = [a, b]
                ac.st(f"{ans1[0]},{ans1[1]} are closest, {ans2[0]},{ans2[1]} are most distant.")
        return

    @staticmethod
    def ac_198(ac=FastIO()):

        # 模板：经典计算最大的反质数（反素数，即约数或者说因数个数大于任何小于它的数的因数个数）
        n = ac.read_int()
        primes = NumberTheory().sieve_of_eratosthenes(50)
        x = reduce(mul, primes)
        while x > n:
            x //= primes.pop()
        # 充要条件为 2^c1*3^c2...且c1>=c2
        m = len(primes)
        ans = 1
        ans = [1, 1]
        stack = [[1, 1, int(math.log2(n)) + 1, 0]]
        while stack:
            x, cnt, mi, i = stack.pop()
            if mi == 0 or i == m:
                if cnt > ans[1] or (cnt == ans[1] and x < ans[0]):
                    ans = [x, cnt]
                continue
            for y in range(mi, -1, -1):
                if x * primes[i]**y <= n:
                    stack.append([x * primes[i]**y, cnt * (y + 1), y, i + 1])
        ac.st(ans[0])
        return

    @staticmethod
    def ac_199(ac=FastIO()):
        # 模板：计算 sum(k%i for i in range(n))
        n, k = ac.read_ints()
        ans = n*k
        left = 1
        while left <= min(n, k):
            right = min(k//(k//left), n)
            ans -= (k//left)*(left+right)*(right-left+1)//2
            left = right+1
        ac.st(ans)
        return

    @staticmethod
    def lg_p1069(ac=FastIO()):
        # 模板：质因数分解，贪心匹配模拟
        n = ac.read_int()
        m1, m2 = ac.read_list_ints()
        lst = NumberTheory().get_prime_factor2(m1)
        ans = inf
        for num in ac.read_list_ints():
            res = 0
            for p, c in lst:
                if num % p != 0:
                    break
                tmp = num
                x = 0
                while tmp % p == 0:
                    tmp //= p
                    x += 1
                res = ac.max(res, math.ceil(c*m2/x))
            else:
                ans = ac.min(ans, res)
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def lg_p1072(ac=FastIO()):
        # 模板：枚举所有因数
        nt = NumberTheory()
        for _ in range(ac.read_int()):
            a0, a1, b0, b1 = ac.read_ints()
            factor = [num for num in nt.get_all_factor(b1)
                      if num % a1 == 0 and math.gcd(num, a0) == a1
                      and b0 * num // math.gcd(num, b0) == b1]
            ac.st(len(factor))
        return

    @staticmethod
    def lg_p1593(ac=FastIO()):
        # 模板：使用质因数分解与快速幂计算a^b的所有因子之和
        mod = 9901
        a, b = ac.read_ints()
        if a == 1 or b == 0:
            ac.st(1)
        else:
            # 分解质因数
            cnt = dict()
            for p, c in NumberTheory().get_prime_factor2(a):
                cnt[p] = c
            # (1+p1+p1^2+...+p1^cb)*...
            ans = 1
            for k in cnt:
                c = cnt[k] * b
                if (k - 1) % mod:  # 即 k % mod ！= 1 此时才有逆元
                    # 等比数列计算乘法逆元，逆元要求与mod互质否则需要额外计算
                    ans *= (pow(k, c + 1, mod) - 1) * pow(k - 1, -1, mod)
                    ans %= mod
                else:
                    # 此时无乘法逆元
                    ans *= (c + 1)
                    ans %= mod
            ac.st(ans)
        return

    @staticmethod
    def lg_p2527(ac=FastIO()):
        # 模板：丑数即只含特定质因子的数
        n, k = ac.read_ints()
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
    def lg_p2557(ac=FastIO()):
        # 模板：利用质因数分解与等比数列计算因子之和
        a, b = ac.read_ints()
        if a == 1 or b == 0:
            ac.st(1)
        else:
            # 分解质因数
            cnt = dict()
            for p, c in NumberTheory().get_prime_factor2(a):
                cnt[p] = c
            # (1+p1+p1^2+...+p1^cb)*...
            ans = 1
            for k in cnt:
                c = cnt[k] * b
                ans *= (k**(c + 1) - 1) // (k - 1)
            ac.st(ans)
        return

    @staticmethod
    def lg_p4446(ac=FastIO()):
        # 模板：预先处理出素数然后计算最大的完全立方数因子
        prime = NumberTheory().sieve_of_eratosthenes(int(10**(18 / 4)) + 1)
        ac.read_int()
        for num in ac.read_list_ints():
            ans = 1
            for p in prime:
                if p > num:
                    break
                c = 0
                while num % p == 0:
                    c += 1
                    num //= p
                ans *= p**(c // 3)

            # 使用二分判断数字是否为完全立方数
            low = 1
            high = int(num**(1 / 3)) + 1
            while low < high - 1:
                mid = low + (high - low) // 2
                if mid**3 <= num:
                    low = mid
                else:
                    high = mid
            if high**3 == num:
                ans *= high
            elif low**3 == num:
                ans *= low
            ac.st(ans)
        return

    @staticmethod
    def lg_p4752(ac=FastIO()):
        # 模板：判断除数是否为质数
        nt = NumberTheory()
        for _ in range(ac.read_int()):
            ac.read_ints()
            cnt = Counter(sorted(ac.read_list_ints()))
            if cnt[0]:
                ac.st("NO")
                continue
            for num in ac.read_list_ints():
                cnt[num] -= 1

            rest = []
            for num in cnt:
                if cnt[num] and num != 1:
                    rest.append([num, cnt[num]])
            if len(rest) >= 2:
                ac.st("NO")
            elif len(rest) == 1:
                if rest[0][1] > 1:
                    ac.st("NO")
                else:
                    if nt.is_prime4(rest[0][0]):
                        ac.st("YES")
                    else:
                        ac.st("NO")
            else:
                ac.st("NO")
        return

    @staticmethod
    def lg_p5248(ac=FastIO()):
        # 模板：经典进制题目
        m, fm = ac.read_ints()
        lst = []
        while fm:
            lst.append(fm % m)
            fm //= m
        ac.st(len(lst))
        ac.lst(lst)
        return

    @staticmethod
    def lg_p5253(ac=FastIO()):
        # 模板：经典方程变换计算 (x-n)*(y-n)=n^2 的对数
        n = ac.read_int()
        lst = NumberTheory().get_prime_factor2(n)
        ans = 1
        for _, c in lst:
            # 转换为求数字的因数个数
            ans *= (2 * c + 1)
        ac.st((ans + 1) // 2)
        return

    @staticmethod
    def lg_p7960(ac=FastIO()):
        # 模板：类似埃氏筛的思路进行预处理
        n = 10**7
        dp = [0] * (n + 1)
        for x in range(1, n + 1):
            if "7" in str(x):
                y = 1
                while x * y <= n:
                    dp[x * y] = 1
                    y += 1
        post = 10**7 + 1
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
        dp = [0]*(s+1)
        dp[0] = 1
        for num in nums:
            for i in range(num, s+1):
                if dp[i-num]:
                    dp[i] = 1
        ans = s+1-sum(dp)
        if reduce(math.gcd, nums) != 1:
            ac.st("INF")
        else:
            ac.st(ans)
        return

    @staticmethod
    def lg_8778(ac=FastIO()):
        # 模板：经典枚举素因子后O(n^0.25)计算是否为完全平方数与立方数
        primes = NumberTheory().sieve_of_eratosthenes(4000)

        def check(xx):
            for r in range(2, 6):
                a = int(xx**(1/r))
                for ww in [a-1, a, a+1, a+2]:
                    if ww**r == xx:
                        return True
            return False

        n = ac.read_int()
        for _ in range(n):
            num = ac.read_int()
            flag = True
            for p in primes:
                if p > num:
                    break
                x = 0
                while num % p == 0:
                    x += 1
                    num //= p
                if x == 1:
                    flag = False
                    break
            if flag and check(num):
                ac.st("yes")
            else:
                ac.st("no")
        return

    @staticmethod
    def lc_1390(nums: List[int]) -> int:
        # 模板：预处理所有数的所有因子
        nt = NumberTheoryAllFactor(10**5)
        ans = 0
        for num in nums:
            if len(nt.factor[num]) == 4:
                ans += sum(nt.factor[num])
        return ans

    @staticmethod
    def lc_1819(nums: List[int]) -> int:
        # 模板：预处理所有整数的所有因子，再枚举gcd计算
        nt = NumberTheoryAllFactor(2 * 10 ** 5 + 10)
        dct = defaultdict(list)
        for num in set(nums):
            for x in nt.factor[num]:
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
                n, k = ac.read_ints()
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
        n, k = ac.read_ints()
        a = ac.read_list_ints()
        nt = NumberTheoryPrimeFactor(max(a))
        pre = defaultdict(int)
        ans = 0
        for num in a:
            cur = []
            lst = []
            for p, c in nt.prime_factor[num]:
                c %= k
                if c:
                    cur.append((p, c))
                    lst.append((p, k-c))
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

            p, q, b = ac.read_ints()
            g = math.gcd(p, q)
            p //= g
            q //= g

            ac.st("YES" if check() else "NO")
        return

    @staticmethod
    def ac_4486(ac=FastIO()):
        # 模板：经典质数分解贪心题
        n = ac.read_int()
        if n == 1:
            ac.lst([1, 0])
            return

        res = NumberTheory().get_prime_factor(n)

        ans = 1
        x = 0
        ind = [2 ** i for i in range(32)]
        lst = []
        for p, c in res:
            ans *= p
            for i in range(32):
                if ind[i] >= c:
                    if ind[i] > c:
                        x = 1
                    lst.append(i)
                    break
        cnt = max(w for w in lst)
        if any(w < cnt for w in lst) or x:
            cnt += 1
        ac.lst([ans, cnt])
        return

    @staticmethod
    def ac_4622(ac=FastIO()):
        # 模板：思维题贪心构造
        n = ac.read_int()
        if n < 4:
            ac.st(1)
        elif n % 2 == 0:
            ac.st(2)
        else:
            if NumberTheory().is_prime4(n-2):
                ac.st(2)
            else:
                ac.st(3)

        return


class TestGeneral(unittest.TestCase):

    def test_prime_cnt(self):
        nt = NumberTheory()
        for _ in range(100):
            x = random.randint(1, 100)
            y = random.randint(1, 10000)
            cnt = 0
            for i in range(1, y+1):
                if math.gcd(i, x) == 1:
                    cnt += 1
            assert nt.get_prime_cnt(x, y) == cnt
        return

    def test_get_prime_factor(self):
        for i in range(1, 10):
            x = random.randint(i, 10**10)
            t0 = time.time()
            cnt1 = NumberTheory().get_prime_factor(x)
            t1 = time.time()
            cnt2 = NumberTheory().get_prime_factor2(x)
            t2 = time.time()
            print(t1-t0, t2-t1)
            assert cnt1 == cnt2

    def test_get_prime_factor_pollard(self):
        for i in range(1, 10):
            nt = NumberTheory()
            for i in range(1, 100000):
                res = nt.get_prime_factor(i)
                cnt = nt.get_prime_factors_with_pollard_rho(i)
                num = 1
                for val, c in res:
                    num *= val ** c
                    if val > 1:
                        assert cnt[val] == c
                assert num == i

        nt = NumberTheory()
        num = 2
        assert nt.get_prime_factor(num) == [[2, 1]]
        num = 1
        assert nt.get_prime_factor(num) == [[1, 1]]
        num = 2 * (3**2) * 7 * (11**3)
        assert nt.get_prime_factor(num) == [[2, 1], [3, 2], [7, 1], [11, 3]]
        return

    def test_euler_phi(self):
        nt = NumberTheory()
        assert nt.euler_phi(10**11 + 131) == 66666666752
        return

    def test_euler_shai(self):
        nt = NumberTheory()
        label = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        pred = nt.euler_flag_prime(30)
        assert label == pred
        assert len(nt.euler_flag_prime(10**6)) == 78498
        return

    def test_eratosthenes_shai(self):
        nt = NumberTheory()
        assert len(nt.sieve_of_eratosthenes(10**6)) == 78498
        return

    def test_factorial_zero_count(self):
        nt = NumberTheory()
        num = random.randint(1, 100)
        s = str(math.factorial(num))
        cnt = 0
        for w in s[::-1]:
            if w == "0":
                cnt += 1
            else:
                break
        assert nt.factorial_zero_count(num) == cnt
        return

    def test_get_k_bin_of_n(self):
        nt = NumberTheory()
        num = random.randint(1, 100)
        assert nt.get_k_bin_of_n(num, 2) == [int(w) for w in bin(num)[2:]]

        assert nt.get_k_bin_of_n(4, -2) == [1, 0, 0]
        return

    def test_rational_number_to_fraction(self):
        nt = NumberTheory()
        assert nt.rational_number_to_fraction("33") == [1, 3]
        return

    def test_is_prime(self):
        nt = NumberTheory()
        assert not nt.is_prime(1)
        assert nt.is_prime(5)
        assert not nt.is_prime(51)
        for _ in range(10):
            i = random.randint(1, 10**4)
            assert nt.is_prime(i) == nt.is_prime4(i) == nt.is_prime5(i)

        for _ in range(1):
            x = random.randint(10**8, 10**9)
            y = x + 10**6
            for num in range(x, y+1):
                nt.is_prime4(x)
        return

    def test_gcd_lcm(self):
        nt = NumberTheory()
        a, b = random.randint(1, 1000), random.randint(1, 1000)
        assert nt.gcd(a, b) == math.gcd(a, b)
        assert nt.lcm(a, b) == math.lcm(a, b)
        return

    def test_get_factor(self):
        nt = NumberTheory()

        num = 1000
        ans = nt.get_factor_upper(num)
        for i in range(1, num+1):
            assert ans[i] == nt.get_all_factor(i)[1:-1]
        return

    def test_roma_int(self):
        nt = NumberTheory()

        num = 1000
        for i in range(1, num+1):
            assert nt.roman_to_int(nt.int_to_roman(i)) == i
        return


if __name__ == '__main__':
    unittest.main()
