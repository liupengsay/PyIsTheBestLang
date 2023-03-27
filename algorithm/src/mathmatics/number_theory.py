

import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
from functools import reduce
from operator import xor
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy
from algorithm.src.fast_io import FastIO

import math
import random
import unittest
from itertools import combinations
from collections import Counter
from algorithm.src.fast_io import FastIO
from functools import reduce
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

================================CodeForces================================
C. Hossam and Trainees（https://codeforces.com/problemset/problem/1771/C）使用pollard_rho进行质因数分解
A. Enlarge GCD（https://codeforces.com/problemset/problem/1034/A）经典求 1 到 n 所有数字的质因子个数总和 
C. Hossam and Trainees（https://codeforces.com/problemset/problem/1771/C）使用pollard_rho进行质因数分解
D. Two Divisors（https://codeforces.com/problemset/problem/1366/D）计算最小的质因子，使用构造判断是否符合条件
A. Orac and LCM（https://codeforces.com/contest/1349/problem/A）质因数分解，枚举最终结果当中质因子的幂次
D. Same GCDs（https://codeforces.com/problemset/problem/1295/D）利用最大公因数的特性转换为欧拉函数求解，即比 n 小且与 n 互质的数个数

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


class NumberTheory:
    def __init__(self):
        return

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
        # 使用素数筛类似的方法获取小于等于 n 的所有数除 1 与自身之外的所有因数
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
        # 欧拉线性筛素数
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
    def sieve_of_eratosthenes(n):  # 埃拉托色尼筛选法，返回小于等于n的素数
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
        inf = float("inf")
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
        inf = float("inf")
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
