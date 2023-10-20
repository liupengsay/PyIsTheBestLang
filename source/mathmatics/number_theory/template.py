import math
import random
from collections import Counter


class NumberTheory:
    def __init__(self):
        return

    @staticmethod
    def get_prime_factor(x):
        # 模板：质因数分解最多支持 1**12
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
        ceil = 10 ** 6
        min_prime = [0] * (ceil + 1)
        for i in range(2, ceil + 1):
            if not min_prime[i]:
                min_prime[i] = i
                for j in range(i * i, ceil + 1, i):
                    min_prime[j] = i

        # 模板：计算 1 到 ceil 所有数字的质数分解
        prime_factor = [[] for _ in range(ceil + 1)]
        for num in range(2, ceil + 1):
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
        lst = [['I', 1], ['IV', 4], ['V', 5], ['IX', 9], ['X', 10], ['XL', 40], ['L', 50], ['XC', 90], ['C', 100],
               ['CD', 400], ['D', 500], ['CM', 900], ['M', 1000]]
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
        dct = {'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900, 'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100,
               'D': 500, 'M': 1000}
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
        factor = [[] for _ in range(n + 1)]
        for i in range(2, n + 1):
            x = 2
            while i * x <= n:
                factor[i * x].append(i)
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
        assert abs(k) >= 2  # 原则上要求
        # 支持正负数
        pos = 1 if k > 0 else -1
        k = abs(k)
        lst = []  # 0123456789" + "".join(chr(i+ord("A")) for i in range(26))
        while n:
            lst.append(n % k)
            n //= k
            n *= pos
        lst.reverse()
        # 最高支持三十六进制的表达
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
        # 获取整数所有的因子包括 1 和它自己
        factor = set()
        for i in range(1, int(math.sqrt(num)) + 1):
            if num % i == 0:
                factor.add(i)
                factor.add(num // i)
        return sorted(list(factor))

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
        return Counter([n]) if f == n else self.get_prime_factors_with_pollard_rho(
            f) + self.get_prime_factors_with_pollard_rho(n // f)
