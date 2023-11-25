import math
import random
from collections import Counter


class NumberTheory:
    def __init__(self):
        return

    @staticmethod
    def get_prime_factor(x):
        # prime factor decomposition supports up to 10**12
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
        # Four Squares Theorem
        # Each number can be represented by the complete sum of squares of at most four numbers
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
    def int_to_roman(num: int) -> str:
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
        # calculate the nth ugly number that only contains the prime factor in primes
        # note that this includes 1
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
        # iterative method for finding the maximum common divisor
        while y:
            x, y = y, x % y
        return x

    def lcm(self, x, y):
        # Least common multiple
        return x * y // self.gcd(x, y)

    @staticmethod
    def factorial_zero_count(num):
        # count of suffix zero of n!
        cnt = 0
        while num > 0:
            cnt += num // 5
            num //= 5
        return cnt

    @staticmethod
    def get_k_bin_of_n(n, k):
        # K-base calculation of integer n
        # supports both positive and negative bases
        # binary, ternary, hexadecimal, and negative bases
        if n == 0:
            return [0]
        if k == 0:
            return []
        assert abs(k) >= 2  # In principle, requirements
        pos = 1 if k > 0 else -1
        k = abs(k)
        lst = []  # 0123456789" + "".join(chr(i+ord("A")) for i in range(26))
        while n:
            lst.append(n % k)
            n //= k
            n *= pos
        lst.reverse()
        return lst

    @staticmethod
    def k_bin_to_ten(k, st: str) -> str:
        # convert k-base characters to decimal characters
        order = "0123456789" + "".join(chr(i + ord("a")) for i in range(26))
        ind = {w: i for i, w in enumerate(order)}
        m = len(st)
        ans = 0
        for i in range(m):
            ans *= k
            ans += ind[st[i]]
        return str(ans)

    def ten_to_k_bin(self, k, st: str) -> str:
        # convert 10 base characters to k base characters
        order = "0123456789" + "".join(chr(i + ord("a")) for i in range(26))
        lst = self.get_k_bin_of_n(int(st), k)
        return "".join(order[i] for i in lst)

    @staticmethod
    def is_prime(num):
        # determine whether a number is prime
        if num <= 1:
            return False
        for i in range(2, min(int(math.sqrt(num)) + 2, num)):
            if num % i == 0:
                return False
        return True

    @staticmethod
    def is_prime4(x):
        """https://zhuanlan.zhihu.com/p/107300262"""
        # determine whether a number is prime
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
        recurrent decimalization of rational numbers to fractions
        1.2727... = (27 / 99) + 1
        1.571428571428... = (571428 / 999999) + 1
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
        """the euler function returns the number of coprime with n that are less than or equal to n"""
        # Note that 1 and 1 are coprime, while prime numbers greater than 1 are not coprime with 1
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
        """euler linear sieve prime number"""
        flag = [False for _ in range(n + 1)]
        prime_numbers = []
        for num in range(2, n + 1):
            if not flag[num]:
                prime_numbers.append(num)
            for prime in prime_numbers:
                if num * prime > n:
                    break
                flag[num * prime] = True
                if num % prime == 0:
                    break
        return prime_numbers

    @staticmethod
    def sieve_of_eratosthenes(n):
        """Eratosthenes screening method returns prime numbers less than or equal to n"""
        primes = [True] * (n + 1)
        p = 2
        while p * p <= n:
            if primes[p]:
                for i in range(p * 2, n + 1, p):
                    primes[i] = False
            p += 1
        primes = [element for element in range(2, n + 1) if primes[element]]
        return primes

    @staticmethod
    def linear_sieve(n):
        """Linear screening of prime numbers and calculating all prime factors"""
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
        """Obtain all factors of an integer, including 1 and itself"""
        factor = set()
        for i in range(1, int(math.sqrt(num)) + 1):
            if num % i == 0:
                factor.add(i)
                factor.add(num // i)
        return sorted(list(factor))

    def pollard_rho(self, n):
        # Randomly return a factor of n [1, 10**9]
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
        """Returns the prime factorization of n and the corresponding number of factors"""
        if n <= 1:
            return Counter()  # special case
        f = self.pollard_rho(n)
        return Counter([n]) if f == n else self.get_prime_factors_with_pollard_rho(
            f) + self.get_prime_factors_with_pollard_rho(n // f)
