import math
import random
from collections import Counter




class NumBase:
    def __init__(self):
        return

    @staticmethod
    def get_k_bin_of_n(n, k):
        """K-base calculation of integer n supports both positive and negative bases"""
        assert abs(k) >= 2  # in principle, requirements
        if n == 0:  # binary, ternary, hexadecimal, and negative bases
            return [0]
        if k == 0:
            return []
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
        """convert k-base characters to decimal characters"""
        order = "0123456789" + "".join(chr(i + ord("a")) for i in range(26))
        ind = {w: i for i, w in enumerate(order)}
        m = len(st)
        ans = 0
        for i in range(m):
            ans *= k
            ans += ind[st[i]]
        return str(ans)

    def ten_to_k_bin(self, k, st: str) -> str:
        """convert 10 base characters to k base characters"""
        order = "0123456789" + "".join(chr(i + ord("a")) for i in range(26))
        lst = self.get_k_bin_of_n(int(st), k)
        return "".join(order[i] for i in lst)


class RomeToInt:
    def __init__(self):
        return

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


class PrimeJudge:
    def __init__(self):
        return

    @staticmethod
    def is_prime_speed(num):  # faster!
        """https://zhuanlan.zhihu.com/p/107300262"""
        assert num > 0
        if num == 1:
            return False
        if (num == 2) or (num == 3):
            return True
        if (num % 6 != 1) and (num % 6 != 5):
            return False
        for i in range(5, int(math.sqrt(num)) + 1, 6):
            if (num % i == 0) or (num % (i + 2) == 0):
                return False
        return True

    @staticmethod
    def is_prime_general(num):
        """general square complexity"""
        assert num > 0
        if num == 1:
            return False
        for i in range(2, min(int(math.sqrt(num)) + 2, num)):
            if num % i == 0:
                return False
        return True

    @staticmethod
    def is_prime_random(num):
        """random guess may not be right"""
        assert num > 0
        if num == 2:
            return True
        if num == 1 or not num ^ 1:
            return False
        for i in range(128):  # can be adjusted
            rand = random.randint(2, num - 1)
            if pow(rand, num - 1, num) != 1:
                return False  # must not be prime number
        return True  # still may be prime number


class NumFactor:
    def __init__(self):
        return

    @staticmethod
    def get_all_factor(num):  # faster when 1 <= num <= 10**6!
        """Obtain all factors of an integer, including 1 and itself"""
        assert num >= 1
        pre = []
        post = []
        for i in range(1, int(math.sqrt(num)) + 1):
            if num % i == 0:
                pre.append(i)
                if num // i != i:
                    post.append(num // i)
        return pre + post[::-1]

    @staticmethod
    def get_all_factor_square(primes, num):  # 1 <= num <= 10**9!
        """Obtain all square factors of an integer, including 1"""
        lst = []
        for p in primes:
            cnt = 0
            while num % p == 0:
                num //= p
                cnt += 1
            if cnt > 1:
                lst.append((p, cnt // 2))
        if int(num ** 0.5) ** 2 == num:
            lst.append((int(num ** 0.5), 1))
        pre = {1}
        for p, c in lst:
            for num in list(pre):
                for i in range(1, c + 1):
                    pre.add(num * p ** i)
        return sorted([x * x for x in pre])

    @staticmethod
    def get_prime_factor(num):  # faster when 1 <= num <= 10**6!
        """prime factor decomposition supports up to 10**12"""
        assert num >= 1
        ans = []
        j = 2
        while j * j <= num:
            if num % j == 0:
                c = 0
                while num % j == 0:
                    num //= j
                    c += 1
                ans.append((j, c))
            j += 1
        if num > 1:
            ans.append((num, 1))
        return ans

    @staticmethod
    def get_prime_with_pollard_rho(num):  # faster when 10**6 <= num <= (1 << 64)!
        """returns the prime factorization of n and the corresponding number of factors for larger number"""

        def pollard_rho(n):
            # randomly return a factor of n [1, 1 << 64]
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
                        return math.gcd(prev - 1, n)
                    if p == n - 1:
                        break
                else:
                    for i in range(2, n):
                        x, y = i, (i * i + 1) % n
                        f = math.gcd(x - y, n)
                        while f == 1:
                            x, y = (x * x + 1) % n, (y * y + 1) % n
                            y = (y * y + 1) % n
                            f = math.gcd(x - y, n)
                        if f != n:
                            return f
            return n

        if num <= 1:
            return Counter()  # special case

        pr = dict()
        sub = dict()
        stack = [num]
        while stack:
            m = stack.pop()
            if m > 0:
                if m in pr:
                    continue
                pr[m] = pollard_rho(m)
                if m == pr[m]:
                    sub[m] = Counter([m])
                    continue
                stack.append(~m)
                stack.append(m // pr[m])
                stack.append(pr[m])
            else:
                m = ~m
                sub[m] = sub[m // pr[m]] + sub[pr[m]]
        return sub[num]

    def get_all_with_pollard_rho(self, num):  # faster when 10**6 <= num <= (1 << 64)!
        """returns the prime factorization of n and the corresponding number of factors for larger number"""
        assert num >= 1

        if num == 1:
            return [1]
        cnt = self.get_prime_with_pollard_rho(num)
        pre = [1]
        for p in cnt:
            nex = []
            for w in pre:
                cur = w * p
                for x in range(1, cnt[p] + 1):
                    nex.append(cur)
                    cur *= p
            pre.extend(nex)
        return sorted(pre)


class PrimeSieve:
    def __init__(self):
        return

    @staticmethod
    def eratosthenes_sieve(n):  # faster!
        """eratosthenes screening method returns prime numbers less than or equal to n"""
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
    def euler_sieve(n):
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


class NumTheory:
    def __init__(self):
        return

    @staticmethod
    def least_square_sum(n: int) -> int:
        """Four Squares Theorem Each number can be represented by the complete sum of squares of at most four numbers"""
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
    def nth_super_ugly_number(n: int, primes) -> int:
        """calculate the nth ugly number that only contains the prime factor in primes"""
        dp = [1] * n  # note that this includes 1
        m = len(primes)
        points = [0] * m
        for i in range(1, n):
            nex = math.inf
            for j in range(m):
                if primes[j] * dp[points[j]] < nex:
                    nex = primes[j] * dp[points[j]]
            dp[i] = nex
            for j in range(m):
                if primes[j] * dp[points[j]] == nex:
                    points[j] += 1
        return dp[n - 1]


class EulerPhi:

    def __init__(self):
        return

    @staticmethod
    def euler_phi_with_prime_factor(n):  # faster!
        """the euler function returns the number of coprime with n that are less than or equal to n"""
        # Note that 1 and 1 are coprime, while prime numbers greater than 1 are not coprime with 1
        assert n >= 1
        if n <= 10 ** 6:
            lst = NumFactor().get_prime_factor(n)
        else:
            cnt = NumFactor().get_prime_with_pollard_rho(n)
            lst = [(p, cnt[p]) for p in cnt]
        ans = n
        for p, _ in lst:
            ans = ans // p * (p - 1)
        return ans

    @staticmethod
    def euler_phi_general(n):
        """the euler function returns the number of coprime with n that are less than or equal to n"""
        # Note that 1 and 1 are coprime, while prime numbers greater than 1 are not coprime with 1
        assert n >= 1
        ans = n
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                ans = ans // i * (i - 1)
                while n % i == 0:
                    n = n // i
        if n > 1:
            ans = ans // n * (n - 1)
        return int(ans)
