from collections import defaultdict


class PrimeFactor:
    def __init__(self, ceil):
        self.ceil = ceil + 100
        # calculate the minimum prime factor for all numbers from 1 to self.ceil
        self.min_prime = [0] * (self.ceil + 1)
        # determine whether all numbers from 1 to self.ceil are prime numbers
        self.is_prime = [0] * (self.ceil + 1)
        # calculate the prime factorization of all numbers from 1 to self.ceil
        self.prime_factor = [[] for _ in range(self.ceil + 1)]
        # calculate all factors of all numbers from 1 to self.ceil, including 1 and the number itself
        self.all_factor = [[1] for _ in range(self.ceil + 1)]
        self.build()
        return

    def build(self):

        # complexity is O(nlogn)
        for i in range(2, self.ceil + 1):
            if not self.min_prime[i]:
                self.is_prime[i] = 1
                self.min_prime[i] = i
                for j in range(i * i, self.ceil + 1, i):
                    if not self.min_prime[j]:
                        self.min_prime[j] = i

        # complexity is O(nlogn)
        for num in range(2, self.ceil + 1):
            i = num
            while num > 1:
                p = self.min_prime[num]
                cnt = 0
                while num % p == 0:
                    num //= p
                    cnt += 1
                self.prime_factor[i].append([p, cnt])

        # complexity is O(nlogn)
        for i in range(2, self.ceil + 1):
            x = 1
            while x * i <= self.ceil:
                self.all_factor[x * i].append(i)
                x += 1
        return

    def comb(self, n, m):
        # Use prime factor decomposition to solve the values of combinatorial mathematics
        # and prime factor decomposition O ((n+m) log (n+m))
        cnt = defaultdict(int)
        for i in range(1, n + 1):  # n!
            for num, y in self.prime_factor[i]:
                cnt[num] += y
        for i in range(1, m + 1):  # m!
            for num, y in self.prime_factor[i]:
                cnt[num] -= y
        for i in range(1, n - m + 1):  # (n-m)!
            for num, y in self.prime_factor[i]:
                cnt[num] -= y
        ans = 1
        for w in cnt:
            ans *= w ** cnt[w]
        return ans
