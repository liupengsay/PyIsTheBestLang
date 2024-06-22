from collections import defaultdict


class AllFactorCnt:
    def __init__(self, n):
        self.n = n
        self.all_factor_cnt = [0, 1] + [2 for _ in range(2, n + 1)]
        for i in range(2, self.n + 1):
            for j in range(i * i, self.n + 1, i):
                self.all_factor_cnt[j] += 2 if j != i * i else 1
        return


class PrimeFactor:
    def __init__(self, n):
        self.n = n
        # calculate the minimum prime factor for all numbers from 1 to self.n
        self.min_prime = [0] * (self.n + 1)
        self.min_prime[1] = 1
        # determine whether all numbers from 1 to self.n are prime numbers
        self.prime_factor = [[] for _ in range(self.n + 1)]
        self.prime_factor_cnt = [0]*(self.n+1)
        # calculate all factors of all numbers from 1 to self.n, including 1 and the number itself
        self.all_factor = [[], [1]] + [[1, i] for i in range(2, self.n + 1)]
        self.build()
        return

    def build(self):

        # complexity is O(nlogn)
        for i in range(2, self.n + 1):
            if not self.min_prime[i]:
                self.min_prime[i] = i
                for j in range(i * i, self.n + 1, i):
                    if not self.min_prime[j]:
                        self.min_prime[j] = i

        for num in range(2, self.n + 1):
            pre = num // self.min_prime[num]
            self.prime_factor_cnt[num] = self.prime_factor_cnt[pre] + int(self.min_prime[num] != self.min_prime[pre])

        # complexity is O(nlogn)
        for num in range(2, self.n + 1):
            i = num
            while num > 1:
                p = self.min_prime[num]
                cnt = 0
                while num % p == 0:
                    num //= p
                    cnt += 1
                self.prime_factor[i].append((p, cnt))


        # complexity is O(nlogn)
        for i in range(2, self.n + 1):
            for j in range(i * i, self.n + 1, i):
                self.all_factor[j].append(i)
                if j > i * i:
                    self.all_factor[j].append(j // i)
        for i in range(self.n + 1):
            self.all_factor[i].sort()
        return

    def comb(self, a, b):
        # Use prime factor decomposition to solve the values of combinatorial mathematics
        # and prime factor decomposition O ((a+b) log (a+b))
        cnt = defaultdict(int)
        for i in range(1, a + 1):  # a!
            for num, y in self.prime_factor[i]:
                cnt[num] += y
        for i in range(1, b + 1):  # b!
            for num, y in self.prime_factor[i]:
                cnt[num] -= y
        for i in range(1, a - b + 1):  # (a-b)!
            for num, y in self.prime_factor[i]:
                cnt[num] -= y
        ans = 1
        for w in cnt:
            ans *= w ** cnt[w]
        return ans

    def get_prime_numbers(self):
        return [i for i in range(2, self.n + 1) if self.min_prime[i] == 0]
