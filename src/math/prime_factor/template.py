from collections import defaultdict


class AllFactor:
    def __init__(self, n):
        self.n = n
        self.min_prime = []
        self.build_min_prime()
        return

    def build_min_prime(self):
        self.min_prime = [0] * (self.n + 1)
        self.min_prime[1] = 1
        for i in range(2, self.n + 1):
            if not self.min_prime[i]:
                self.min_prime[i] = i
                for j in range(i * i, self.n + 1, i):
                    if not self.min_prime[j]:
                        self.min_prime[j] = i
        return

    def get_all_factor(self, num):
        all_factor = [1]
        while num > 1:
            p = self.min_prime[num]
            cnt = 0
            while num % p == 0:
                cnt += 1
                num //= p
            nex = all_factor[:]
            val = 1
            for i in range(1, cnt + 1):
                val *= p
                nex.extend([x * val for x in all_factor])
            all_factor = nex[:]
        return all_factor

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
        self.prime_factor_cnt = [0] * (self.n + 1)
        self.prime_factor_mi_cnt = [0] * (self.n + 1)
        # calculate all factors of all numbers from 1 to self.n, including 1 and the number itself
        self.all_factor = [[], [1]] + [[1, i] for i in range(2, self.n + 1)]
        self.euler_phi = list(range(self.n + 1))
        self.build()

        return

    def build(self):

        # complexity is O(nlogn)
        for i in range(2, self.n + 1):
            if not self.min_prime[i]:
                self.min_prime[i] = i
                # i is prime <> self.min_prime[i]==i
                for j in range(i * i, self.n + 1, i):
                    if not self.min_prime[j]:
                        self.min_prime[j] = i

        for num in range(2, self.n + 1):
            pre = num // self.min_prime[num]
            self.prime_factor_cnt[num] = self.prime_factor_cnt[pre] + int(self.min_prime[num] != self.min_prime[pre])
            cur = num
            p = self.min_prime[cur]
            cnt = 0
            while cur % p == 0:
                cnt += 1
                cur //= p
            self.prime_factor_mi_cnt[num] = self.prime_factor_mi_cnt[cur] + cnt

        # complexity is O(nlogn)
        for num in range(2, self.n + 1):
            i = num
            phi = num
            while num > 1:
                p = self.min_prime[num]
                cnt = 0
                while num % p == 0:
                    num //= p
                    cnt += 1
                self.prime_factor[i].append((p, cnt))
                phi = phi // p * (p - 1)
            self.euler_phi[i] = phi

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
        return [i for i in range(2, self.n + 1) if self.min_prime[i] == i]


class PrimeFactor2:
    def __init__(self, n):
        self.n = n
        self.min_prime = []
        self.prime_factor = []
        self.prime_factor_cnt = []
        self.prime_factor_mi_cnt = []
        self.all_factor = []
        self.euler_phi = []
        self.mobius = []

        self.build_min_prime()
        self.build_prime_factor()

        self.build_all_factor()
        self.build_prime_factor_cnt()
        return

    def build_min_prime(self):
        self.min_prime = [0] * (self.n + 1)
        self.min_prime[1] = 1
        for i in range(2, self.n + 1):
            if not self.min_prime[i]:
                self.min_prime[i] = i
                for j in range(i * i, self.n + 1, i):
                    if not self.min_prime[j]:
                        self.min_prime[j] = i
        return

    def build_prime_factor(self):
        self.mobius = [1] * (self.n + 1)
        self.prime_factor = [[] for _ in range(self.n + 1)]
        self.euler_phi = list(range(self.n + 1))
        for num in range(2, self.n + 1):
            i = num
            phi = num
            flag = 0
            self.mobius[i] *= -1
            while num > 1:
                p = self.min_prime[num]
                cnt = 0
                while num % p == 0:
                    num //= p
                    cnt += 1
                if cnt > 1:
                    flag = 1
                self.mobius[i] *= -1
                self.prime_factor[i].append((p, cnt))
                phi = phi // p * (p - 1)
            if flag:
                self.mobius[i] = 0
        return

    def build_all_factor(self):
        self.all_factor = [[]] + [[1] for _ in range(1, self.n + 1)]
        for i in range(2, self.n + 1):
            for j in range(i * 2, self.n + 1, i):
                self.all_factor[j].append(i)
        return

    def build_all_factor2(self):
        self.all_factor = [[], [1]] + [[1, i] for i in range(2, self.n + 1)]
        for i in range(2, self.n + 1):
            for j in range(i * i, self.n + 1, i):
                self.all_factor[j].append(i)
                if j > i * i:
                    self.all_factor[j].append(j // i)
        for i in range(self.n + 1):
            self.all_factor[i].sort()
        return

    def build_prime_factor_cnt(self):
        self.prime_factor_cnt = [0] * (self.n + 1)
        self.prime_factor_mi_cnt = [0] * (self.n + 1)
        for num in range(2, self.n + 1):
            pre = num // self.min_prime[num]
            self.prime_factor_cnt[num] = self.prime_factor_cnt[pre] + int(self.min_prime[num] != self.min_prime[pre])
            cur = num
            p = self.min_prime[cur]
            cnt = 0
            while cur % p == 0:
                cnt += 1
                cur //= p
            self.prime_factor_mi_cnt[num] = self.prime_factor_mi_cnt[cur] + cnt
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
        return [i for i in range(2, self.n + 1) if self.min_prime[i] == i]

    def get_rad_factor(self, num):
        prime = self.get_prime_factor(num)
        rad_factor = [1]
        for p in prime:
            length = len(rad_factor)
            for _ in range(length):
                rad_factor.append(rad_factor[-length] * (-p))
        return rad_factor

    def get_rad_factor2(self, num):
        prime = self.get_prime_factor(num)
        m = len(prime)
        rad_factor = []
        dp = [1] * (1 << m)
        cnt = [0] * (1 << m)
        ind = {1 << i: i for i in range(m)}
        for i in range(1, 1 << m):
            cnt[i] = 1 + cnt[i & (-i) ^ i]
            dp[i] = prime[ind[i & (-i)]] * dp[i & (-i) ^ i]
            x = dp[i]
            c = cnt[i]
            rad_factor.append(x if c % 2 == 0 else -x)
        return rad_factor

    def get_rad_factor3(self, num):
        prime = self.get_prime_factor(num)
        rad_factor = [1]
        for p in prime:
            length = len(rad_factor)
            for _ in range(length):
                rad_factor.append(rad_factor[-length] * p)
        return rad_factor

    def get_prime_factor(self, num):
        prime = []
        while num > 1:
            prime.append(self.min_prime[num])
            while num % prime[-1] == 0:
                num //= prime[-1]
        return prime


class RadFactor:
    def __init__(self, n):
        self.n = n
        self.min_prime = [0] * (self.n + 1)
        self.min_prime[1] = 1
        self.build()
        return

    def build(self):
        for i in range(2, self.n + 1):
            if not self.min_prime[i]:
                self.min_prime[i] = i
                for j in range(i * i, self.n + 1, i):
                    if not self.min_prime[j]:
                        self.min_prime[j] = i
        return

    def get_rad_factor(self, num):
        prime = self.get_prime_factor(num)
        rad_factor = [1]
        for p in prime:
            length = len(rad_factor)
            for _ in range(length):
                rad_factor.append(rad_factor[-length] * (-p))
        return rad_factor

    def get_rad_factor2(self, num):
        prime = self.get_prime_factor(num)
        m = len(prime)
        rad_factor = []
        for i in range(1 << m):
            c = 0
            x = 1
            for j in range(m):
                if (i >> j) & 1:
                    c += 1
                    x *= prime[j]
            rad_factor.append(x if c % 2 == 0 else -x)
        return rad_factor

    def get_prime_factor(self, num):
        prime = []
        while num > 1:
            prime.append(self.min_prime[num])
            while num % prime[-1] == 0:
                num //= prime[-1]
        return prime
