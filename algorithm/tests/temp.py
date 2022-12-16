import random
from math import log, log10
from collections import Counter

def gcd(x, y):
    return x if y == 0 else gcd(y, x % y)

def fpow(a, x, n):
	ans = 1
	while x > 0:
		if x & 1:
			ans = ans * a % n
		a = a * a % n
		x >>= 1
	return ans

# there change the times of Rabin-Miller
TIMES = 10
def is_prime(n):
    def check(a, n, x, t):
        ret = fpow(a, x, n)
        last = ret
        for i in range(0, t):
            ret = ret * ret % n
            if ret == 1 and last != 1 and last != n - 1:
                return True
            last = ret
        if ret != 1:
            return True
        return False

    if not isinstance(n, int):
        raise TypeError(str(n) + ' is not an integer!')
    if n <= 0:
        raise ValueError('%d <= 0' % n)
    if n in {2, 3, 5, 7, 11}:
        return True
    for i in {2, 3, 5, 7, 11}:
        if n % i == 0:
            return False
    x = n - 1
    t = 0
    while not x & 1:
        x >>= 1
        t += 1
    for i in range(0, TIMES):
        a = random.randint(1, n - 2)
        if check(a, n, x, t):
            return False
    return True

def pollard_rho_2(n, c):
    x = random.randint(0, n)
    i, k, y = 1, 2, x
    while True:
        i += 1
        x = (x * x) % n + c
        d = gcd(y - x, n)
        if d != 1 and d != n:
            return d
        if y == x:
            return n
        if i == k:
            y = x
            k <<= 1

def pollard_rho_1(n):
    if not isinstance(n, int):
        raise TypeError(str(n) + ' is not an integer!')
    if n == 1:
        return None
    if is_prime(n):
        return [n]
    ans = []
    p = n
    while p >= n:
        p = pollard_rho_2(p, random.randint(1, n - 1))
    ans.extend(pollard_rho_1(p))
    ans.extend(pollard_rho_1(n // p))
    return ans

def factorization(n):
    return Counter(pollard_rho_1(n))

if __name__ == '__main__':
    n = int(input())
    print('len:', len(str(n)))
    print(factorization(n))