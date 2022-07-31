from typing import List


import math


def euler_phi(n):
    ans = n
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            ans = ans // i * (i - 1)
            while n % i == 0:
                n = n / i
    if n > 1:
        ans = ans // n * (n - 1)
    return int(ans)

def Eratosthenes(n):
    p = 0
    is_prime = [True]*(n+1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, n + 1):
        if is_prime[i]:
            is_prime[p] = i
            p = p + 1
            if i * i <= n:
                j = i * i
                while j <= n:
                    is_prime[j] = False
                    j = j + i
    return p

def sieve_of_eratosthenes(n):#埃拉托色尼筛选法，返回少于n的素数
    primes = [True] * (n+1)#范围0到n的列表
    p = 2#这是最小的素数
    while p * p <= n:#一直筛到sqrt(n)就行了
        if primes[p]:#如果没被筛，一定是素数
            for i in range(p * 2, n + 1, p):#筛掉它的倍数即可
                primes[i] = False
        p += 1
    primes = [element for element in range(2, n+1) if primes[element]]#得到所有少于n的素数
    return len(primes)

print(euler_phi(108))
print(euler_phi(1))
print(euler_phi(2))
print(euler_phi(10))
print(Eratosthenes(108))
print(Eratosthenes(1))
print(Eratosthenes(2))
print(Eratosthenes(10))

print(sieve_of_eratosthenes(108))
print(sieve_of_eratosthenes(1))
print(sieve_of_eratosthenes(2))
print(sieve_of_eratosthenes(10))