


def divPrime(num):
    lt = []
    while num != 1:
        for i in range(2, int(num+1)):
            if num % i == 0:  # i是num的一个质因数
                lt.append(i)
                num = num / i # 将num除以i，剩下的部分继续分解
                break
    return lt
divPrime(18)


def sieve_of_eratosthenes(n):  # 埃拉托色尼筛选法，返回少于n的素数
    primes = [True] * (n + 1)  # 范围0到n的列表
    p = 2  # 这是最小的素数
    while p * p <= n:  # 一直筛到sqrt(n)就行了
        if primes[p]:  # 如果没被筛，一定是素数
            for i in range(p * 2, n + 1, p):  # 筛掉它的倍数即可
                primes[i] = False
        p += 1
    primes = [element for element in range(
        2, n + 1) if primes[element]]  # 得到所有少于n的素数
    print(len(primes))
    return primes

print(sieve_of_eratosthenes(10000))