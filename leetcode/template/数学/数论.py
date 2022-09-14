
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
    return len(primes)


def get_k_bin_of_n(n: int, k: int) -> int:
    lst = []
    while n:
        lst.append(n % k)
        n //= k
    return lst[::-1]


def test_sieve_of_eratosthenes():
    print(sieve_of_eratosthenes(10**4))
    return


def test_get_k_bin_of_n():
    for i in range(1, 100, 10000):
        assert [int(w) for w in bin(i)[2:]] == get_k_bin_of_n(i, 2)
    return


if __name__ == '__main__':
    test_sieve_of_eratosthenes()
