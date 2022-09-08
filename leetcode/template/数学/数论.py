

def get_n_bin(n, b):
    # 整数n的b进制计算
    def check(cur, num):
        if cur <= 1:
            return [[0, cur]]
        cnt = 0
        while cur:
            cur //= num
            cnt += 1
        return [[cnt, 1]] + check(cur - num**cnt)

    lst = check(n, b)
    tmp = [0] * (lst[0][0] + 1)
    for i, val in lst:
        tmp[-i - 1] = val
    return tmp


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


def test_sieve_of_eratosthenes():
    print(sieve_of_eratosthenes(10**4))
    return


if __name__ == '__main__':
    test_sieve_of_eratosthenes()
