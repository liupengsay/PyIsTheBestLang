
import math
import random
import unittest
from itertools import combinations


"""
算法：数论、欧拉筛、线性筛、素数、欧拉函数、因子分解、素因子分解、进制转换、因数分解
功能：有时候数位DP类型题目可以使用N进制来求取
题目：


参考：OI WiKi（xx）
P1865 A % B Problem（https://www.luogu.com.cn/problem/P1865）通过线性筛素数后进行二分查询区间素数个数
P1748 H数（https://www.luogu.com.cn/problem/P1748）丑数可以使用堆模拟可以使用指针递增也可以使用容斥原理与二分进行计算
264. 丑数 II（https://leetcode.cn/problems/ugly-number-ii/）只含2、3、5质因数的第 n 个丑数
1201. 丑数 III（https://leetcode.cn/problems/ugly-number-iii/）只含特定因子数即能被其中一个数整除的第 n 个丑数
313. 超级丑数（https://leetcode.cn/problems/super-ugly-number/）只含某些特定质因数的第 n 个丑数
P1952 火星上的加法运算（https://www.luogu.com.cn/problem/P1952）N进制加法

P1592 互质（https://www.luogu.com.cn/problem/P1592）使用二分与容斥原理计算与 n 互质的第 k 个正整数
"""


class NumberTheory:
    def __init__(self):
        return

    @staticmethod
    def nth_super_ugly_number(n: int, primes) -> int:

        # 计算只含 primes 中的质因数的第 n 个丑数
        dp = [0] * (n + 1)
        m = len(primes)
        pointers = [0] * m
        nums = [1] * m

        for i in range(1, n + 1):
            min_num = min(nums)
            dp[i] = min_num
            for j in range(m):
                if nums[j] == min_num:
                    pointers[j] += 1
                    nums[j] = dp[pointers[j]] * primes[j]

        return dp[n]

    def gcd(self, x, y):
        # # 最大公约数
        if y == 0:
            return x
        else:
            return self.gcd(y, x % y)

    def lcm(self, x, y):
        # 最小公倍数
        return x * y // self.gcd(x, y)

    @staticmethod
    def get_factor_upper(n):
        # 使用素数筛类似的方法获取小于等于 n 的所有数除 1 与自身之外的所有因数
        factor = [[] for _ in range(n+1)]
        for i in range(2, n+1):
            x = 2
            while i*x <= n:
                factor[i*x].append(i)
                x += 1
        return factor

    @staticmethod
    def factorial_zero_count(num):
        # 阶乘后的后缀零个数
        cnt = 0
        while num > 0:
            cnt += num // 5
            num //= 5
        return cnt

    @staticmethod
    def get_k_bin_of_n(n, k):
        # 整数n的k进制计算（支持正数进制与负数进制）二进制、三进制、十六进制、负进制
        if n == 0:
            return [0]
        if k == 0:
            return []
        # 支持正负数
        pos = 1 if k > 0 else -1
        k = abs(k)
        lst = []
        while n:
            lst.append(n % k)
            n //= k
            n *= pos
        lst.reverse()
        # 最高支持三十六进制的表达
        # "0123456789" + "".join(chr(i+ord("A")) for i in range(26))
        return lst

    @staticmethod
    def k_bin_to_ten(k, st: str) -> str:
        # k进制字符转为 10 进制字符
        order = "0123456789" + "".join(chr(i + ord("a")) for i in range(26))
        ind = {w: i for i, w in enumerate(order)}
        m = len(st)
        ans = 0
        for i in range(m):
            ans *= k
            ans += ind[st[i]]
        return str(ans)

    def ten_to_k_bin(self, k, st: str) -> str:
        # 10 进制字符转为 k 进制字符
        order = "0123456789" + "".join(chr(i + ord("a")) for i in range(26))
        lst = self.get_k_bin_of_n(int(st), k)
        return "".join(order[i] for i in lst)

    @staticmethod
    def is_prime(num):
        # 判断数是否为质数
        if num <= 1:
            return False
        for i in range(2, min(int(math.sqrt(num)) + 2, num)):
            if num % i == 0:
                return False
        return True

    @staticmethod
    def is_prime1(x):
        if x == 1:
            return False
        for i in range(2, x):
            if x % i == 0:
                return False
        return True

    @staticmethod
    def is_prime2(x):
        if x == 1:
            return False
        for i in range(2, int(x ** 0.5) + 1):
            if x % i == 0:
                return False
        return True

    @staticmethod
    def is_prime3(x):
        if x == 1:
            return False
        if x == 2:
            return True
        elif x % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True

    @staticmethod
    def is_prime4(x):
        """https://zhuanlan.zhihu.com/p/107300262
        任何一个自然数，总可以表示成以下六种形式之一：6n，6n+1，6n+2，6n+3，6n+4，6n+5（n=0,1,2...）
        我们可以发现，除了2和3，只有形如6n+1和6n+5的数有可能是质数。
        且形如6n+1和6n+5的数如果不是质数，它们的因数也会含有形如6n+1或者6n+5的数，因此可以得到如下算法：
        """
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
    def rational_number_to_fraction(st):
        """
        # 有理数循环小数化为分数
        1.2727... = (27 / 99) + 1
        1.571428571428... = (571428 / 999999) + 1
        有n位循环 = (循环部分 / n位9) + 整数部分
        最后约简
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
        # 欧拉函数返回小于等于n的与n互质的个数
        # 注意1和1互质，而大于1的质数与1不互质
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
        # 欧拉线性筛素数
        # 说明：返回小于等于 n 的所有素数
        flag = [False for _ in range(n + 1)]
        prime_numbers = []
        for num in range(2, n + 1):
            if not flag[num]:
                prime_numbers.append(num)
            for prime in prime_numbers:
                if num * prime > n:
                    break
                flag[num * prime] = True
                if num % prime == 0:  # 这句是最有意思的地方  下面解释
                    break
        return prime_numbers

    @staticmethod
    def sieve_of_eratosthenes(n):  # 埃拉托色尼筛选法，返回小于等于n的素数
        primes = [True] * (n + 1)  # 范围0到n的列表
        p = 2  # 这是最小的素数
        while p * p <= n:  # 一直筛到sqrt(n)就行了
            if primes[p]:  # 如果没被筛，一定是素数
                for i in range(p * 2, n + 1, p):  # 筛掉它的倍数即可
                    primes[i] = False
            p += 1
        primes = [
            element for element in range(
                2, n + 1) if primes[element]]  # 得到所有小于等于n的素数
        return primes

    @staticmethod
    def get_all_factor(num):
        # 获取整数所有的因子包括1和它自己
        factor = set()
        for i in range(1, int(math.sqrt(num)) + 1):
            if num % i == 0:
                factor.add(i)
                factor.add(num // i)
        return sorted(list(factor))

    @staticmethod
    def get_prime_factor(num):
        # 质因数分解
        res = []
        for i in range(2, int(math.sqrt(num)) + 1):
            cnt = 0
            while num % i == 0:
                num //= i
                cnt += 1
            if cnt:
                res.append([i, cnt])
            if i > num:
                break
        if num != 1 or not res:
            res.append([num, 1])
        return res

    def get_prime_cnt(self, x, y):
        # P1592 互质
        # P2429 制杖题

        # 使用容斥原理计算 [1, y] 内与 x 互质的个数
        if x == 1:
            return y

        lst = self.get_prime_factor(x)
        prime = [p for p, _ in lst]
        m = len(lst)
        # 求与 x 不互质的数，再减去这部分数
        res = 0
        for i in range(1, m+1):
            for item in combinations(prime, i):
                cur = 1
                for num in item:
                    cur *= num
                res += (y//cur)*(-1)**(i+1)
        return y-res


class TestGeneral(unittest.TestCase):

    def test_prime_cnt(self):
        nt = NumberTheory()
        for _ in range(100):
            x = random.randint(1, 100)
            y = random.randint(1, 10000)
            cnt = 0
            for i in range(1, y+1):
                if math.gcd(i, x) == 1:
                    cnt += 1
            print(x, y, nt.get_prime_cnt(x, y), cnt)
            assert nt.get_prime_cnt(x, y) == cnt
        return

    def test_get_prime_factor(self):
        nt = NumberTheory()
        for i in range(1, 100000):
            res = nt.get_prime_factor(i)
            num = 0
            for val, c in res:
                num += val * c
            assert num == i
        return

    def test_euler_phi(self):
        nt = NumberTheory()
        assert nt.euler_phi(10**11 + 131) == 66666666752
        return

    def test_euler_shai(self):
        nt = NumberTheory()
        correctResult_30 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        euler_flag_primeResult_30 = nt.euler_flag_prime(30)
        assert correctResult_30 == euler_flag_primeResult_30
        assert len(nt.euler_flag_prime(10**6)) == 78498
        return

    def test_eratosthenes_shai(self):
        nt = NumberTheory()
        assert len(nt.sieve_of_eratosthenes(10**6)) == 78498
        return

    def test_factorial_zero_count(self):
        nt = NumberTheory()
        num = random.randint(1, 100)
        s = str(math.factorial(num))
        cnt = 0
        for w in s[::-1]:
            if w == "0":
                cnt += 1
            else:
                break
        assert nt.factorial_zero_count(num) == cnt
        return

    def test_get_k_bin_of_n(self):
        nt = NumberTheory()
        num = random.randint(1, 100)
        assert nt.get_k_bin_of_n(num, 2) == [int(w) for w in bin(num)[2:]]

        assert nt.get_k_bin_of_n(4, -2) == [1, 0, 0]
        return

    def test_rational_number_to_fraction(self):
        nt = NumberTheory()
        assert nt.rational_number_to_fraction("33") == [1, 3]
        return

    def test_is_prime(self):
        nt = NumberTheory()
        assert not nt.is_prime(1)
        assert nt.is_prime(5)
        assert not nt.is_prime(51)
        for _ in range(10):
            i = random.randint(1, 10**4)
            assert nt.is_prime(i) == nt.is_prime4(i)

        for _ in range(1):
            x = random.randint(10**8, 10**9)
            y = x + 10**6
            for num in range(x, y+1):
                nt.is_prime4(x)
        return

    def test_gcd_lcm(self):
        nt = NumberTheory()
        a, b = random.randint(1, 1000), random.randint(1, 1000)
        assert nt.gcd(a, b) == math.gcd(a, b)
        assert nt.lcm(a, b) == math.lcm(a, b)
        return

    def test_get_prime_factor(self):
        nt = NumberTheory()

        num = 2
        assert nt.get_prime_factor(num) == [[2, 1]]

        num = 1
        assert nt.get_prime_factor(num) == [[1, 1]]

        num = 2 * (3**2) * 7 * (11**3)
        assert nt.get_prime_factor(num) == [[2, 1], [3, 2], [7, 1], [11, 3]]
        return

    def test_get_factor(self):
        nt = NumberTheory()

        num = 1000
        ans = nt.get_factor_upper(num)
        for i in range(1, num+1):
            assert ans[i] == nt.get_all_factor(i)[1:-1]
        return


if __name__ == '__main__':
    unittest.main()
