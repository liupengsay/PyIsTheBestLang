"""
Algorithm：use xor of random seed as key of mapping
Ability：speed up and avoid hash crush
Reference：https://judge.yosupo.jp/problem/associative_array


===================================CodeForces===================================
1665B（https://codeforces.com/contest/1665/problem/B）hash|xor_random_seed
1702C（https://codeforces.com/contest/1702/problem/C）hash|xor_random_seed


================================Library Checker================================
Associative Array（https://judge.yosupo.jp/problem/associative_array）hash|xor_random_seed


"""
from collections import Counter

from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def libc_aa(ac=FastIO()):
        """template question of associative array"""
        pre = dict()
        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            if lst[0] == 0:
                k, v = lst[1:]
                pre[k ^ ac.random_seed] = v
            else:
                ac.st(pre.get(lst[1] ^ ac.random_seed, 0))
        return

    @staticmethod
    def cf_1655b(ac=FastIO()):
        def solve():
            n = ac.read_int()
            nums = ac.read_list_ints()
            cur = max(Counter([x ^ ac.random_seed for x in nums]).values())
            ans = 0
            while cur < n:
                ans += 1
                other = cur
                x = ac.min(n - cur, other)
                ans += x
                cur += x
                other -= x
            ac.st(ans)
            return

        for _ in range(ac.read_int()):
            solve()
        return

    @staticmethod
    def cf_1702c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1702/problem/C
        tag: hash|xor_random_seed
        """
        def solve():
            ac.read_str()
            n, k = ac.read_list_ints()
            u = ac.read_list_ints()
            start = dict()
            end = dict()
            for i, x in enumerate(u):
                if x ^ ac.random_seed not in start:
                    start[x ^ ac.random_seed] = i
                end[x ^ ac.random_seed] = i
            for _ in range(k):
                a, b = ac.read_list_ints()
                if a ^ ac.random_seed in start and b ^ ac.random_seed in end and start[a ^ ac.random_seed] < end[b ^ ac.random_seed]:
                    ac.st("YES")
                else:
                    ac.st("NO")
            return

        for _ in range(ac.read_int()):
            solve()
        return