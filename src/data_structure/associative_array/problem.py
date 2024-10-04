"""
Algorithm：use xor of random seed as key of mapping
Ability：speed up and avoid hash crush
Reference：https://judge.yosupo.jp/problem/associative_array


===================================CodeForces===================================
1665B（https://codeforces.com/contest/1665/problem/B）hash|xor_random_seed
1702C（https://codeforces.com/contest/1702/problem/C）hash|xor_random_seed
1676F（https://codeforces.com/contest/1676/problem/F）hash|dp|sort
776C（https://codeforces.com/problemset/problem/776/C）prefix_sum|hash|random_seed|random_xor
1188B（https://codeforces.com/problemset/problem/1188/B）hash|math|classical

================================Library Checker================================
1（https://judge.yosupo.jp/problem/associative_array）hash|xor_random_seed


"""
from collections import Counter

from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def library_check_1(ac=FastIO()):
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
        """
        url: https://codeforces.com/contest/1665/problem/B
        tag: hash|xor_random_seed
        """
        for _ in range(ac.read_int()):
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
                if (a ^ ac.random_seed in start
                        and b ^ ac.random_seed in end
                        and start[a ^ ac.random_seed] < end[b ^ ac.random_seed]):
                    ac.yes()
                else:
                    ac.no()
            return

        for _ in range(ac.read_int()):
            solve()
        return

    @staticmethod
    def cf_776c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/776/C
        tag: prefix_sum|hash|random_seed|random_xor
        """
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        ceil = 10 ** 15
        val = []
        x = 1
        for _ in range(64):
            val.append(x)
            x *= k
            if abs(x) > ceil:
                break
        val = set(val)
        ans = x = 0
        pre = dict()
        ac.get_random_seed()
        pre[x ^ ac.random_seed] = 1
        for num in nums:
            x += num
            for v in val:
                ans += pre.get((x - v) ^ ac.random_seed, 0)
            pre[x ^ ac.random_seed] = pre.get(x ^ ac.random_seed, 0) + 1
        ac.st(ans)
        return

    @staticmethod
    def cf_1188b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1188/B
        tag: hash|math|classical
        """
        n, p, k = ac.read_list_ints()
        cnt = dict()
        ac.get_random_seed()
        nums = ac.read_list_ints()
        ans = 0
        for num in nums:
            cur = ((num ** 4 - k * num) % p) ^ ac.random_seed
            ans += cnt.get(cur, 0)
            cnt[cur] = cnt.get(cur, 0) + 1
        ac.st(ans)
        return
