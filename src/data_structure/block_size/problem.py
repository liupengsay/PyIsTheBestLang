"""
Algorithm：block_query|offline_query|sort|two_pointers
Description：sort the query interval into blocks and alternate between moving two pointers dynamically maintain query values

====================================LeetCode====================================
1157（https://leetcode.cn/problems/online-majority-element-in-subarray/description/）range_super_mode|CF1514D|random_guess|binary_search|bit_operation|segment_tree

=====================================LuoGu======================================

===================================CodeForces===================================
220B（https://codeforces.com/contest/220/problem/B）block_query|counter
86D（https://codeforces.com/contest/86/problem/D）block_query|math
617E（https://codeforces.com/contest/617/problem/E）block_query|xor_pair|counter
1514D（https://codeforces.com/contest/1514/problem/D）range_super_mode|random_guess|binary_search|bit_operation|segment_tree

====================================AtCoder=====================================
ABC132F（https://atcoder.jp/contests/abc132/tasks/abc132_f）block_query|counter|dp|prefix_sum


"""
import bisect
from collections import defaultdict, Counter
from itertools import accumulate
from operator import xor

from src.data_structure.block_size.template import BlockSize
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def abc_132f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc132/tasks/abc132_f
        tag: block_query|counter|dp|prefix_sum
        """
        mod = 10 ** 9 + 7
        n, k = ac.read_list_ints()
        cnt, _ = BlockSize().get_divisor_split(n)
        m = len(cnt)
        dp = cnt[:]
        for _ in range(k - 1):
            pre = list(ac.accumulate(dp)[1:])[::-1]
            dp = [(cnt[i] * pre[i]) % mod for i in range(m)]
        ac.st(sum(dp) % mod)
        return

    @staticmethod
    def cf1514_d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1514/problem/D
        tag: range_super_mode|random_guess|binary_search|bit_operation|segment_tree
        """
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        cnt = [0] * (n + 1)
        freq = [0] * (n + 1)
        size = int(n ** 0.5) + 500

        queries = [[] for _ in range(size)]

        for i in range(m):
            a, b = ac.read_list_ints_minus_one()
            queries[b // size].append([a, b, i])

        def update(num, p):
            nonlocal cnt, ceil
            if p == 1:
                cnt[num] += 1
                freq[cnt[num] - 1] -= 1
                freq[cnt[num]] += 1
                if cnt[num] > ceil:
                    ceil = cnt[num]
            else:
                cnt[num] += -1
                freq[cnt[num] + 1] -= 1
                freq[cnt[num]] += 1
                if not freq[cnt[num] + 1] and ceil == cnt[num] + 1:
                    ceil = cnt[num]
            return

        ans = [0] * m
        x = y = 0
        cnt[nums[0]] = 1
        freq[1] = 1
        ceil = 1
        for i in range(size):
            if i % 2:
                queries[i].sort(key=lambda it: -it[0])
            else:
                queries[i].sort(key=lambda it: it[0])
            for a, b, j in queries[i]:
                while y > b:
                    update(nums[y], -1)
                    y -= 1
                while y < b:
                    y += 1
                    update(nums[y], 1)
                while x > a:
                    x -= 1
                    update(nums[x], 1)
                while x < a:
                    update(nums[x], -1)
                    x += 1
                if ceil * 2 <= b - a + 1:
                    ans[j] = 1
                else:
                    ans[j] = ceil - (b - a + 1 - ceil + 1) + 1
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def cf1514_d_2(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1514/problem/D
        tag: range_super_mode|random_guess|binary_search|bit_operation|segment_tree|random_seed
        """
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        ind = [[] for _ in range(1 << 20)]
        cnt = [0] * 20
        pre = [cnt[:]]
        for i, num in enumerate(nums):
            ind[num].append(i)
            for j in range(20):
                if num & (1 << j):
                    cnt[j] += 1
            pre.append(cnt[:])
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            num = 0
            length = y - x + 1
            for j in range(20):
                if pre[y + 1][j] - pre[x][j] >= length / 2:
                    num |= 1 << j

            ceil = bisect.bisect_right(ind[num], y) - bisect.bisect_left(ind[num], x)
            if ceil * 2 <= length:
                ac.st(1)
            else:
                ac.st(ceil - (length - ceil + 1) + 1)
        return

    @staticmethod
    def cf_220b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/220/problem/B
        tag: block_query|counter|block_size|offline_query|classical
        """

        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        size = int(n ** 0.5) + 1

        queries = [[] for _ in range(size)]
        for i in range(m):
            a, b = ac.read_list_ints_minus_one()
            queries[b // size].append([a, b, i])

        def update(num, p):
            nonlocal cur, cnt
            if num == cnt[num]:
                cur -= 1
            cnt[num] += p
            if num == cnt[num]:
                cur += 1
            return

        cur = 0
        ans = [0] * m
        x = y = 0
        cnt = defaultdict(int)
        cnt[nums[0]] = 1
        if nums[0] == 1:
            cur += 1
        for i in range(size):
            if i % 2:
                queries[i].sort(key=lambda it: -it[0])
            else:
                queries[i].sort(key=lambda it: it[0])
            for a, b, j in queries[i]:
                while y > b:
                    update(nums[y], -1)
                    y -= 1
                while y < b:
                    y += 1
                    update(nums[y], 1)
                while x > a:
                    x -= 1
                    update(nums[x], 1)
                while x < a:
                    update(nums[x], -1)
                    x += 1
                ans[j] = cur
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def cf_86d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/86/problem/D
        tag: block_size|offline_query|math
        """

        n, t = ac.read_list_ints()
        nums = ac.read_list_ints()
        size = int(n ** 0.5) + 1

        queries = [[] for _ in range(t)]
        for i in range(t):
            a, b = ac.read_list_ints_minus_one()
            queries[b // size].append([a, b, i])

        def update(p, z):
            nonlocal cur
            num = nums[p]
            cur -= cnt[num] * cnt[num] * num
            cnt[num] += z
            cur += cnt[num] * cnt[num] * num
            return

        cnt = [0] * (10 ** 6 + 1)
        x = y = 0
        ans = [0] * t
        cnt = Counter()
        cur = nums[0]
        cnt[nums[0]] = 1
        for i in range(size):
            if i % 2:
                queries[i].sort(key=lambda it: -it[0])
            else:
                queries[i].sort(key=lambda it: it[0])

            for a, b, j in queries[i]:
                while y > b:
                    update(y, -1)
                    y -= 1
                while y < b:
                    y += 1
                    update(y, 1)

                while x > a:
                    x -= 1
                    update(x, 1)
                while x < a:
                    update(x, -1)
                    x += 1
                ans[j] = cur
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def cf_617e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/617/problem/E
        tag: block_size|offline_query|xor_pair|counter
        """

        n, m, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        pre = list(accumulate(nums, xor, initial=0))

        size = int(n ** 0.5) + 1
        queries = [[] for _ in range(size)]
        for i in range(m):
            a, b = ac.read_list_ints()
            queries[b // size].append([a, b, i])

        def update(num, p):
            nonlocal cur
            if p == 1:
                cur += dct[num ^ k]
                dct[num] += 1
            else:
                dct[num] -= 1
                cur -= dct[num ^ k]
            return

        dct = [0] * (2 * 10 ** 6 + 1)
        x = y = 0
        ans = [0] * m
        dct[pre[0]] += 1
        cur = 0
        for i in range(size):
            if i % 2:
                queries[i].sort(key=lambda it: -it[0])
            else:
                queries[i].sort(key=lambda it: it[0])
            for a, b, j in queries[i]:
                a -= 1
                while y > b:
                    update(pre[y], -1)
                    y -= 1
                while y < b:
                    y += 1
                    update(pre[y], 1)
                while x > a:
                    x -= 1
                    update(pre[x], 1)
                while x < a:
                    update(pre[x], -1)
                    x += 1
                ans[j] = cur
        for a in ans:
            ac.st(a)
        return
