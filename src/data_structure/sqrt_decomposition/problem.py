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

from src.data_structure.sqrt_decomposition.template import BlockSize
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
        tag: block_query|counter|sqrt_decomposition|offline_query|classical
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
        tag: sqrt_decomposition|offline_query|math
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
        tag: sqrt_decomposition|offline_query|xor_pair|counter
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

    @staticmethod
    def cf_1806e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1806/E
        tag: sqrt_decomposition
        """
        n, q = ac.read_list_ints()
        nums = ac.read_list_ints()
        parent = [-1] + ac.read_list_ints_minus_one()
        fxx = [0] * n
        depth = [0] * n
        cnt = [0] * n
        for i in range(n):
            fxx[i] += nums[i] * nums[i]
            if i:
                fxx[i] += fxx[parent[i]]
                depth[i] = depth[parent[i]] + 1
            cnt[depth[i]] += 1

        size = 100
        dct = dict()
        for _ in range(q):
            x, y = ac.read_list_ints_minus_one()
            if x > y:
                y, x = x, y
            stack = [(x, y)]
            while x != y and x * n + y not in dct:
                x, y = parent[x], parent[y]
                stack.append((x, y))
            m = len(stack)
            val = dct[x * n + y] if x != y else fxx[x]
            if x != y and cnt[depth[x]] <= size:
                dct[x * n + y] = val
            for i in range(m - 2, -1, -1):
                x, y = stack[i]
                val += nums[x] * nums[y]
                if cnt[depth[x]] <= size:
                    dct[x * n + y] = val
            ac.st(val)
        return

    @staticmethod
    def lg_p3396(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3396
        tag: sqrt_decomposition
        """
        n, q = ac.read_list_ints()
        nums = ac.read_list_ints()
        size = ac.min(n - 1, 10 ** 7 // n)  # 100
        pre = [[]]
        for p in range(1, size + 1):
            lst = [0] * p
            for i in range(n):
                lst[(i + 1) % p] += nums[i]
            pre.append(lst[:])
        for _ in range(q):
            lst = ac.read_list_strs()
            if lst[0] == "A":
                x, y = [int(w) for w in lst[1:]]
                if x <= size:
                    ac.st(pre[x][y])
                else:
                    w = y if y else x
                    ans = 0
                    while w <= n:
                        ans += nums[w - 1]
                        w += x
                    ac.st(ans)
            else:
                x, y = [int(w) for w in lst[1:]]
                num = nums[x - 1]
                nums[x - 1] = y
                for p in range(1, size + 1):
                    pre[p][x % p] -= num
                    pre[p][x % p] += y
        return

    @staticmethod
    def cf_103d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/103/problem/D
        tag: sqrt_decomposition
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        size = ac.min(n, 10 ** 7 // (3 * 10 ** 5))
        q = ac.read_int()
        queries = [ac.read_list_ints() for _ in range(q)]
        ind = [[] for _ in range(size + 1)]
        for i in range(q):
            t, k = queries[i]
            if k <= size:
                ind[k].append(i)
        ans = [-1] * q
        for k in range(1, size + 1):
            if ind[k]:
                cur = [[0] for _ in range(k)]
                for i in range(n):
                    cur[(i + 1) % k].append(cur[(i + 1) % k][-1] + nums[i])
                for x in ind[k]:
                    t, k = queries[x]
                    t -= 1
                    j = (t + 1) % k
                    ans[x] = cur[j][-1] - cur[j][t // k]
                del cur
        for i in range(q):
            t, k = queries[i]
            if k > size:
                t -= 1
                res = 0
                for x in range(t, n, k):
                    res += nums[x]
                ac.st(res)
            else:
                ac.st(ans[i])
        return
