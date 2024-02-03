"""
Algorithm：binary_search|interactive|game
Description：

====================================LeetCode====================================
xx（xxx）xxxxxxxxxxxxxxxxxxxx

=====================================LuoGu======================================
xx（xxx）xxxxxxxxxxxxxxxxxxxx

===================================CodeForces===================================
1479A（https://codeforces.com/problemset/problem/1479/A）interactive
1486C2（https://codeforces.com/problemset/problem/1486/C2）interactive
1503B（https://codeforces.com/problemset/problem/1503/B）interactive
1624F（https://codeforces.com/contest/1624/problem/F）binary_search|interactive
1713D（https://codeforces.com/contest/1713/problem/D）binary_search|interactive
1846F（https://codeforces.com/problemset/problem/1846/F）interactive
1697D（https://codeforces.com/contest/1697/problem/D）strictly_binary_search|interactive
1729E（https://codeforces.com/problemset/problem/1729/E）interactive
1762D（https://codeforces.com/problemset/problem/1762/D）interactive
1903E（https://codeforces.com/problemset/problem/1903/E）interactive
1918E（https://codeforces.com/contest/1918/problem/E）interactive|binary_search|quick_sort
1807E（https://codeforces.com/contest/1807/problem/E）interactive|binary_search

"""
import random
import sys

from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1713d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1713/problem/D
        tag: binary_search|interactive
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = list(range(1, 2 ** n + 1))
            while len(nums) >= 4:
                n = len(nums)
                ans = []
                for i in range(0, n, 4):
                    a, b, c, d = nums[i], nums[i + 1], nums[i + 2], nums[i + 3]
                    x = ac.inter_ask(["?", a, c])
                    if x == 1:
                        y = ac.inter_ask(["?", a, d])
                        if y == 1:
                            ans.append(a)
                        else:
                            ans.append(d)
                    elif x == 2:
                        y = ac.inter_ask(["?", b, c])
                        if y == 1:
                            ans.append(b)
                        else:
                            ans.append(c)
                    else:
                        y = ac.inter_ask(["?", b, d])
                        if y == 1:
                            ans.append(b)
                        else:
                            ans.append(d)

                nums = ans[:]
            if len(nums) == 1:
                ac.inter_out(["!", nums[0]])
            else:
                a, b = nums[:]
                x = ac.inter_ask(["?", a, b])
                ac.inter_out(["!", a if x == 1 else b])
        return

    @staticmethod
    def cf_1697d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1697/problem/D
        tag: strictly_binary_search|interactive
        """
        n = ac.read_int()
        ans = [""] * n
        dct = dict()

        ac.lst(["?", 1, 1])
        sys.stdout.flush()
        w = ac.read_str()
        ans[0] = w
        dct[w] = 0

        pre = 1
        for i in range(1, n):
            ac.lst(["?", 2, 1, i + 1])
            sys.stdout.flush()
            cur = ac.read_int()
            if cur > pre:
                ac.lst(["?", 1, i + 1])
                sys.stdout.flush()
                w = ac.read_str()
                ans[i] = w
                dct[w] = i
                pre = cur
            else:
                lst = sorted(dct.values())
                m = len(lst)
                low, high = 0, m - 1
                while low < high:
                    mid = low + (high - low + 1) // 2
                    target = len(set(ans[lst[mid]:i]))
                    ac.lst(["?", 2, lst[mid] + 1, i + 1])
                    sys.stdout.flush()
                    cur = ac.read_int()
                    if cur == target:
                        low = mid
                    else:
                        high = mid - 1
                ans[i] = ans[lst[high]]
                dct[ans[i]] = i
        ac.lst(["!", "".join(ans)])
        sys.stdout.flush()
        return

    @staticmethod
    def cf_1918e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1918/problem/E
        tag: binary_search|interactive|quick_sort
        """

        def query(x):
            ac.lst(["?", x + 1], True)
            cur = ac.read_str()
            if cur == ">":
                return 1
            elif cur == "<":
                return -1
            return 0

        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = [0] * n
            stack = [[1, n, list(range(n))]]
            while stack:
                left, right, ind = stack.pop()
                mid = ind[random.randint(0, len(ind) - 1)]
                while query(mid):
                    continue

                smaller = []
                bigger = []
                for i in ind:
                    if i == mid:
                        continue
                    if query(i) == 1:
                        bigger.append(i)
                    else:
                        smaller.append(i)
                    query(mid)
                nums[mid] = left + len(smaller)
                if left < nums[mid]:
                    stack.append([left, nums[mid] - 1, smaller])
                if nums[mid] < right:
                    stack.append([nums[mid] + 1, right, bigger])
            ac.lst(["!"] + nums, True)
        return
