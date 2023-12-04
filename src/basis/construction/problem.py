"""

算法：构造
功能：通常需要根据题目要求，进行贪心排序之后的构造出合理可行的答案
题目：xx（xx）

====================================LeetCode====================================
280（https://leetcode.com/problems/wiggle-sort/）按照降序先赋给奇数索引再给偶数索引
2663（https://leetcode.com/problems/lexicographically-smallest-beautiful-string/）贪心构造不含任何回文子串的字典序最小的字符串
1982（https://leetcode.com/problems/find-array-given-subset-sums/）经典构造
1253（https://leetcode.com/problems/reconstruct-a-2-row-binary-matrix/）经典构造贪心思维
2573（https://leetcode.com/problems/find-the-string-with-lcp/）经典LCP构造与验证，类似并查集思想

=====================================LuoGu======================================
8846（https://www.luogu.com.cn/problem/P8846）厘清题意进行贪心构造
2902（https://www.luogu.com.cn/problem/P2902）构造不同颜色对
5823（https://www.luogu.com.cn/problem/P5823）构造相同元素间隔为等差1的数组
7383（https://www.luogu.com.cn/problem/P7383）贪心构造
7947（https://www.luogu.com.cn/problem/P7947）贪心构造积为 n 和为 k 的数列，乘积分解成质因数其和最小
9101（https://www.luogu.com.cn/problem/P9101）构造恰好 k 条路径的有向无环图
8976（https://www.luogu.com.cn/problem/P8976）经典枚举连续段进行构造数组的前半段与后半段和满足要求
8910（https://www.luogu.com.cn/problem/P8910）经典置换环构造
8880（https://www.luogu.com.cn/problem/P8880）脑筋急转弯构造分奇数与偶数讨论

===================================CodeForces===================================
1396A（https://codeforces.com/problemset/problem/1396/A）贪心构造
1118E（https://codeforces.com/problemset/problem/1118/E）模拟贪心构造
960C（https://codeforces.com/problemset/problem/960/C）贪心构造
1793B（https://codeforces.com/contest/1793/problem/B）思维题贪心构造
1375D（https://codeforces.com/problemset/problem/1375/D）经典MEX的构造题，使得数组非降，即最小的未出现在数组中的数
1348D（https://codeforces.com/problemset/problem/1348/D）利用二进制优化将数分割，并利用分裂特点进行计算
1554D（https://codeforces.com/problemset/problem/1554/D）构造，将n取一半向下取整
1788C（https://codeforces.com/problemset/problem/1788/C）构造，选取 1 到 2n 的数两两配对且和为连续数
1367D（https://codeforces.com/problemset/problem/1367/D）逆向思维模拟构造
1485D（https://codeforces.com/problemset/problem/1485/D）按照数据范围进行构造
1722G（https://codeforces.com/problemset/problem/1722/G）根据奇偶性和异或和特点进行构造
1822D（https://codeforces.com/contest/1822/problem/D）构造前缀和取模互不相同的排列
D - Make Them Even（https://atcoder.jp/contests/abc109/tasks/abc109_d）根据奇偶性与遍历顺序构造
1509D（https://codeforces.com/contest/1509/problem/D）LCS最短公共超序列构造，特殊值域下的O(n)鸽巢原理
1473C（https://codeforces.com/contest/1473/problem/C）special property of inverse pair of s1s2..sn..s2s1
1469D（https://codeforces.com/contest/1469/problem/D）square ceil greedy implemention
1478B（https://codeforces.com/contest/1478/problem/B）brute_force|bag_dp|construction

====================================AtCoder=====================================
B - Construct Sequences（https://atcoder.jp/contests/agc007/tasks/agc007_b）脑筋急转弯思维构造公式
D - Non-decreasing（https://atcoder.jp/contests/abc081/tasks/arc086_b）经典贪心构造分类讨论
D - Grid Components（https://atcoder.jp/contests/abc092/tasks/arc093_b）思维题构造
F - XOR Matching（https://atcoder.jp/contests/abc126/tasks/abc126_f）思维题构造，利用0^1^2^..^(2**m-1)=0的特点

参考：OI WiKi（xx）
"""
import math
from collections import deque, Counter, defaultdict
from typing import List

from src.mathmatics.number_theory.template import NumberTheory
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1478b(ac=FastIO()):
        for _ in range(ac.read_int()):
            q, d = ac.read_list_ints()
            queries = ac.read_list_ints()
            ceil = 10 * d + 9
            dp = [0] * (ceil + 1)
            dp[0] = 1
            for i in range(1, ceil + 1):
                if str(d) in str(i):
                    for j in range(i, ceil + 1):
                        if dp[j - i]:
                            dp[j] = 1
            for num in queries:
                if num >= 10 * d + 9 or dp[num]:
                    ac.st("YES")
                else:
                    ac.st("NO")
        return

    @staticmethod
    def cf_1367d(ac=FastIO()):
        # 模板：逆向思维进行生成模拟构造
        for _ in range(ac.read_int()):
            s = ac.read_str()
            m = ac.read_int()
            nums = ac.read_list_ints()
            ans = [""] * m
            lst = deque(sorted(list(s), reverse=True))
            while max(nums) >= 0:
                zero = [i for i in range(m) if nums[i] == 0]
                k = len(zero)
                while len(set(list(lst)[:k])) != 1:
                    lst.popleft()
                for i in zero:
                    nums[i] = -1
                    ans[i] = lst.popleft()
                while lst and lst[0] == ans[zero[0]]:
                    lst.popleft()
                for i in range(m):
                    if nums[i] != -1:
                        nums[i] -= sum(abs(i - j) for j in zero)
            ac.st("".join(ans))
        return

    @staticmethod
    def cf_1788c(ac=FastIO()):
        # 模板：选取 1 到 2n 的数两两配对且和为连续数
        for _ in range(ac.read_int()):
            n = ac.read_int()
            if n % 2:
                ac.st("YES")
                x = n // 2
                for i in range(1, n + 1):
                    if i <= x:
                        ac.lst([i, i + n + x + 1])
                    else:
                        ac.lst([i, i - x + n])
            else:
                ac.st("NO")
        return

    @staticmethod
    def lc_280(nums: List[int]) -> None:
        # 模板：摆动排序数组构造
        nums.sort()
        n = len(nums)
        ans = [0] * n
        j = n - 1
        for i in range(1, n, 2):
            ans[i] = nums[j]
            j -= 1
        j = 0
        for i in range(0, n, 2):
            ans[i] = nums[j]
            j += 1
        for i in range(n):
            nums[i] = ans[i]
        return

    @staticmethod
    def lc_1982(n: int, sums: List[int]) -> List[int]:
        # 模板：经典根据数组所有子集的和构造原数组
        low = min(sums)
        if low < 0:
            sums = [num - low for num in sums]

        # 从小到大依次构造元素
        cnt = Counter(sums)
        lst = sorted(cnt.keys())
        cnt[0] -= 1
        ans = []
        pre = defaultdict(int)
        pre_sum = []
        for _ in range(n):
            for num in lst:
                if cnt[num] > pre[num]:
                    ans.append(num)
                    for p in pre_sum[:]:
                        pre[p + num] += 1
                        pre_sum.append(p + num)
                    pre[num] += 1
                    pre_sum.append(num)
                    break

        # 返回任意一个结果
        for i in range(1 << n):
            cur = [j for j in range(n) if i & (1 << j)]
            if sum(ans[j] for j in cur) == -low:
                for j in cur:
                    ans[j] *= -1
                return ans
        return []

    @staticmethod
    def lc_2663(s: str, k: int) -> str:
        # 模板：贪心构造不含任何回文子串的字典序最小的字符串
        n = len(s)
        for i in range(n - 1, -1, -1):
            # 倒序枚举
            for x in range(ord(s[i]) - ord("a") + 1, k):
                w = chr(ord("a") + x)
                # 只要没有长度为 2 和长度为 3 的回文子串则都不存在任何大于 1 长度的回文子串
                if (i == 0 or s[i - 1] != w) and not (i >= 2 and w == s[i - 2]):
                    ans = s[:i] + w
                    # 贪心赋值且字典序最小
                    while len(ans) < n:
                        for y in range(0, k):
                            x = chr(y + ord("a"))
                            if x != ans[-1] and (len(ans) < 2 or ans[-2] != x):
                                ans += x
                                break
                    return ans
        return ""

    @staticmethod
    def lg_p7947(ac=FastIO()):
        # 模板：贪心构造积为 n 和为 k 的数列，乘积分解成质因数其和最小
        n, k = ac.read_list_ints()
        ans = []
        for p, c in NumberTheory().get_prime_factor(n):
            ans.extend([p] * c)
        if sum(ans) > k:
            ac.st(-1)
        else:
            ans.extend([1] * (k - sum(ans)))
            ac.st(len(ans))
            ac.lst(ans)
        return

    @staticmethod
    def lg_p9101(ac=FastIO()):
        # 模板：构造恰好 k 条路径的有向无环图
        k = ac.read_int()
        dct = defaultdict(list)
        kk = k
        if k == 1:  # 特判要求至少有两个节点
            ac.st(2)
            ac.lst([2, -1])
            ac.lst([-1, -1])
            return

        # 多个 A -> BC -> D的结构形式
        x = 1
        cnt = len(bin(k)[2:]) - 1
        level = []
        for i in range(2 * cnt + 1):
            if i % 2 == 0:
                level.append([x])
                x += 1
            else:
                level.append([x, x + 1])
                x += 2
        for i in range(2 * cnt):
            if len(level[i]) == 1:
                dct[level[i][0]] = level[i + 1][:]
            else:
                dct[level[i][0]] = level[i + 1][:]
                dct[level[i][1]] = level[i + 1][:]

        # 使用二进制逐层往上增加节点
        k -= (1 << cnt)
        end = x - 1
        pre = 1 << cnt
        for i in range(2 * cnt - 1, 0, -2):
            xx = level[i][1]
            pre //= 2
            if pre <= k:
                k -= pre
                if end not in dct[xx]:
                    dct[xx].append(end)
                else:
                    # 另外增加节点
                    dct[xx].append(x)
                    dct[x].append(end)
                    x += 1

        ac.st(x - 1)
        # 将终点替换为最大的编号
        ind = {end: x - 1, x - 1: end}
        for i in range(1, x):
            i = ind.get(i, i)
            ans = [ind.get(j, j) for j in dct[i]]
            while len(ans) < 2:
                ans.append(-1)
            ac.lst(ans)
        return

    @staticmethod
    def lg_p8976(ac=FastIO()):
        # 模板：经典枚举连续段进行构造数组的前半段与后半段和满足要求
        for _ in range(ac.read_int()):
            n, a, b = ac.read_list_ints()
            mid = n // 2 + 1
            if a + b > n * (n + 1) // 2 or ac.max(a, b) > (n // 2) * (mid + n) // 2:
                ac.st(-1)
                continue
            s = n * (n + 1) // 2
            lst = [a, b]
            ans = []
            for i in range(n // 2 + 1):  # 枚举起始的连续段
                if ans:
                    break
                x = n // 2 - i
                for aa, bb in [[0, 1], [1, 0]]:
                    if x:
                        rest = lst[aa] - i * (i + 1) // 2
                        # 计算剩余的连续段
                        y = math.ceil((rest * 2 / x - x + 1) / 2)
                        y = ac.max(y, i + 1)

                        if y + x - 1 <= n:
                            cur = i * (i + 1) // 2 + x * (y + y + x - 1) // 2
                            if cur >= lst[aa] and s - cur >= lst[bb]:
                                pre = list(range(1, i + 1)) + list(range(y, y + x))
                                post = list(range(i + 1, y)) + list(range(y + x, n + 1))
                                ans = pre + post if aa == 0 else post + pre
                                break
                    else:
                        if n // 2 * (1 + n // 2) // 2 >= a and s - a >= b:
                            ans = list(range(1, n + 1))
                            break
            if not ans:
                ac.st(-1)
            else:
                ac.lst(ans)
        return

    @staticmethod
    def lg_p8910(ac=FastIO()):
        # 模板：经典置换环构造
        for _ in range(ac.read_int()):
            n, k = ac.read_list_ints()
            nex = [0] * (n + 1)
            for i in range(k):
                nex[i] = n - k + i
            for i in range(k, n):
                nex[i] = i - k
            ans = []
            for i in range(n):
                if nex[i] != i:
                    # 找到一个环上所有点
                    lst = [i]
                    while nex[lst[-1]] != i:
                        lst.append(nex[lst[-1]])
                    m = len(lst)
                    # 依次相邻进行赋值
                    ans.append([n + 1, lst[0] + 1])
                    for x in range(1, m):
                        ans.append([lst[x - 1] + 1, lst[x] + 1])
                    ans.append([lst[m - 1] + 1, n + 1])
                    for x in lst:
                        nex[x] = x
            ac.st(len(ans))
            for a in ans:
                ac.lst(a)
        return

    @staticmethod
    def lg_p8880(ac=FastIO()):
        # 模板：脑筋急转弯构造分奇数与偶数讨论
        n = ac.read_int()
        if n % 2 == 0:
            ac.st(-1)
            return
        nums = ac.read_list_ints()
        ind = {num: i for i, num in enumerate(nums)}
        a = [-1] * n
        b = [-1] * n
        for i in range(n):
            j = (i - 1) % n
            x = (i + j) % n
            a[ind[x]] = i
            b[ind[x]] = j
        ac.lst(a)
        ac.lst(b)
        return