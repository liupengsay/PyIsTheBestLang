"""
Algorithm：背包DP、分组背包、一维（无限有限）背包、二维背包、多重背包、分组背包、限制背包、填表法（过去状态预测未来状态）、刷表法（当前状态预测未来状态）、可撤销背包
Function：一重背包DP，数量有限从后往前遍历，数量无限则从前往后遍历；多重背包DP，可二进制优化拆分

====================================LeetCode====================================
140（https://leetcode.com/problems/word-break-ii/） 01 背包生成specific_plan
2218（https://leetcode.com/problems/maximum-value-of-k-coins-from-piles/）分组背包DP
2585（https://leetcode.com/contest/weekly-contest-335/problems/number-of-ways-to-earn-points/）看似二进制优化背包，实则数量转移
2189（https://leetcode.com/problems/number-of-ways-to-build-house-of-cards/）转换为01背包求解
254（https://leetcode.com/problems/factor-combinations/）乘法结合背包DP
1449（https://leetcode.com/problems/form-largest-integer-with-digits-that-add-up-to-target/）代价一定情况下的最大数值
1049（https://leetcode.com/problems/last-stone-weight-ii/）问题，转化为01背包求解
2742（https://leetcode.com/problems/painting-the-walls/description/）剪枝DP，可以转换为01背包求解
2518（https://leetcode.com/problems/number-of-great-partitions/）01背包counter
1155（https://leetcode.com/problems/number-of-dice-rolls-with-target-sum/description/）类似分组背包，可线性刷表法与填表法
2902（https://leetcode.com/problems/count-of-sub-multisets-with-bounded-sum/description/）按照单调队列的思想mod|分组DP，prefix_sum优化，也有容斥的思想，可撤销背包

=====================================LuoGu======================================
1048（https://www.luogu.com.cn/problem/P1048）一维背包DP，数量有限，从后往前遍历
1049（https://www.luogu.com.cn/problem/P1049）一维背包DP
1776（https://www.luogu.com.cn/problem/P1776）多重背包，二进制拆分优化，进一步单调队列优化
1509（https://www.luogu.com.cn/problem/P1509）四重背包
1060（https://www.luogu.com.cn/problem/P1509）一维背包DP
1566（https://www.luogu.com.cn/problem/P1566#submit）限制counter背包
1759（https://www.luogu.com.cn/problem/P1759）二重背包并specific_plans
1794（https://www.luogu.com.cn/problem/P1794）二重背包
1806（https://www.luogu.com.cn/problem/P1806）连续值一维有限背包counter
1853（https://www.luogu.com.cn/problem/P1853）一维无限背包有技巧成倍缩小背包范围
1874（https://www.luogu.com.cn/problem/P1874）类似区间与背包的结合brute_force前一个字符串|号分割点求和
1977（https://www.luogu.com.cn/problem/P1977）分组有限背包
1586（https://www.luogu.com.cn/problem/P1586）分组无限背包
1566（https://www.luogu.com.cn/problem/P1566）一维有限背包counter
1509（https://www.luogu.com.cn/problem/P1509）二重背包，转移的时候比较优先级有两个
1504（https://www.luogu.com.cn/problem/P1504）一维有限背包DP
2066（https://www.luogu.com.cn/problem/P2066）分组有限背包，转移的时候比较优先级有两个
2340（https://www.luogu.com.cn/problem/P2340）01背包变种问题还带负数|和
2370（https://www.luogu.com.cn/problem/P2370）最小生成树的思想sorting后greedy背包放入，达成条件后即中止
2386（https://www.luogu.com.cn/problem/P2386）背包DP去重组合|和counter
2623（https://www.luogu.com.cn/problem/P2623）综合背包，函数取最大值一维有限背包，连续个数二进制优化背包，无限个数背包
1474（https://www.luogu.com.cn/problem/P1474）一维无限背包counter
1466（https://www.luogu.com.cn/problem/P1466）一维有限背包|和counter
1455（https://www.luogu.com.cn/problem/P1455）union_find搭配购买组合|一维有限背包
1230（https://www.luogu.com.cn/problem/P1230）sorting后根据时间限制动态更新一维有限背包
1077（https://www.luogu.com.cn/problem/P1077）一维有限背包counter
2725（https://www.luogu.com.cn/problem/P2725）01无限背包counter
2918（https://www.luogu.com.cn/problem/P2918）一维无限背包，需要根据题意增|背包容量上限
3027（https://www.luogu.com.cn/problem/P3027）一维无限背包，需要根据题意利润
3030（https://www.luogu.com.cn/problem/P3030）分组brute_force有限背包
3040（https://www.luogu.com.cn/problem/P3040）二维变种背包
4817（https://www.luogu.com.cn/problem/P4817）一维有限背包DP变种
5087（https://www.luogu.com.cn/problem/P5087）二维有限背包变种问题
6205（https://www.luogu.com.cn/problem/P6205）一维无限背包
6389（https://www.luogu.com.cn/problem/P6389）一维有限背包变种问题，寻找和尽可能接近的两个分组
6567（https://www.luogu.com.cn/problem/P6567）一维二进制优化有限背包，即物品数为连续值时需要二进制优化
6771（https://www.luogu.com.cn/problem/P6771）sorting后，一维有限变种背包，二进制优化
2842（https://www.luogu.com.cn/problem/P2842）一维无限背包DP不区分顺序
2840（https://www.luogu.com.cn/problem/P2840）一维无限背包DP区分顺序
2834（https://www.luogu.com.cn/problem/P2834）一维无限背包DP不区分顺序
1064（https://www.luogu.com.cn/problem/P1064）有依赖的01背包，brute_force状态分组讨论，分组背包
1156（https://www.luogu.com.cn/problem/P1156）转换为背包01DP求解
1273（https://www.luogu.com.cn/problem/P1273）树上分组背包
1284（https://www.luogu.com.cn/problem/P1284）brute_force三角形两边作为二维bool背包，并三角形面积公式
1441（https://www.luogu.com.cn/problem/P1441）brute_force|背包DP
1537（https://www.luogu.com.cn/problem/P1537）问题二进制背包优化bool背包，划分成和相等的两部分
1541（https://www.luogu.com.cn/problem/P1541）四维背包brute_force，填表法
1759（https://www.luogu.com.cn/problem/P1759）二维背包并输出lexicographical_order最小的方案
1833（https://www.luogu.com.cn/problem/P1833）完全背包与单点队列优化多重背包组合
2014（https://www.luogu.com.cn/problem/P2014）增|一个虚拟源点将DAG转换为树上背包
2079（https://www.luogu.com.cn/problem/P2079）滚动hash背包DP，两层hash节省空间
2170（https://www.luogu.com.cn/problem/P2170）连通块|二进制01背包优化
2214（https://www.luogu.com.cn/problem/P2214）变种背包DPgreedy
2306（https://www.luogu.com.cn/problem/P2306）data_range|counter后二进制优化的01背包
2320（https://www.luogu.com.cn/problem/P2320）二进制分解greedy反向
2737（https://www.luogu.com.cn/problem/P2737）完全背包变种问题
2760（https://www.luogu.com.cn/problem/P2760）单调队列优化的多重背包
2854（https://www.luogu.com.cn/problem/P2854）分组01背包
2938（https://www.luogu.com.cn/problem/P2938）分组完全背包
2979（https://www.luogu.com.cn/problem/P2979）分组01背包
3010（https://www.luogu.com.cn/problem/P3010）变形01背包，两heapq差值最小的分配方案数
3423（https://www.luogu.com.cn/problem/P3423）二进制优化多重背包与方案输出
3983（https://www.luogu.com.cn/problem/P3983）两个分组完全背包
5322（https://www.luogu.com.cn/problem/P5322）典型二维 DP 转换为分组背包
5365（https://www.luogu.com.cn/problem/P5365）01背包 DP brute_force数量
5662（https://www.luogu.com.cn/problem/P5662）完全背包变形greedy题目
1417（https://www.luogu.com.cn/problem/P1417）greedysorting后 01 背包最大值

===================================CodeForces===================================
577B（https://codeforces.com/problemset/problem/577/B）mod|counter二进制优化与背包DP，寻找非空子序列的和整除给定的数
543A（https://codeforces.com/problemset/problem/543/A）二维有限背包DP，当作无限处理
148E（https://codeforces.com/problemset/problem/148/E）01背包brute_force，两层动态规划
1433F（https://codeforces.com/problemset/problem/1433/F）01背包brute_force，两层动态规划
1657D（https://codeforces.com/contest/1657/problem/D）一维无限乘积背包预处理，欧拉级数复杂度，结合binary_searchgreedy

====================================AtCoder=====================================
D - Mixing Experiment（https://atcoder.jp/contests/abc054/tasks/abc054_d）二维01背包
D - Match Matching（https://atcoder.jp/contests/abc118/tasks/abc118_d）greedy背包DP，并还原方案
E - All-you-can-eat（https://atcoder.jp/contests/abc145/tasks/abc145_e）brain_teaser|01背包，需要先sorting，刷表法解决

=====================================AcWing=====================================
4（https://www.acwing.com/problem/content/4/）二进制优化多重背包
6（https://www.acwing.com/problem/content/description/6/）单调队列优化多重背包
7（https://www.acwing.com/problem/content/7/）01背包、完全背包与多重背包混合
8（https://www.acwing.com/problem/content/8/）二维01背包
9（https://www.acwing.com/problem/content/9/）分组01背包问题
10（https://www.acwing.com/problem/content/10/）树上背包
11（https://www.acwing.com/problem/content/description/11/）背包问题求方案数
12（https://www.acwing.com/problem/content/12/）背包问题求specific_plan，有两种写法
4081（https://www.acwing.com/problem/content/4084/）转换为二维背包问题求解

"""
import bisect
from collections import defaultdict, deque, Counter
from functools import lru_cache
from itertools import combinations
from math import inf
from typing import List

from src.dp.bag_dp.template import BagDP
from src.graph.union_find.template import UnionFind
from src.mathmatics.number_theory.template import NumberTheory
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1433f(ac=FastIO()):
        # 两层背包DP，矩阵动态规划转移
        m, n, k = ac.read_list_ints()
        pre = [-inf] * k
        pre[0] = 0
        x = n // 2
        for _ in range(m):
            nums = ac.read_list_ints()
            dp = [[-inf] * k for _ in range(x + 1)]
            dp[0][0] = 0
            for num in nums:
                nex = [ls[:] for ls in dp]
                for i in range(x):
                    for j in range(k):
                        d = (j + num) % k
                        nex[i + 1][d] = ac.max(dp[i][j] + num, nex[i + 1][d])
                dp = [ls[:] for ls in nex]
            tmp = [max(dp[i][j] for i in range(x + 1)) for j in range(k)]

            cur = pre[:]
            for i in range(k):
                for j in range(k):
                    cur[(i + j) % k] = ac.max(cur[(i + j) % k], pre[i] + tmp[j])
            pre = cur[:]

        ac.st(pre[0])
        return

    @staticmethod
    def cf_543a(ac=FastIO()):
        # 分组背包 DP 有限作为无限
        n, m, b, mod = ac.read_list_ints()
        nums = ac.read_list_ints()
        pre = [[0] * (b + 1) for _ in range(m + 1)]
        pre[0][0] = 1
        for num in nums:
            for i in range(1, m + 1):
                # 由于每个用户的天数都可以取到 m 所以当作类似无限背包转移
                for j in range(num, b + 1):
                    pre[i][j] = (pre[i][j] + pre[i - 1][j - num]) % mod
        ac.st(sum(pre[m]) % mod)
        return

    @staticmethod
    def cf_577b(m, nums):
        # mod|counter二进制优化与背包DP，寻找非空子序列的和整除给定的数
        cnt = [0] * m
        for num in nums:
            cnt[num % m] += 1
        if cnt[0] or max(cnt) >= m:
            return "YES"
        pre = [0] * m
        for i in range(1, m):
            if cnt[i]:
                for x in BagDP().bin_split_1(cnt[i]):
                    cur = pre[:]
                    y = (x * i) % m
                    cur[y] = 1
                    for j in range(m):
                        if pre[j]:
                            cur[(j + y) % m] = 1
                    pre = cur[:]
                if pre[0]:
                    return "YES"
        return "NO"

    @staticmethod
    def lc_2218(piles: List[List[int]], k: int) -> int:

        # 线性有限分组背包 DP 注意转移
        cur = [0] * (k + 1)
        for lst in piles:

            n = len(lst)
            pre = [0] * (n + 1)
            for i in range(n):
                pre[i + 1] = pre[i] + lst[i]
            # 注意这里需要拷贝
            nex = cur[:]
            for j in range(1, k + 1):
                for x in range(min(n + 1, j + 1)):
                    nex[j] = max(nex[j], cur[j - x] + pre[x])
            cur = nex[:]
        return cur[-1]

    @staticmethod
    def lg_p6567(ac=FastIO()):
        # 一维有限二进制优化背包
        n, m = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        target = ac.read_list_ints()
        ceil = max(target)
        dp = [0] * (ceil + 1)
        dp[0] = 1
        for k, a in nums:
            for b in BagDP().bin_split_1(a):
                x = b * k
                for i in range(ceil, x - 1, -1):
                    if dp[i - x]:
                        dp[i] = 1
        for t in target:
            if dp[t]:
                ac.st("Yes")
            else:
                ac.st("No")
        return

    @staticmethod
    def lc_2742_1(cost: List[int], time: List[int]) -> int:

        # 剪枝DP，可以转换为01背包求解
        @lru_cache(None)
        def dfs(i, pre):
            if pre >= n - i:  # 剪枝
                return 0
            if i == n:
                return inf
            res = dfs(i + 1, pre - 1)
            cur = dfs(i + 1, pre + time[i]) + cost[i]
            if cur < res:
                res = cur
            return res

        n = len(cost)
        return dfs(0, 0)

    @staticmethod
    def lc_2742_2(cost: List[int], time: List[int]) -> int:

        # 剪枝DP，可以转换为01背包求解
        n = len(cost)
        dp = [sum(time)] * (n + 1)
        dp[0] = 0
        for i in range(n):
            c, t = cost[i], time[i]
            for j in range(n, -1, -1):
                s = j - t - 1 if j - t - 1 >= 0 else 0  # 此时表示付费油漆匠刷的时候免费的油漆匠不一定要刷满t
                if dp[s] + c < dp[j]:
                    dp[j] = dp[s] + c
        return dp[-1]

    @staticmethod
    def lc_2518(nums: List[int], k: int) -> int:
        # 01背包counter
        mod = 10 ** 9 + 7
        dp = [0] * k
        s = sum(nums)
        if s < 2 * k:
            return 0
        dp[0] = 1
        n = len(nums)
        for num in nums:
            for i in range(k - 1, num - 1, -1):
                dp[i] += dp[i - num]
        ans = pow(2, n, mod)
        ans -= 2 * sum(dp)
        return ans % mod

    @staticmethod
    def lc_2585(target: int, types: List[List[int]]) -> int:
        # 看似二进制优化 DP 实则矩阵 DP 转移
        mod = 10 ** 9 + 7
        n = len(types)
        pre = [0] * (target + 1)
        pre[0] = 1
        for i in range(n):
            c, m = types[i]
            cur = pre[:]
            for x in range(1, c + 1):
                for j in range(target - x * m + 1):
                    if x * m + j <= target:
                        cur[x * m + j] += pre[j]
            pre = [num % mod for num in cur]
        return pre[-1]

    @staticmethod
    def cf_1657d(ac=FastIO()):
        n, c = ac.read_list_ints()
        dp = [0] * (c + 1)
        for _ in range(n):
            cc, dd, hh = ac.read_list_ints()
            dp[cc] = ac.max(dp[cc], dd * hh)

        for i in range(1, c + 1):
            dp[i] = ac.max(dp[i], dp[i - 1])
            x = dp[i]
            for y in range(i * 2, c + 1, i):
                dp[y] = ac.max(dp[y], x * (y // i))

        ans = []
        for _ in range(ac.read_int()):
            h, d = ac.read_list_ints()
            if h * d >= dp[c]:
                ans.append(-1)
            else:
                ans.append(bisect.bisect_right(dp, h * d))
        ac.lst(ans)
        return

    @staticmethod
    def lc_254(n: int) -> List[List[int]]:
        # 因子分解与背包dp分解
        lst = NumberTheory().get_all_factor(n)
        m = len(lst)
        dp = defaultdict(list)
        dp[1] = [[]]
        for i in range(1, m - 1):
            for j in range(i, m):
                if lst[j] % lst[i] == 0:
                    x = lst[j] // lst[i]
                    for p in dp[x]:
                        dp[lst[j]].append(p + [lst[i]])
        return [ls for ls in dp[n] if ls]

    @staticmethod
    def abc_118d(ac=FastIO()):
        # greedy背包DP，并还原方案
        score = [2, 5, 5, 4, 5, 6, 3, 7, 6]
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        nums.sort(reverse=True)
        dp = [-inf] * (n + 1)
        dp[0] = 0
        for num in nums:
            val = score[num - 1]
            for i in range(val, n + 1):
                if dp[i - val] + 1 > dp[i]:
                    dp[i] = dp[i - val] + 1
        ans = []
        i = n
        while i:
            for num in nums:
                val = score[num - 1]
                if i >= val and dp[i] == dp[i - val] + 1:
                    ans.append(num)
                    i -= val
                    break
        ans.sort(reverse=True)
        ac.st("".join(str(x) for x in ans))
        return

    @staticmethod
    def abc_145e(ac=FastIO()):
        # brain_teaser|01背包，需要先sorting，刷表法解决
        n, t = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort()
        dp = [0] * (t + 3010)
        for x, y in nums:
            for i in range(t - 1, -1, -1):
                if dp[i] + y > dp[i + x]:
                    dp[i + x] = dp[i] + y
        ac.st(max(dp))
        return

    @staticmethod
    def ac_6(ac=FastIO()):
        # 单调队列优化的多重背包问题，即限定个数和体积价值求最大值
        n, m = ac.read_list_ints()
        dp = [0] * (m + 1)
        for _ in range(n):
            # 体积 价值 数量
            v, w, s = ac.read_list_ints()
            for r in range(v):
                stack = deque()
                for i in range(r, m + 1, v):
                    while stack and stack[0][0] < i - s * v:
                        stack.popleft()
                    while stack and stack[-1][1] + (i - stack[-1][0]) // v * w <= dp[i]:
                        stack.pop()
                    stack.append([i, dp[i]])
                    dp[i] = stack[0][1] + (i - stack[0][0]) // v * w
        ac.st(dp[-1])
        return

    @staticmethod
    def ac_10(ac=FastIO()):

        # 树上背包
        n, m = ac.read_list_ints()
        vol = []
        weight = []
        parent = [-1] * n
        dct = [[] for _ in range(n)]
        root = 0
        for i in range(n):
            v, w, p = ac.read_list_ints()
            p -= 1
            parent[i] = p
            if p != -2:
                dct[p].append(i)
            else:
                root = i
            vol.append(v)
            weight.append(w)

        # 树上背包
        stack = [root]
        sub = [[0] * (m + 1) for _ in range(n)]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in dct[i]:
                    stack.append(j)
            else:
                i = ~i
                sub[i][vol[i]] = weight[i]
                for j in dct[i]:
                    cur = sub[i][:]
                    for x in range(vol[i], m + 1):  # 必须选择父节点的物品
                        for y in range(m + 1 - x):
                            cur[x + y] = max(cur[x + y], sub[i][x] + sub[j][y])
                    sub[i] = cur[:]
        ac.st(max(sub[root]))
        return

    @staticmethod
    def ac_11(ac=FastIO()):
        # 01背包求方案数
        n, m = ac.read_list_ints()
        dp = [0] * (m + 1)
        cnt = [1] * (m + 1)  # 注意方案数都初始化为1
        mod = 10 ** 9 + 7
        for _ in range(n):
            v, w = ac.read_list_ints()
            for i in range(m, v - 1, -1):
                if dp[i - v] + w > dp[i]:
                    dp[i] = dp[i - v] + w
                    cnt[i] = cnt[i - v]
                elif dp[i - v] + w == dp[i]:
                    cnt[i] += cnt[i - v]
                    cnt[i] %= mod
        ac.st(cnt[-1])
        return

    @staticmethod
    def ac_12_1(ac=FastIO()):
        # 01背包求specific_plan
        n, m = ac.read_list_ints()
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        nums = [ac.read_list_ints() for _ in range(n)]

        # 要求lexicographical_order最小所以倒着来
        for i in range(n - 1, -1, -1):
            v, w = nums[i]
            for j in range(m, -1, -1):
                dp[i][j] = dp[i + 1][j]
                if j >= v and dp[i + 1][j - v] + w > dp[i][j]:
                    dp[i][j] = dp[i + 1][j - v] + w

        # 再正着求最小的lexicographical_order
        j = m
        path = []
        for i in range(n):
            v, w = nums[i]
            if j >= v and dp[i][j] == dp[i + 1][j - v] + w:
                j -= v
                path.append(i + 1)
        ac.lst(path)
        return

    @staticmethod
    def ac_12_2(ac=FastIO()):
        # 01背包求specific_plan
        n, m = ac.read_list_ints()
        dp = [[0, [-1]] for _ in range(m + 1)]
        for ind in range(n):
            v, w = ac.read_list_ints()
            for i in range(m, v - 1, -1):
                if dp[i - v][0] + w > dp[i][0] or (
                        dp[i - v][0] + w == dp[i][0] and dp[i - v][1] + [ind + 1] < dp[i][1]):
                    dp[i] = [dp[i - v][0] + w, dp[i - v][1] + [ind + 1]]
        ac.lst(dp[-1][1][1:])
        return

    @staticmethod
    def lg_p1064(ac=FastIO()):
        # 有依赖的分组背包
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(m)]
        sub = [[] for _ in range(m)]
        for i in range(m):
            v, p, q = ac.read_list_ints()
            if q == 0:
                dct[i].append([v, p])
            else:
                sub[q - 1].append([v, p])
        dp = [[0] * (n + 1) for _ in range(2)]
        pre = 0
        for i in range(m):
            if dct[i]:
                cur = 1 - pre
                dp[cur] = dp[pre][:]
                x = len(sub[i])
                for j in range(1 << x):
                    lst = dct[i] + [sub[i][k] for k in range(x) if j & (1 << k)]
                    gain = sum(v * p for v, p in lst)
                    cost = sum(v for v, _ in lst)
                    for xx in range(n, cost - 1, -1):
                        dp[cur][xx] = ac.max(
                            dp[cur][xx], dp[pre][xx - cost] + gain)
                pre = cur
        ac.st(dp[pre][-1])
        return

    @staticmethod
    def lg_p1156(ac=FastIO()):
        # 变形背包
        n, m = ac.read_list_ints()

        dct = [ac.read_list_ints() for _ in range(m)]
        dct.sort(key=lambda it: it[0])

        dp = [-inf] * (n + 1)  # dp[height]=life 到达该高度后剩余的生命值
        dp[0] = 10
        for t, f, h in dct:
            if dp[0] < t:
                ac.st(dp[0])
                return
            for i in range(n, -1, -1):
                if dp[i] >= t:
                    if i + h >= n:
                        ac.st(t)
                        return
                    # 不吃
                    if i + h <= n:
                        dp[i + h] = ac.max(dp[i + h], dp[i])
                    # 吃掉
                    dp[i] += f
        ac.st(dp[0])
        return

    @staticmethod
    def lg_p1273(ac=FastIO()):
        # 树上分组背包
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for j in range(n - m):
            lst = ac.read_list_ints()
            for i in range(1, len(lst), 2):
                # 边的成本
                dct[j].append([lst[i] - 1, lst[i + 1]])
        # 节点收益
        nums = [0] * (n - m) + ac.read_list_ints()
        sub = [[] for _ in range(n)]
        stack = [0]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j, _ in dct[i]:
                    stack.append(j)
            else:
                i = ~i
                # sub[i][j]表示人数为j时的最大收益
                sub[i].append(0)
                if i >= n - m:
                    sub[i].append(nums[i])
                    continue

                for j, cost in dct[i]:
                    cur = sub[i][:]
                    for k1 in range(m + 1):
                        if k1 >= len(sub[i]):
                            break
                        for k2 in range(m - k1 + 1):
                            if k2 >= len(sub[j]):
                                break
                            if len(cur) < k1 + k2 + 1:
                                cur.extend([-inf] * (k1 + k2 + 1 - len(cur)))
                            # 左边k1右边k2个用户时聚拢的最大收益
                            cur[k1 + k2] = ac.max(cur[k1 + k2], sub[j][k2] + sub[i][k1] - cost)
                    sub[j] = []
                    sub[i] = cur[:]
        for x in range(m, -1, -1):
            if x < len(sub[0]) and sub[0][x] >= 0:
                ac.st(x)
                return
        ac.st(0)
        return

    @staticmethod
    def lg_p1284(ac=FastIO()):

        # brute_force三角形两边作为二维bool背包
        n = ac.read_int()

        def check():
            # 三角形面积公式
            ss = (a + b + c) / 2
            return (ss * (ss - a) * (ss - b) * (ss - c)) ** 0.5

        nums = []
        while len(nums) < n:
            nums.extend(ac.read_list_ints())

        # 二维背包 dp[i][j] 表示能否凑成两条便分别为 i 和 j
        s = sum(nums)
        dp = [[0] * (s // 2 + 1) for _ in range(s // 2 + 1)]
        dp[0][0] = 1
        for num in nums:
            for i in range(s // 2, -1, -1):
                for j in range(s // 2, -1, -1):
                    if j >= num and dp[i][j - num]:
                        dp[i][j] = 1
                    if i >= num and dp[i - num][j]:
                        dp[i][j] = 1
        ans = -1
        for a in range(s // 2 + 1):
            for b in range(s // 2 + 1):
                if dp[a][b]:
                    c = s - a - b
                    # 三角形合法判断公式
                    if b + c > a > 0 and a + c > b > 0 and a + b > c > 0:
                        cur = check()
                        ans = ac.max(ans, cur)
        if ans == -1:
            ac.st(ans)
        else:
            ac.st(int(ans * 100))
        return

    @staticmethod
    def lg_p1441(ac=FastIO()):
        # brute_force|背包DP
        n, m = ac.read_list_ints()
        a = ac.read_list_ints()
        ans = 0
        s = sum(a)
        for item in combinations(a, n - m):
            dp = [0] * (s + 1)
            dp[0] = 1
            for num in item:
                for i in range(s, num - 1, -1):
                    if dp[i - num]:
                        dp[i] = 1
            cur = sum(dp) - 1
            ans = ac.max(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1537(ac=FastIO()):
        # 问题二进制背包优化bool背包，划分成和相等的两部分
        case = 0
        while True:
            lst = ac.read_list_ints()
            if sum(lst) == 0:
                break

            case += 1
            ac.st(f"Collection #{case}:")
            s = sum(lst[i] * (i + 1) for i in range(6))
            if s % 2:
                ac.st("Can't be divided.")
                ac.st("")
                continue

            m = s // 2
            dp = [0] * (m + 1)
            dp[0] = 1
            for x in range(6):
                w, s = x + 1, lst[x]
                if s:
                    for num in BagDP().bin_split_1(s):
                        for i in range(m, w * num - 1, -1):
                            if dp[i - num * w]:
                                dp[i] = 1
            if dp[-1]:
                ac.st("Can be divided.")
            else:
                ac.st("Can't be divided.")
            ac.st("")
        return

    @staticmethod
    def lg_p1541(ac=FastIO()):

        # 四维背包
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        cnt = Counter(ac.read_list_ints())
        a, b, c, d = cnt[1], cnt[2], cnt[3], cnt[4]
        dp = [[[[0] * (d + 1) for _ in range(c + 1)] for _ in range(b + 1)] for _ in range(a + 1)]
        dp[0][0][0][0] = nums[0]
        ans = 0
        for i in range(a + 1):
            for j in range(b + 1):
                for k in range(c + 1):
                    for p in range(d + 1):
                        if i + 2 * j + 3 * k + 4 * p <= n - 1:
                            pre = 0
                            if i:
                                pre = ac.max(pre, dp[i - 1][j][k][p])
                            if j:
                                pre = ac.max(pre, dp[i][j - 1][k][p])
                            if k:
                                pre = ac.max(pre, dp[i][j][k - 1][p])
                            if p:
                                pre = ac.max(pre, dp[i][j][k][p - 1])
                            dp[i][j][k][p] = ac.max(dp[i][j][k][p], pre + nums[i + 2 * j + 3 * k + 4 * p])
                        if i + 2 * j + 3 * k + 4 * p == n - 1:
                            ans = ac.max(ans, dp[i][j][k][p])
        ac.st(ans)
        return

    @staticmethod
    def lg_p1759(ac=FastIO()):
        # 二维背包输出lexicographical_order最小的方案
        m, v, n = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        dp = [[[0, []] for _ in range(v + 1)] for _ in range(m + 1)]
        # 同时记录时间与lexicographical_order最小的方案
        for i in range(n):
            a, b, c = nums[i]
            for j in range(m, a - 1, -1):
                for k in range(v, b - 1, -1):
                    t, p = dp[j - a][k - b]
                    if dp[j][k][0] < t + c or (dp[j][k][0] == t + c and p + [i + 1] < dp[j][k][1]):
                        dp[j][k] = [t + c, p + [i + 1]]
        ans1, ans2 = dp[m][v]
        ac.st(ans1)
        ac.lst(ans2)
        return

    @staticmethod
    def lg_p1776(ac=FastIO()):
        # 单调队列优化的多重背包问题，即限定个数和体积价值求最大值
        n, m = ac.read_list_ints()
        dp = [0] * (m + 1)
        for _ in range(n):
            a, b, c = ac.read_list_ints()
            # 体积 价值 数量
            v, w, s = b, a, c
            for r in range(v):
                stack = deque()
                for i in range(r, m + 1, v):
                    while stack and stack[0][0] < i - s * v:
                        stack.popleft()
                    while stack and stack[-1][1] + (i - stack[-1][0]) // v * w <= dp[i]:
                        stack.pop()
                    stack.append([i, dp[i]])
                    dp[i] = stack[0][1] + (i - stack[0][0]) // v * w
        ac.st(dp[-1])
        return

    @staticmethod
    def lg_p1799(ac=FastIO()):
        # 典型二维matrix_dp
        n = ac.read_int()
        if not n:
            ac.st(0)
            return
        nums = ac.read_list_ints()
        dp = [[-inf] * (n + 1) for _ in range(n)]
        dp[0][0] = 0
        dp[0][1] = 1 if nums[0] == 1 else 0
        for i in range(1, n):
            dp[i][0] = 0
            for j in range(1, i + 2):
                # 前i个数取j个的最大得分
                dp[i][j] = ac.max(dp[i - 1][j], dp[i - 1][j - 1] + int(nums[i] == j))
        ac.st(max(dp[n - 1]))
        return

    @staticmethod
    def lg_p1833(ac=FastIO()):

        def check(st):
            hh, mm = st.split(":")
            return int(hh) * 60 + int(mm)

        # 完全背包与单点队列优化多重背包组合
        s, e, n = ac.read_list_strs()
        t = check(e) - check(s)
        dp = [0] * (t + 1)
        for _ in range(int(n)):
            tt, cc, p = ac.read_list_ints()
            if not p:
                for i in range(tt, t + 1):
                    dp[i] = ac.max(dp[i], dp[i - tt] + cc)
            else:
                v, w, s = tt, cc, p
                for r in range(v):
                    stack = deque()
                    for i in range(r, t + 1, v):
                        while stack and stack[0][0] < i - s * v:
                            stack.popleft()
                        while stack and stack[-1][1] + (i - stack[-1][0]) // v * w <= dp[i]:
                            stack.pop()
                        stack.append([i, dp[i]])
                        dp[i] = stack[0][1] + (i - stack[0][0]) // v * w
        ac.st(dp[-1])
        return

    @staticmethod
    def lg_p2014(ac=FastIO()):
        # 增|一个虚拟源点将DAG转换为树上背包
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n + 1)]
        nums = [0]
        for i in range(n):
            k, s = ac.read_list_ints()
            nums.append(s)
            dct[k].append(i + 1)
        dp = [[0] * (m + 2) for _ in range(n + 1)]
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                dp[i][1] = nums[i]
                for j in dct[i]:
                    if j != fa:
                        cur = dp[i][:]
                        for x in range(m + 2):
                            for y in range(m + 2 - x):
                                cur[x + y] = ac.max(cur[x + y], dp[i][x] + dp[j][y])
                        dp[i] = cur[:]
        ac.st(dp[0][m + 1])
        return

    @staticmethod
    def lg_p2079(ac=FastIO()):
        # 滚动hash背包DP，两层hash节省空间
        n, v = ac.read_list_ints()
        dp = [defaultdict(lambda: defaultdict(lambda: -inf)), defaultdict(lambda: defaultdict(lambda: -inf))]
        pre = 0
        dp[pre][0][0] = 0
        for i in range(n):
            c, x, y = ac.read_list_ints()
            cur = 1 - pre
            for c1 in dp[pre]:
                for x1 in dp[pre][c1]:
                    if c1 + c <= v:
                        dp[cur][c1 + c][x1 + x] = ac.max(dp[cur][c1 + c][x1 + x], dp[pre][c1][x1] + y)
                    dp[cur][c1][x1] = ac.max(dp[cur][c1][x1], dp[pre][c1][x1])
            pre = cur
        ans = -inf
        for c1 in dp[pre]:
            for x1 in dp[pre][c1]:
                if x1 >= 0:
                    ans = ac.max(ans, dp[pre][c1][x1])
        ac.st(ans)
        return

    @staticmethod
    def lg_p2170(ac=FastIO()):
        # 连通块|二进制01背包优化
        n, m, k = ac.read_list_ints()
        uf = UnionFind(n)
        for _ in range(k):
            i, j = ac.read_list_ints_minus_one()
            uf.union(i, j)
        dct = defaultdict(int)
        for i in range(n):
            dct[uf.find(i)] += 1
        lst = list(dct.values())
        del uf

        # 二进制优化的01背包
        target = ac.min(2 * m, n)
        dp = [0] * (target + 1)
        dp[0] = 1
        cnt = Counter(lst)
        for num in cnt:
            for x in BagDP().bin_split_1(cnt[num]):
                for i in range(target, x * num - 1, -1):
                    if dp[i - x * num]:
                        dp[i] = 1
        ans = 0
        for i in range(1, target + 1):
            if dp[i] and abs(i - m) < abs(ans - m):
                ans = i
        ac.st(ans)
        return

    @staticmethod
    def lg_p2214(ac=FastIO()):
        # 变种背包DPgreedy
        n, b = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(b)]
        voice = [ac.read_int() for _ in range(n)]

        # 从后往前原始得分
        for i in range(n - 1, 0, -1):
            if voice[i - 1] > 0:
                voice[i] -= voice[i - 1] - 1
        ceil = max(voice)
        if any(v < 0 for v in voice):
            ac.st(-1)
            return

        # 完全背包最少数量
        dp = [inf] * (ceil + 1)
        dp[0] = 0
        for num in nums:
            for i in range(num, ceil + 1):
                dp[i] = ac.min(dp[i - num] + 1, dp[i])
        ans = sum(dp[x] for x in voice)
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def lg_p2306(ac=FastIO()):
        # data_range|counter后二进制优化的01背包
        n, m, k = ac.read_list_ints()
        cnt = defaultdict(lambda: defaultdict(int))
        for _ in range(n):
            a, b = ac.read_list_ints()
            cnt[a][b] += 1
        dp = [0] * (m + 1)
        for a in cnt:
            for b in cnt[a]:
                for x in BagDP().bin_split_1(cnt[a][b]):
                    for i in range(m, x * a - 1, -1):
                        dp[i] = ac.max(dp[i], dp[i - x * a] + x * b)
        ans = max(dp)
        if ans >= k:
            ac.st("yes")
        else:
            ac.st("no")
        ac.st(ans)
        return

    @staticmethod
    def lg_p2320(ac=FastIO()):
        # 二进制分解greedy反向
        m = ac.read_int()
        ans = []
        while m:
            ans.append((m + 1) // 2)
            m //= 2
        ac.st(len(ans))
        ac.st(*ans[::-1])
        return

    @staticmethod
    def lg_p2737(ac=FastIO()):
        # 完全背包变种问题
        n = ac.read_int()
        ceil = 256 ** 2 + 1
        nums = [ac.read_int() for _ in range(n)]
        dp = [0] * (ceil + 1)
        dp[0] = 1
        for i in range(1, ceil + 1):
            for num in nums:
                if i >= num and dp[i - num]:
                    dp[i] = 1
        ans = 0
        for i in range(1, ceil + 1):
            if not dp[i]:
                ans = i
        ac.st(ans if ans < ceil else 0)
        return

    @staticmethod
    def lg_p2760(ac=FastIO()):
        # 单调队列优化的多重背包
        m, n, p, t = ac.read_list_ints()
        rest = ac.min(p, t - 1)
        dp = [0] * (rest + 1)
        grid = [ac.read_list_ints() for _ in range(m)]
        mat = [ac.read_list_ints() for _ in range(m)]
        for a in range(m):
            for b in range(n):
                if grid[a][b]:
                    v, w, s = (a + 1 + b + 1) * 2, grid[a][b], mat[a][b]
                    for r in range(v):
                        stack = deque()
                        for i in range(r, rest + 1, v):
                            while stack and stack[0][0] < i - s * v:
                                stack.popleft()
                            while stack and stack[-1][1] + (i - stack[-1][0]) // v * w <= dp[i]:
                                stack.pop()
                            stack.append([i, dp[i]])
                            dp[i] = stack[0][1] + (i - stack[0][0]) // v * w
        ac.st(dp[-1])
        return

    @staticmethod
    def lg_p2854(ac=FastIO()):
        # 分组01背包
        length, n, b = ac.read_list_ints()
        dp = [[-inf] * (b + 1) for _ in range(length + 1)]
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort()
        for x, w, f, c in nums:
            if x == 0:
                if c <= b:
                    dp[x + w][c] = ac.max(dp[x + w][c], f)
            else:
                for i in range(b + 1):
                    if i + c <= b and x + w <= length:
                        dp[x + w][i + c] = ac.max(dp[x + w][i + c], dp[x][i] + f)
        ans = max(dp[length])
        ac.st(ans if ans > -inf else -1)
        return

    @staticmethod
    def lg_p2938(ac=FastIO()):
        # 分组完全背包
        s, d, m = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(s)]
        for i in range(1, d):
            dp = [0] * (m + 1)
            for j in range(s):
                a, b = nums[j][i - 1], nums[j][i]
                if b > a:
                    for p in range(a, m + 1):
                        dp[p] = ac.max(dp[p], dp[p - a] + b)
            m = max(m - i + dp[i] for i in range(m + 1))
        ac.st(m)
        return

    @staticmethod
    def lg_p2979(ac=FastIO()):
        # 分组01背包
        n, t, k = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        m = 5 * t // 4 + 1

        # 先算不缩减高度的
        dp1 = [0] * (m + 1)
        for v, h in nums:
            for i in range(h, m + 1):
                if dp1[i - h] + v > dp1[i]:
                    dp1[i] = dp1[i - h] + v
        ans = dp1[t]
        # brute_force最后一棒高度大于等于 k 的
        for v, h in nums:
            if h >= k:
                for i in range(t, h - 1, -1):
                    ans = ac.max(ans, dp1[(i - h) * 5 // 4] + v)
        ac.st(ans)
        return

    @staticmethod
    def lg_p3010(ac=FastIO()):
        # 变形01背包，两heapq差值最小的分配方案数
        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]
        s = sum(nums)
        mod = 10 ** 6
        # 临界点作为目标值
        t = s // 2
        dp = [0] * (t + 1)  # 背包
        dp[0] = 1
        cnt = [0] * (t + 1)  # 方案数
        cnt[0] = 1
        for num in nums:
            for i in range(t, num - 1, -1):
                if dp[i - num]:
                    dp[i] = 1
                    cnt[i] += cnt[i - num]
                    cnt[i] %= mod

        # brute_force最小差值
        for i in range(t, -1, -1):
            if dp[i]:
                ac.st(s - 2 * i)
                ac.st(cnt[i])
                break
        return

    @staticmethod
    def lg_p3423(ac=FastIO()):
        # 二进制优化多重背包并方案数
        n = ac.read_int()
        b = ac.read_list_ints()
        c = ac.read_list_ints()
        k = ac.read_int()
        dp = [inf] * (k + 1)
        dp[0] = 0
        state = [[] for _ in range(k + 1)]
        for j in range(n):
            bb, cc = b[j], c[j]
            for x in BagDP().bin_split_1(cc):
                for i in range(k, x * bb - 1, -1):
                    if dp[i - x * bb] + x < dp[i]:
                        dp[i] = dp[i - x * bb] + x
                        state[i] = state[i - x * bb][:] + [[bb, x]]
        cnt = defaultdict(int)
        for bb, xx in state[k]:
            cnt[bb] += xx
        ac.st(dp[k])
        ac.lst([cnt[bb] for bb in b])
        return

    @staticmethod
    def lg_p3983(ac=FastIO()):
        # 两个分组完全背包
        n = ac.read_int()
        # 第一个背包每个重量可拆分后的最大价格
        m = 10
        a = [0] + ac.read_list_ints()
        for i in range(1, m + 1):
            for j in range(i + 1):
                a[i] = ac.max(a[i], a[j] + a[i - j])
        # 第二个背包船运载的最大盈利
        cost = [0] + [1, 3, 5, 7, 9, 10, 11, 14, 15, 17]
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(i, n + 1):
                dp[j] = ac.max(dp[j], dp[j - i] + a[i] - cost[i])
        ac.st(dp[-1])
        return

    @staticmethod
    def lg_p5322(ac=FastIO()):
        # 典型二维 DP 转换为分组背包
        s, n, m = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(s)]
        dp = [0] * (m + 1)
        for i in range(n):
            # dp[j] 表示派出 j 个士兵到前 i 个城堡的得分
            lst = [grid[x][i] for x in range(s)]
            lst.sort()
            for j in range(m, -1, -1):
                for ind, w in enumerate(lst):
                    if j <= w * 2:
                        break
                    dp[j] = ac.max(dp[j], dp[j - 2 * w - 1] + (ind + 1) * (i + 1))
        ac.st(dp[-1])
        return

    @staticmethod
    def lg_p5365(ac=FastIO()):
        # 01背包 DP brute_force数量
        n, m = ac.read_list_ints()
        kk = ac.read_list_ints()
        cc = ac.read_list_ints()
        s = sum(kk[i] * cc[i] for i in range(n))
        dp = [0] * (s + 1)
        dp[0] = 1
        for i in range(n):
            k, c = kk[i], cc[i]
            for x in range(s, -1, -1):
                for p in range(1, k + 1):
                    if x < p * c:
                        break
                    dp[x] = ac.max(dp[x], dp[x - p * c] * p)
        for i in range(s + 1):
            if dp[i] >= m:
                ac.st(i)
                break
        return

    @staticmethod
    def lg_p5662(ac=FastIO()):
        # 完全背包变形greedy题目
        t, n, m = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(t)]
        for i in range(1, t):
            dp = [0] * (m + 1)
            for j in range(n):
                b, a = grid[i][j], grid[i - 1][j]
                if b > a:
                    for x in range(a, m + 1):
                        dp[x] = ac.max(dp[x], dp[x - a] + b)
            # 注意此时的 m 更新值
            m = max(m - i + dp[i] for i in range(m + 1))
        ac.st(m)
        return

    @staticmethod
    def lg_p1417(ac=FastIO()):
        # greedysorting后 01 背包最大值
        t, n = ac.read_list_ints()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        c = ac.read_list_ints()
        dp = [0] * (t + 1)
        ind = list(range(n))
        ind.sort(key=lambda it: -b[it] / c[it])
        for i in ind:
            aa, bb, cc = a[i], b[i], c[i]
            for j in range(t, cc - 1, -1):
                dp[j] = ac.max(dp[j], dp[j - cc] + aa - j * bb)
        ac.st(max(dp))
        return

    @staticmethod
    def ac_4081(ac=FastIO()):
        # matrix_dp类似背包思想

        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()

        def check2(xx):
            res = 0
            while xx % 2 == 0:
                res += 1
                xx //= 2
            return res

        def check5(xx):
            res = 0
            while xx % 5 == 0:
                res += 1
                xx //= 5
            return res

        cnt2 = [check2(num) for num in nums]
        cnt5 = [check5(num) for num in nums]

        s5 = sum(cnt5)
        dp = [[-inf] * (s5 + 1) for _ in range(k + 1)]
        dp[0][0] = 0
        for i in range(n):
            a2 = cnt2[i]
            a5 = cnt5[i]
            for j in range(k, 0, -1):
                for p in range(s5, a5 - 1, -1):
                    x, y = dp[j][p], dp[j - 1][p - a5] + a2
                    if y > x:
                        dp[j][p] = y
        ans = 0
        for a5 in range(s5 + 1):
            cur = ac.min(dp[k][a5], a5)
            if cur > ans:
                ans = cur
        ac.st(ans)
        return

    @staticmethod
    def lc_100029(nums: List[int], ll: int, r: int) -> int:
        # 按照单调队列的思想mod|分组DP，prefix_sum优化，也有容斥的思想
        cnt = Counter(nums)
        mod = 10 ** 9 + 7
        dp = [0] * (r + 1)
        dp[0] = 1
        for num in cnt:
            if num:
                c = cnt[num]
                for i in range(num):
                    pre = [0]
                    x = 0
                    for j in range(i, r + 1, num):
                        val = pre[-1] + dp[j]
                        dp[j] += pre[x]
                        if x - c >= 0:
                            dp[j] -= pre[x - c]
                        dp[j] %= mod
                        pre.append(val % mod)
                        x += 1
        return sum(dp[ll:]) * (cnt[0] + 1) % mod

    @staticmethod
    def lc_1049(stones: List[int]) -> int:
        # 问题，转化为01背包求解
        s = sum(stones)
        dp = [0] * (s // 2 + 1)
        dp[0] = 1
        for num in stones:
            for i in range(s // 2, num - 1, -1):
                if dp[i - num]:
                    dp[i] = 1
        return min(abs(s - 2 * i) for i in range(s // 2 + 1) if dp[i])