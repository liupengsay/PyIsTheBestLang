
import unittest
from typing import List

from algorithm.src.fast_io import FastIO, inf
from collections import Counter

"""
算法：线性DP
功能：遍历数组，根据前序或者后序结果进行更新，最大非空连续子序列和
题目：

===================================力扣===================================
2361. 乘坐火车路线的最少费用（https://leetcode.cn/problems/minimum-costs-using-the-train-line/）当前状态只跟前一个状态有关
2318. 不同骰子序列的数目（https://leetcode.cn/problems/number-of-distinct-roll-sequences/）当前状态只跟前一个状态有关使用枚举计数
2263. 数组变为有序的最小操作次数（https://leetcode.cn/problems/make-array-non-decreasing-or-non-increasing/）当前状态只跟前一个状态有关
2209. 用地毯覆盖后的最少白色砖块（https://leetcode.cn/problems/minimum-white-tiles-after-covering-with-carpets/）前缀优化与处理进行转移
2188. 完成比赛的最少时间（https://leetcode.cn/problems/minimum-time-to-finish-the-race/）预处理DP
2167. 移除所有载有违禁货物车厢所需的最少时间（https://leetcode.cn/problems/minimum-time-to-remove-all-cars-containing-illegal-goods/）使用前缀后缀DP预处理后进行枚举
2431. 最大限度地提高购买水果的口味（https://leetcode.cn/problems/maximize-total-tastiness-of-purchased-fruits/）线性DP进行模拟计算
6355. 质数减法运算（https://leetcode.cn/contest/weekly-contest-338/problems/collect-coins-in-a-tree/）线性DP
2547. 拆分数组的最小代价（https://leetcode.cn/problems/minimum-cost-to-split-an-array/）线性DP并使用一个变量维护计数
2638. Count the Number of K-Free Subsets（https://leetcode.cn/problems/count-the-number-of-k-free-subsets/）线性DP计数
2597. 美丽子集的数目（https://leetcode.cn/problems/the-number-of-beautiful-subsets/）线性DP计数


===================================洛谷===================================
P1970 [NOIP2013 提高组] 花匠（https://www.luogu.com.cn/problem/P1970）使用贪心与动态规划计算最长的山脉子数组
P1564 膜拜（https://www.luogu.com.cn/problem/P1564）线性DP
P1481 魔族密码（https://www.luogu.com.cn/problem/P1481）线性DP
P2029 跳舞（https://www.luogu.com.cn/problem/P2029）线性DP
P2031 脑力达人之分割字串（https://www.luogu.com.cn/problem/P2031）线性DP
P2062 分队问题（https://www.luogu.com.cn/problem/P2062）线性DP+前缀最大值DP剪枝优化
P2072 宗教问题（https://www.luogu.com.cn/problem/P2072）两个线性DP

P2096 最佳旅游线路（https://www.luogu.com.cn/problem/P2096）最大连续子序列和变种
P5761 [NOI1997] 最佳游览（https://www.luogu.com.cn/problem/P5761）最大连续子序列和变种
P2285 [HNOI2004]打鼹鼠（https://www.luogu.com.cn/problem/P2285）线性DP+前缀最大值DP剪枝优化

P2642 双子序列最大和（https://www.luogu.com.cn/problem/P2642）枚举前后两个非空的最大子序列和
P1470 [USACO2.3]最长前缀 Longest Prefix（https://www.luogu.com.cn/problem/P1470）线性DP
P1096 [NOIP2007 普及组] Hanoi 双塔问题（https://www.luogu.com.cn/problem/P1096）经典线性DP

P2896 [USACO08FEB]Eating Together S（https://www.luogu.com.cn/problem/P2896）前后缀动态规划
P2904 [USACO08MAR]River Crossing S（https://www.luogu.com.cn/problem/P2904）前缀和预处理加线性DP

P3062 [USACO12DEC]Wifi Setup S（https://www.luogu.com.cn/problem/P3062）线性DP枚举

P3842 [TJOI2007]线段（https://www.luogu.com.cn/problem/P3842）线性DP进行模拟
P3903 导弹拦截III（https://www.luogu.com.cn/problem/P3903）线性DP枚举当前元素作为谷底与山峰的子序列长度
P5414 [YNOI2019] 排序（https://www.luogu.com.cn/problem/P5414）贪心，使用线性DP计算最大不降子序列和
P6191 [USACO09FEB]Bulls And Cows S（https://www.luogu.com.cn/problem/P6191）线性DP枚举计数
P6208 [USACO06OCT] Cow Pie Treasures G（https://www.luogu.com.cn/problem/P6208）线性DP模拟
P7404 [JOI 2021 Final] とてもたのしい家庭菜園 4（https://www.luogu.com.cn/problem/P7404）动态规划枚举，计算变成山脉数组的最少操作次数
P7541 [COCI2009-2010#1] DOBRA（https://www.luogu.com.cn/problem/P7541）线性DP记忆化搜索，类似数位DP
P7767 [COCI 2011/2012 #5] DNA（https://www.luogu.com.cn/problem/P7767）线性DP，计算前缀变成全部相同字符的最少操作次数
P2246 SAC#1 - Hello World（升级版）（https://www.luogu.com.cn/problem/P2246）字符串计数线性DP
P4933 大师（https://www.luogu.com.cn/problem/P4933）线性DP使用等差数列计数
P1874 快速求和（https://www.luogu.com.cn/problem/P1874）线性DP
P2513 [HAOI2009]逆序对数列（https://www.luogu.com.cn/problem/P2513）前缀和优化DP

================================CodeForces================================
https://codeforces.com/problemset/problem/75/D（经典压缩数组，最大子段和升级）
https://codeforces.com/problemset/problem/1084/C（线性DP加前缀和优化）
https://codeforces.com/problemset/problem/166/E（线性DP计数）
https://codeforces.com/problemset/problem/1221/D（线性DP模拟）
C. Chef Monocarp（https://codeforces.com/problemset/problem/1437/C）二维线性DP，两个数组线性移动进行匹配计算最大或者最小值
D. Armchairs（https://codeforces.com/problemset/problem/1525/D）二维线性DP，两个数组线性移动进行匹配计算最大或者最小值
A. Garland（https://codeforces.com/problemset/problem/1286/A）线性经典dp
D. Make The Fence Great Again（https://codeforces.com/problemset/problem/1221/D）线性DP，最多变化为增加0、1、2

参考：OI WiKi（xx）
"""


class LinearDP:
    def __init__(self):
        return

    @staticmethod
    def liner_dp_template(nums):
        # 线性 DP 递推模板（以最长上升子序列长度为例）
        n = len(nums)
        dp = [0] * (n + 1)
        for i in range(n):
            dp[i + 1] = 1
            for j in range(i):
                if nums[i] > nums[j] and dp[j] + 1 > dp[i + 1]:
                    dp[i + 1] = dp[j] + 1
        return max(dp)


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_2361(regular: List[int], express: List[int], express_cost: int) -> List[int]:
        # 模板：线性 DP 转移
        n = len(regular)
        cost = [[0, 0] for _ in range(n + 1)]
        cost[0][1] = express_cost
        for i in range(1, n + 1):
            cost[i][0] = min(cost[i - 1][0] + regular[i - 1], cost[i - 1][1] + express[i - 1])
            cost[i][1] = min(cost[i][0] + express_cost, cost[i - 1][1] + express[i - 1])
        return [min(c) for c in cost[1:]]

    @staticmethod
    def cf_1286a(ac=FastIO()):

        n = ac.read_int()
        nums = ac.read_list_ints()
        ex = set(nums)
        cnt = Counter([i % 2 for i in range(1, n + 1) if i not in ex])

        # 模板：经典记忆化搜索的模拟线性DP写法
        @ac.bootstrap
        def dfs(i, single, double, pre):
            if (i, single, double, pre) in dct:
                yield
            if i == n:
                dct[(i, single, double, pre)] = 0
                yield
            res = inf
            if nums[i] != 0:
                v = nums[i] % 2
                yield dfs(i + 1, single, double, v)
                cur = dct[(i + 1, single, double, v)]
                if pre != -1 and pre != v:
                    cur += 1
                res = ac.min(res, cur)
            else:
                if single:
                    yield dfs(i + 1, single - 1, double, 1)
                    cur = dct[(i + 1, single - 1, double, 1)]
                    if pre != -1 and pre != 1:
                        cur += 1
                    res = ac.min(res, cur)
                if double:
                    yield dfs(i + 1, single, double - 1, 0)
                    cur = dct[(i + 1, single, double - 1, 0)]
                    if pre != -1 and pre != 0:
                        cur += 1
                    res = ac.min(res, cur)
            dct[(i, single, double, pre)] = res
            yield

        dct = dict()
        dfs(0, cnt[1], cnt[0], -1)
        ac.st(dct[(0, cnt[1], cnt[0], -1)])
        return

    @staticmethod
    def lc_2638(nums: List[int], k: int) -> int:
        # 模板：线性DP计数
        n = len(nums)
        dp = [1] * (n+1)
        dp[1] = 2
        for i in range(2, 51):
            dp[i] = dp[i - 1] + dp[i - 2]
        dct = set(nums)
        ans = 1
        for num in nums:
            if num - k not in dct:
                cnt = 0
                while num in dct:
                    cnt += 1
                    num += k
                ans *= dp[cnt]
        return ans

    @staticmethod
    def lc_2597(nums: List[int], k: int) -> int:
        # 模板：线性DP计数
        power = [1 << i for i in range(21)]

        def check(tmp):
            m = len(tmp)
            dp = [1] * (m + 1)
            dp[1] = power[tmp[0]] - 1 + dp[0]
            for i in range(1, m):
                dp[i + 1] = dp[i - 1] * (power[tmp[i]] - 1) + dp[i]
            return dp[-1]

        cnt = Counter(nums)
        ans = 1
        for num in cnt:
            if num - k not in cnt:
                lst = []
                while num in cnt:
                    lst.append(cnt[num])
                    num += k
                ans *= check(lst)
        return ans - 1

    @staticmethod
    def cf_1525d(ac=FastIO()):
        n = ac.read_int()
        nums = ac.read_list_ints()
        occu = [i for i in range(n) if nums[i]]
        free = [i for i in range(n) if not nums[i]]
        if not occu:
            ac.st(0)
            return
        a, b = len(occu), len(free)
        dp = [[inf] * (b + 1) for _ in range(a + 1)]
        dp[0] = [0] * (b + 1)
        for i in range(a):
            for j in range(b):
                dp[i + 1][j + 1] = ac.min(dp[i + 1][j], dp[i][j] + abs(occu[i] - free[j]))
        ac.st(dp[-1][-1])
        return

    @staticmethod
    def cf_1437c(n, nums):
        # 模板：两个数组线性移动进行匹配计算最大或者最小值
        nums.sort()
        m = 2 * n
        dp = [[float("inf")] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        for i in range(m):
            dp[i + 1][0] = 0
            for j in range(n):
                dp[i + 1][j + 1] = min(dp[i][j + 1], dp[i][j] + abs(nums[j] - i - 1))
        return dp[m][n]

    @staticmethod
    def lg_p4933(ac=FastIO()):
        # 模板：不同等差子序列的个数
        n = ac.read_int()
        nums = ac.read_list_ints()
        mod = 998244353
        ans = n
        dp = [defaultdict(int) for _ in range(n)]
        for i in range(n):
            for j in range(i):
                dp[i][nums[i]-nums[j]] += dp[j][nums[i]-nums[j]] + 1
                dp[i][nums[i] - nums[j]] %= mod
            for j in dp[i]:
                ans += dp[i][j]
                ans %= mod
        ac.st(ans)
        return

class TestGeneral(unittest.TestCase):

    def test_linear_dp(self):
        ld = LinearDP()
        nums = [6, 3, 5, 2, 1, 6, 8, 9]
        assert ld.liner_dp_template(nums) == 4
        return


if __name__ == '__main__':
    unittest.main()
