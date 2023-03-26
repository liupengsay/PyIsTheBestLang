import bisect
import unittest
from typing import List

from algorithm.src.fast_io import FastIO, inf

"""

算法：二分查找
功能：利用单调性确定最优选择，通常可以使用SortedList、Bisect，还可以使用精度控制二分
题目：xx（xx）
===================================力扣===================================
4. 寻找两个正序数组的中位数（https://leetcode.cn/problems/median-of-two-sorted-arrays/）经典二分思想查找题
2426 满足不等式的数对数目（https://leetcode.cn/problems/number-of-pairs-satisfying-inequality/）根据不等式变换和有序集合进行二分查找
2179 统计数组中好三元组数目（https://leetcode.cn/problems/count-good-triplets-in-an-array/）维护区间范围内的个数
2141 同时运行 N 台电脑的最长时间（https://leetcode.cn/problems/maximum-running-time-of-n-computers/）贪心选择最大的 N 个电池作为基底，然后二分确定在其余电池的加持下可以运行的最长时间
2102 序列顺序查询（https://leetcode.cn/problems/sequentially-ordinal-rank-tracker/）使用有序集合维护优先级姓名实时查询
2563. 统计公平数对的数目（https://leetcode.cn/problems/count-the-number-of-fair-pairs/）使用二分查找确定范围个数

===================================洛谷===================================
P1577 切绳子（https://www.luogu.com.cn/problem/P1577）数学整除向下取整与二分
P1570 KC 喝咖啡（https://www.luogu.com.cn/problem/P1570）公式转换后使用贪心加二分
P1843 奶牛晒衣服（https://www.luogu.com.cn/problem/P1843）贪心加二分
P2309 loidc，卖卖萌（https://www.luogu.com.cn/problem/P2309）使用前缀和有序列表加二分求解和为正数的子串个数
P2390 地标访问（https://www.luogu.com.cn/problem/P2390）枚举加二分起始也可以使用双指针
P2759 奇怪的函数（https://www.luogu.com.cn/problem/P2759）公式变换后使用二分求解
P1404 平均数（https://www.luogu.com.cn/problem/P1404）公式变换后使用前缀和加二分
P2855 [USACO06DEC]River Hopscotch S（https://www.luogu.com.cn/problem/P2855）使用贪心加二分
P2884 [USACO07MAR]Monthly Expense S（https://www.luogu.com.cn/problem/P2884）最大最小之类的经典二分问题
P2985 [USACO10FEB]Chocolate Eating S（https://www.luogu.com.cn/problem/P2985）使用贪心加二分进行模拟
P3184 [USACO16DEC]Counting Haybales S（https://www.luogu.com.cn/problem/P3184）二分查找区间范围内个数
P3611 [USACO17JAN]Cow Dance Show S（https://www.luogu.com.cn/problem/P3611）二分贪心加堆优化模拟
P3743 kotori的设备（https://www.luogu.com.cn/problem/P3743）经典二分查找注意check函数
P4058 [Code+#1]木材（https://www.luogu.com.cn/problem/P4058）经典二分查找注意check函数
P4670 [BalticOI 2011 Day2]Plagiarism（https://www.luogu.com.cn/problem/P4670）排序后二分查找计数
P5119 [USACO18DEC]Convention S（https://www.luogu.com.cn/problem/P5119）经典贪心加二分问题
P5250 【深基17.例5】木材仓库（https://www.luogu.com.cn/problem/P5250）维护一个有序集合
P6174 [USACO16JAN]Angry Cows S（https://www.luogu.com.cn/problem/P6174）经典贪心加二分问题
P6281 [USACO20OPEN] Social Distancing S（https://www.luogu.com.cn/problem/P6281）经典贪心加二分问题
P6423 [COCI2008-2009#2] SVADA（https://www.luogu.com.cn/problem/P6423）利用单调性进行二分计算
P7177 [COCI2014-2015#4] MRAVI（https://www.luogu.com.cn/problem/P7177）二分加树形dfs模拟

================================CodeForces================================
https://codeforces.com/problemset/problem/1251/D（使用贪心进行中位数二分求解）
https://codeforces.com/problemset/problem/830/A（使用贪心进行距离点覆盖二分求解）
https://codeforces.com/problemset/problem/847/E（使用贪心二分与指针进行判断）
https://codeforces.com/problemset/problem/732/D（使用贪心二分与指针进行判断）
https://codeforces.com/problemset/problem/778/A（二分和使用指针判断是否check
https://codeforces.com/problemset/problem/913/C（DP预处理最优单价，再二分加贪心进行模拟求解）

G2. Teleporters (Hard Version)（https://codeforces.com/problemset/problem/1791/G2）贪心排序，前缀和枚举二分
D. Multiplication Table（https://codeforces.com/problemset/problem/448/D）经典二分查找计算n*m的乘法表第k大元素
D. Cleaning the Phone（https://codeforces.com/problemset/problem/1475/D）贪心排序，前缀和枚举二分
D. Odd-Even Subsequence（https://codeforces.com/problemset/problem/1370/D）利用单调性二分，再使用贪心check
D. Max Median（https://codeforces.com/problemset/problem/1486/D）利用单调性二分，再使用经典哈希前缀和计算和为正数的最长连续子序列

参考：OI WiKi（xx）
"""


class BinarySearch:
    def __init__(self):
        return

    @staticmethod
    def find_int_left(low, high, check):
        # 模板: 整数范围内二分查找，选择最靠左满足check
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                high = mid
            else:
                low = mid
        return low if check(low) else high

    @staticmethod
    def find_int_right(low, high, check):
        # 模板: 整数范围内二分查找，选择最靠右满足check
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                low = mid
            else:
                high = mid
        return high if check(high) else low

    @staticmethod
    def find_float_left(low, high, check, error=1e-6):
        # 模板: 浮点数范围内二分查找, 选择最靠左满足check
        while low < high - error:
            mid = low + (high - low) / 2
            if check(mid):
                high = mid
            else:
                low = mid
        return low if check(low) else high

    @staticmethod
    def find_float_right(low, high, check, error=1e-6):
        # 模板: 浮点数范围内二分查找, 选择最靠右满足check
        while low < high - error:
            mid = low + (high - low) / 2
            if check(mid):
                low = mid
            else:
                high = mid
        return high if check(high) else low


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_448d(ac=FastIO()):
        # 模板：计算 n*m 乘法矩阵内的第 k 大元素
        n, m, k = ac.read_ints()

        def check(num):
            res = 0
            for x in range(1, m + 1):
                res += min(n, num // x)
            return res >= k

        # 初始化大小
        if m > n:
            m, n = n, m

        ans = BinarySearch().find_int_left(1, m * n, check)
        ac.st(ans)
        return

    @staticmethod
    def cf_1370d(ac=FastIO()):
        n, k = map(int, input().split())
        nums = list(map(int, input().split()))

        def check(x):
            # 奇偶交替，依次枚举奇数索引与偶数索引不超过x
            for ind in [0, 1]:
                cnt = 0
                for num in nums:
                    if not ind:
                        cnt += 1
                        ind = 1 - ind
                    else:
                        if num <= x:
                            cnt += 1
                            ind = 1 - ind
                    if cnt >= k:
                        return True
            return False

        low = min(nums)
        high = max(nums)
        ans = BinarySearch().find_int_left(low, high, check)
        ac.st(ans)
        return

    @staticmethod
    def cf_1475d(ac=FastIO()):
        # 模板：贪心排序后，枚举并使用前缀和进行二分查询
        for _ in range(ac.read_int()):
            n, m = ac.read_ints()
            a = ac.read_list_ints()
            b = ac.read_list_ints()
            if sum(a) < m:
                ac.st(-1)
                continue

            # 排序
            a1 = [a[i] for i in range(n) if b[i] == 1]
            a2 = [a[i] for i in range(n) if b[i] == 2]
            a1.sort(reverse=True)
            a2.sort(reverse=True)

            # 前缀和
            x, y = len(a1), len(a2)
            pre1 = [0] * (x + 1)
            for i in range(x):
                pre1[i + 1] = pre1[i] + a1[i]

            # 初始化后进行二分枚举
            ans = inf
            pre = 0
            j = bisect.bisect_left(pre1, m - pre)
            if j <= x:
                ans = ac.min(ans, j)

            for i in range(y):
                cnt = i + 1
                pre += a2[i]
                j = bisect.bisect_left(pre1, m - pre)
                if j <= x:
                    ans = ac.min(ans, j + cnt * 2)
            ac.st(ans)
        return

    @staticmethod
    def cf_1791g2(ac=FastIO()):
        # 模板：find_int_right
        for _ in range(ac.read_int()):

            n, c = ac.read_ints()
            cost = ac.read_list_ints()
            lst = [[ac.min(x, n + 1 - x) + cost[x - 1], x + cost[x - 1]]
                   for x in range(1, n + 1)]
            lst.sort(key=lambda it: it[0])
            pre = [0] * (n + 1)
            for i in range(n):
                pre[i + 1] = pre[i] + lst[i][0]

            def check(y):
                res = pre[y]
                if y > i:
                    res -= lst[i][0]
                res += lst[i][1]
                return res <= c

            ans = 0
            for i in range(n):
                if lst[i][1] <= c:
                    cur = BinarySearch().find_int_right(0, n, check)
                    cur = cur - int(cur > i) + 1
                    ans = ac.max(ans, cur)
            ac.st(ans)
        return

    @staticmethod
    def lc_2563(nums, lower, upper):
        # 模板：查找数值和在一定范围的数对个数
        nums.sort()
        n = len(nums)
        ans = 0
        for i in range(n):
            x = bisect.bisect_right(nums, upper-nums[i], hi=i)
            y = bisect.bisect_left(nums, lower-nums[i], hi=i)
            ans += x - y
        return ans

    @staticmethod
    def lc_4(nums1: List[int], nums2: List[int]) -> float:
        # 模板：经典双指针二分移动查找两个正序数组的中位数
        def get_kth_num(k):
            ind1 = ind2 = 0
            while k:
                if ind1 == m:
                    return nums2[ind2 + k - 1]
                if ind2 == n:
                    return nums1[ind1 + k - 1]
                if k == 1:
                    return min(nums1[ind1], nums2[ind2])
                index1 = min(ind1 + k // 2 - 1, m - 1)
                index2 = min(ind2 + k // 2 - 1, n - 1)
                val1 = nums1[index1]
                val2 = nums2[index2]
                if val1 < val2:
                    k -= index1 - ind1 + 1
                    ind1 = index1 + 1
                else:
                    k -= index2 - ind2 + 1
                    ind2 = index2 + 1
            return

        m, n = len(nums1), len(nums2)
        s = m + n
        if s % 2:
            return get_kth_num(s // 2 + 1)
        else:
            return (get_kth_num(s // 2 + 1) + get_kth_num(s // 2)) / 2

    @staticmethod
    def cf_1486d(ac=FastIO()):
        # 模板：经典转换为二分和哈希前缀求最长和为正数的最长连续子序列
        n, k = ac.read_ints()
        nums = ac.read_list_ints()
        low = 0
        high = n - 1
        lst = sorted(nums)

        def check(x):
            x = lst[x]
            dct = dict()
            pre = 0
            dct[0] = -1
            for i, num in enumerate(nums):
                pre += 1 if num >= x else -1
                if pre > 0 and i + 1 >= k:
                    return True
                # 为负数时，只需贪心考虑第一次为 pre-1 时的长度即可
                if pre - 1 in dct and i - dct[pre - 1] >= k:
                    return True
                if pre not in dct:
                    dct[pre] = i
            return False

        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                low = mid
            else:
                high = mid
        ans = high if check(high) else low
        ac.st(lst[ans])
        return


class TestGeneral(unittest.TestCase):

    def test_binary_search(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
