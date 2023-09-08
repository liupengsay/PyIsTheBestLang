import bisect
import math
import unittest
from collections import deque, defaultdict
from itertools import accumulate
from typing import List, Callable
from math import inf

from src.data_structure.sorted_list import LocalSortedList
from src.fast_io import FastIO
from src.graph.lca import OfflineLCA
from src.graph.union_find import UnionFind

"""

算法：二分查找
功能：利用单调性确定最优选择，通常可以使用SortedList、Bisect，还可以使用精度控制二分
题目：xx（xx）
===================================力扣===================================
4. 寻找两个正序数组的中位数（https://leetcode.cn/problems/median-of-two-sorted-arrays/）经典二分思想查找题
81. 搜索旋转排序数组 II（https://leetcode.cn/problems/search-in-rotated-sorted-array-ii/）经典有重复数字的旋转数组
154. 寻找旋转排序数组中的最小值 II（https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array-ii/）经典有重复数字的旋转数组
162. 寻找峰值（https://leetcode.cn/problems/find-peak-element/）经典二分思想查找题
2426. 满足不等式的数对数目（https://leetcode.cn/problems/number-of-pairs-satisfying-inequality/）根据不等式变换和有序集合进行二分查找
2179. 统计数组中好三元组数目（https://leetcode.cn/problems/count-good-triplets-in-an-array/）维护区间范围内的个数
2141. 同时运行 N 台电脑的最长时间（https://leetcode.cn/problems/maximum-running-time-of-n-computers/）贪心选择最大的 N 个电池作为基底，然后二分确定在其余电池的加持下可以运行的最长时间
2102. 序列顺序查询（https://leetcode.cn/problems/sequentially-ordinal-rank-tracker/）使用有序集合维护优先级姓名实时查询
2563. 统计公平数对的数目（https://leetcode.cn/problems/count-the-number-of-fair-pairs/）使用二分查找确定范围个数
2604. 吃掉所有谷子的最短时间（https://leetcode.cn/problems/minimum-time-to-eat-all-grains/）二分加指针贪心 check
1201. 丑数 III（https://leetcode.cn/problems/ugly-number-iii/）二分加容斥原理计数
1739. 放置盒子（https://leetcode.cn/problems/building-boxes/）可推公式二分也可数学方法计算
1889. 装包裹的最小浪费空间（https://leetcode.cn/problems/minimum-space-wasted-from-packaging/）排序加前缀和预处理与贪心二分
2071. 你可以安排的最多任务数目（https://leetcode.cn/problems/maximum-number-of-tasks-you-can-assign/）经典二分加贪心
2594. 修车的最少时间（https://leetcode.cn/problems/minimum-time-to-repair-cars/）经典二分
2517. 礼盒的最大甜蜜度（https://leetcode.cn/problems/maximum-tastiness-of-candy-basket/）经典二分
1482. 制作 m 束花所需的最少天数（https://leetcode.cn/problems/minimum-number-of-days-to-make-m-bouquets/）经典二分
2528. 最大化城市的最小供电站数目（https://leetcode.cn/problems/maximize-the-minimum-powered-city/description/）经典二分使用前缀和差分数组贪心验证
2560. 打家劫舍 IV（https://leetcode.cn/problems/house-robber-iv/）经典二分DP来check

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
P1314 [NOIP2011 提高组] 聪明的质监员（https://www.luogu.com.cn/problem/P1314）经典二分寻找最接近目标值的和
P3017 [USACO11MAR]Brownie Slicing G（https://www.luogu.com.cn/problem/P3017）经典二分将矩阵分成a*b个子矩阵且子矩阵和的最小值最大
P1083 [NOIP2012 提高组] 借教室（https://www.luogu.com.cn/problem/P1083）经典二分结合差分进行寻找第一个失效点
P1281 书的复制（https://www.luogu.com.cn/problem/P1281）典型二分并输出方案
P1381 单词背诵（https://www.luogu.com.cn/problem/P1381）典型二分加滑动窗口check，脑筋急转弯
P1419 寻找段落（https://www.luogu.com.cn/problem/P1419）二分加优先队列
P1525 [NOIP2010 提高组] 关押罪犯（https://www.luogu.com.cn/problem/P1525）经典二分加BFS进行二分图划分，可以使用染色法或者并查集
P1542 包裹快递（https://www.luogu.com.cn/problem/P1542）二分加使用分数进行高精度计算
P2237 [USACO14FEB]Auto-complete S（https://www.luogu.com.cn/problem/P2237）脑筋急转弯排序后二分查找
P2810 Catch the theives（https://www.luogu.com.cn/problem/P2810）二分加枚举
P3718 [AHOI2017初中组]alter（https://www.luogu.com.cn/problem/P3718）二分加贪心
P3853 [TJOI2007]路标设置（https://www.luogu.com.cn/problem/P3853）经典二分贪心题
P4343 [SHOI2015]自动刷题机（https://www.luogu.com.cn/problem/P4343）上下界二分加模拟
P5844 [IOI2011]ricehub（https://www.luogu.com.cn/problem/P5844）经典中位数贪心与前缀和二分
P5878 奖品（https://www.luogu.com.cn/problem/P5878）经典二分加枚举
P6004 [USACO20JAN] Wormhole Sort S（https://www.luogu.com.cn/problem/P6004）经典二分加并查集
P6058 [加油武汉]体温调查（https://www.luogu.com.cn/problem/P6058）使用深搜序与离线 LCA 计算相邻叶子之间距离并二分确定时间
P6069 『MdOI R1』Group（https://www.luogu.com.cn/problem/P6069）经典方差计算公式变形，使用二分加变量维护区间的方差值大小
P6733 「Wdsr-2」间歇泉（https://www.luogu.com.cn/problem/P6733）二分加STL进行 Check
P8161 [JOI 2022 Final] 自学 (Self Study)（https://www.luogu.com.cn/problem/P8161）经典贪心加二分求解
P8198 [传智杯 #4 决赛] 背单词的小智（https://www.luogu.com.cn/problem/P8198）经典二分加指针
P9050 [PA2021] Sumy（https://www.luogu.com.cn/problem/P9050）经典二分注意值域区间与贪心模拟

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
D2. Coffee and Coursework (Hard Version)（https://codeforces.com/problemset/problem/1118/D2）利用单调性贪心二分
I. Photo Processing（https://codeforces.com/problemset/problem/883/I）二分加双指针dp


================================AtCoder================================
D - No Need （https://atcoder.jp/contests/abc056/tasks/arc070_b）经典利用单调性进行二分，用背包DP进行check
D - Widespread（https://atcoder.jp/contests/abc063/tasks/arc075_b）经典利用单调性进行二分，贪心进行check

================================AcWing================================
120. 防线（https://www.acwing.com/problem/content/122/）根据单调性二分
14. 不修改数组找出重复的数字（https://www.acwing.com/problem/content/description/15/）利用鸽巢原理二分查找重复的数，修改数组且只用O(1)空间
3973. 无线网络（https://www.acwing.com/problem/content/3976/）浮点数二分与滑动窗口双指针
4863. 构造新矩阵（https://www.acwing.com/problem/content/description/4866/）经典二分加鸽巢原理
5048. 无线网络（https://www.acwing.com/problem/content/description/5051/）经典浮点数二分并求出具体方案

参考：OI WiKi（xx）
"""


class BinarySearch:
    def __init__(self):
        return

    @staticmethod
    def find_int_left(low: int, high: int, check: Callable) -> int:
        # 模板: 整数范围内二分查找，选择最靠左满足check
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                high = mid
            else:
                low = mid
        return low if check(low) else high

    @staticmethod
    def find_int_right(low: int, high: int, check: Callable) -> int:
        # 模板: 整数范围内二分查找，选择最靠右满足check
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                low = mid
            else:
                high = mid
        return high if check(high) else low

    @staticmethod
    def find_float_left(low: float, high: float, check: Callable, error=1e-6) -> float:
        # 模板: 浮点数范围内二分查找, 选择最靠左满足check
        while low < high - error:
            mid = low + (high - low) / 2
            if check(mid):
                high = mid
            else:
                low = mid
        return low if check(low) else high

    @staticmethod
    def find_float_right(low: float, high: float, check: Callable, error=1e-6) -> float:
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
    def lg_p1314(ac=FastIO()):

        # 模板：经典二分寻找最接近目标值的和
        n, m, s = ac.read_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        queries = [ac.read_list_ints() for _ in range(m)]

        def check(w):
            cnt = [0] * (n + 1)
            pre = [0] * (n + 1)
            for i in range(n):
                cnt[i + 1] = cnt[i] + int(nums[i][0] >= w)
                pre[i + 1] = pre[i] + int(nums[i][0] >= w) * nums[i][1]
            res = 0
            for a, b in queries:
                res += (pre[b] - pre[a - 1]) * (cnt[b] - cnt[a - 1])
            return res

        ans = inf
        low = 0
        high = max(ls[0] for ls in nums)
        while low < high - 1:
            mid = low + (high - low) // 2
            x = check(mid)
            ans = ac.min(ans, abs(s - x))
            if x <= s:
                high = mid - 1
            else:
                low = mid + 1
        ans = ac.min(ans, ac.min(abs(s - check(low)), abs(s - check(high))))
        ac.st(ans)
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
    def lg_p3017(ac=FastIO()):

        # 模板：经典二分将矩阵分成a*b个子矩阵且子矩阵和的最小值最大
        def check(x):

            def cut():
                cur = 0
                c = 0
                for num in pre:
                    cur += num
                    if cur >= x:
                        c += 1
                        cur = 0
                return c >= b

            cnt = i = 0
            pre = [0] * n
            while i < m:
                if cut():
                    pre = [0] * n
                    cnt += 1
                else:
                    for j in range(n):
                        pre[j] += grid[i][j]
                    i += 1
            if cut():
                cnt += 1
            return cnt >= a

        m, n, a, b = ac.read_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        low = 0
        high = sum(sum(g) for g in grid) // (a * b)
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                low = mid
            else:
                high = mid
        ans = high if check(high) else low
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
    def lc_1889(packages: List[int], boxes: List[List[int]]) -> int:
        # 模板：排序加前缀和预处理与贪心二分
        packages.sort()
        pre = list(accumulate(packages, initial=0))
        n = len(packages)
        ans = inf
        mod = 10 ** 9 + 7
        for box in boxes:
            box.sort()
            if box[-1] < packages[-1]:
                continue
            i = cur = 0
            for num in box:
                if i == n:
                    break
                if num < packages[i]:
                    continue

                j = bisect.bisect_right(packages, num) - 1
                cur += (j - i + 1) * num - (pre[j + 1] - pre[i])
                i = j + 1
            if cur < ans:
                ans = cur
        return ans % mod if ans < inf else -1

    @staticmethod
    def lc_2141(n: int, batteries: List[int]) -> int:
        # 模板：经典贪心二分

        batteries.sort(reverse=True)
        rest = sum(batteries[n:])

        def check(w):
            res = 0
            for num in batteries[:n]:
                if num < w:
                    res += w - num
            return res <= rest

        return BinarySearch().find_int_right(0, batteries[n - 1] + rest, check)

    @staticmethod
    def lc_2528(stations: List[int], r: int, k: int) -> int:
        # 模板：经典二分使用前缀和差分数组贪心验证
        n = len(stations)
        nums = [0] * n
        for i in range(n):
            left = max(0, i - r)
            nums[left] += stations[i]
            if i + r + 1 < n:
                nums[i + r + 1] -= stations[i]
        for i in range(1, n):
            nums[i] += nums[i - 1]

        def check(x):
            diff = [0] * (n + 2 * r + 10)
            res = 0
            for j in range(n):
                diff[j] += diff[j - 1] if j else 0
                cur = diff[j] + nums[j]
                if cur < x:
                    res += x - cur
                    diff[j] += x - cur
                    diff[j + 2 * r + 1] -= x - cur

            return res <= k

        return BinarySearch().find_int_right(0, max(nums) + k, check)

    @staticmethod
    def lc_2563(nums, lower, upper):
        # 模板：查找数值和在一定范围的数对个数
        nums.sort()
        n = len(nums)
        ans = 0
        for i in range(n):
            x = bisect.bisect_right(nums, upper - nums[i], hi=i)
            y = bisect.bisect_left(nums, lower - nums[i], hi=i)
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

    @staticmethod
    def lg_p1083(ac=FastIO()):

        # 模板：经典二分结合差分进行寻找第一个失效点

        def check(s):
            diff = [0] * n
            for c, a, b in lst[:s]:
                diff[a - 1] += c
                if b < n:
                    diff[b] -= c
            if diff[0] > nums[0]:
                return False
            pre = diff[0]
            for i in range(1, n):
                pre += diff[i]
                if pre > nums[i]:
                    return False
            return True

        n, m = ac.read_ints()
        nums = ac.read_list_ints()
        lst = [ac.read_list_ints() for _ in range(m)]
        ans = BinarySearch().find_int_right(0, n, check)
        if ans == n:
            ac.st(0)
        else:
            ac.st(-1)
            ac.st(ans + 1)
        return

    @staticmethod
    def ac_120(ac=FastIO()):

        def check(pos):
            res = 0
            for s, e, d in nums:
                if s <= pos:
                    res += (ac.min(pos, e) - s) // d + 1
            return res % 2 == 1

        def compute(pos):
            res = 0
            for s, e, d in nums:
                if s <= pos <= e:
                    res += (pos - s) % d == 0
            return [pos, res]

        # 模板：利用单调性二分
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = [ac.read_list_ints() for _ in range(n)]
            low = min(x for x, _, _ in nums)
            high = max(x for _, x, _ in nums)
            while low < high - 1:
                mid = low + (high - low) // 2
                if check(mid):
                    high = mid
                else:
                    low = mid
            if check(low):
                ac.lst(compute(low))
            elif check(high):
                ac.lst(compute(high))
            else:
                ac.st("There's no weakness.")
        return

    @staticmethod
    def abc_56d(ac=FastIO()):
        # 模板：经典利用单调性进行二分，用背包DP进行check
        n, k = ac.read_ints()
        nums = ac.read_list_ints()
        nums.sort()

        def check(i):
            dp = [0]*k
            dp[0] = 1
            xx = nums[i]
            if xx >= k:
                return False

            for j in range(n):
                if j != i:
                    x = nums[j]
                    for p in range(k-1, x-1, -1):
                        if dp[p-x]:
                            dp[p] = 1
                            if p + xx >= k:
                                return False  # 此时为必要

            return True  # 为非必要的目标元素

        ans = BinarySearch().find_int_right(0, n-1, check)  # 非必要具有单调性，更小的也为非必要
        if check(ans):
            ac.st(ans+1)
        else:
            ac.st(0)
        return

    @staticmethod
    def abc_63d(ac=FastIO()):
        # 模板：经典利用单调性进行二分，贪心进行check
        n, a, b = ac.read_ints()
        nums = [ac.read_int() for _ in range(n)]

        def check(s):
            res = 0
            for num in nums:
                if num > s*b:
                    res += ac.ceil((num-s*b), (a-b))
            return res <= s

        ans = BinarySearch().find_int_left(0, ac.ceil(max(nums), b), check)
        ac.st(ans)
        return

    @staticmethod
    def ac_14(nums):
        # 模板：利用鸽巢原理进行二分
        n = len(nums) - 1
        low = 1
        high = n
        while low < high:
            mid = low + (high - low) // 2
            cnt = 0
            for num in nums:
                if low <= num <= mid:
                    cnt += 1
            if cnt > mid - low + 1:
                high = mid
            else:

                low = mid + 1
        return low

    @staticmethod
    def lg_p1281(ac=FastIO()):
        # 模板：典型二分并输出方案
        m, k = ac.read_ints()
        nums = ac.read_list_ints()

        def check(xx):
            res = pp = 0
            for ii in range(m - 1, -1, -1):
                if pp + nums[ii] > xx:
                    res += 1
                    pp = nums[ii]
                else:
                    pp += nums[ii]
                if res + 1 > k:
                    return False
            return True

        x = BinarySearch().find_int_left(max(nums), sum(nums), check)
        ans = []
        pre = nums[m - 1]
        post = m - 1
        for i in range(m - 2, -1, -1):
            if pre + nums[i] > x:
                ans.append([i + 2, post + 1])
                pre = nums[i]
                post = i
            else:
                pre += nums[i]
        ans.append([1, post + 1])
        for a in ans[::-1]:
            ac.lst(a)
        return

    @staticmethod
    def lg_p1381(ac=FastIO()):
        # 模板：典型二分
        n = ac.read_int()
        dct = set([ac.read_str() for _ in range(n)])
        m = ac.read_int()
        words = [ac.read_str() for _ in range(m)]
        cur = set()
        for w in words:
            if w in dct:
                cur.add(w)

        def check(x):
            cnt = defaultdict(int)
            cc = 0
            for i in range(m):
                # 滑动窗口判断是否可行
                if words[i] in dct:
                    cnt[words[i]] += 1
                    if cnt[words[i]] == 1:
                        cc += 1
                        if cc == s:
                            return True
                if i >= x - 1:
                    if words[i - x + 1] in dct:
                        cnt[words[i - x + 1]] -= 1
                        if not cnt[words[i - x + 1]]:
                            cc -= 1
            return False
        # 贪心选取所有能背的单词
        s = len(cur)
        ac.st(s)
        if not s:
            ac.st(0)
            return
        ans = BinarySearch().find_int_left(1, m, check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1419(ac=FastIO()):

        # 模板：二分加优先队列
        def check(x):
            stack = deque()
            res = []
            # 单调队列记录前序最小值
            for i in range(n):
                while stack and stack[0][0] <= i - k:
                    stack.popleft()
                while stack and stack[-1][1] >= pre[i] - x * i:
                    stack.pop()
                stack.append([i, pre[i] - x * i])
                res.append(stack[0][1])   # 记录长度在 k 左右的最小前缀变化和
                if i >= s - 1:
                    if pre[i + 1] - x * (i + 1) >= res[i - s + 1]:
                        return True
            return False

        n = ac.read_int()
        s, t = ac.read_ints()
        nums = []
        for _ in range(n):
            nums.append(int(input().strip()))
        pre = [0] * (n + 1)
        for j in range(n):
            pre[j + 1] = pre[j] + nums[j]

        # 二分最大平均值
        k = t - s
        ans = BinarySearch().find_float_right(min(nums), max(nums), check)
        ac.st("%.3f" % ans)
        return

    @staticmethod
    def lg_p1525(ac=FastIO()):
        # 模板：经典二分加BFS进行二分图划分
        n, m = ac.read_ints()
        lst = [ac.read_list_ints() for _ in range(m)]

        def check(weight):
            edges = [[i, j] for i, j, w in lst if w > weight]
            dct = defaultdict(list)
            for i, j in edges:
                dct[i].append(j)
                dct[j].append(i)
            # 使用染色法判断是否可以二分
            visit = [0] * (n + 1)
            for i in range(1, n + 1):
                if visit[i] == 0:
                    stack = [i]
                    visit[i] = 1
                    order = 2
                    while stack:
                        nex = []
                        for j in stack:
                            for y in dct[j]:
                                if not visit[y]:
                                    visit[y] = order
                                    nex.append(y)
                        order = 1 if order == 2 else 2
                        stack = nex

            return all(visit[i] != visit[j] for i, j in edges)

        # 二分最小的最大值
        low = 0
        high = max(ls[-1] for ls in lst)
        ans = BinarySearch().find_int_left(low, high, check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1542(ac=FastIO()):
        # 模板：二分加使用分数进行高精度计算
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]

        def add(lst1, lst2):
            # 进行分数加减
            a, b = lst1
            c, d = lst2
            d1 = a * d + c * b
            d2 = b * d
            return [d1, d2]

        def check(xx):
            # 最早与最晚出发
            t1 = 0
            res = [xx, 1]
            while int(res[0]) != res[0]:
                res[0] *= 10
                res[1] *= 10
            res = [int(w) for w in res]
            t1 = [0, 1]
            for x, y, s in nums:
                cur = add(t1, [s * res[1], res[0]])
                if cur[0] > y * cur[1]:
                    return False
                t1 = cur[:]
                if cur[0] < x * cur[1]:
                    t1 = [x, 1]
            return True

        ans = BinarySearch().find_float_left(1e-4, 10**7, check)
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def cf_1118d2(ac=FastIO()):

        # 模板：贪心二分
        n, m = ac.read_ints()
        nums = ac.read_list_ints()
        s = sum(nums)
        if s < m:
            ac.st(-1)
            return
        nums.sort(reverse=True)

        def check(x):
            ans = 0
            for i in range(n):
                j = i // x
                ans += ac.max(0, nums[i] - j)
            return ans >= m

        ac.st(BinarySearch().find_int_left(1, n, check))
        return

    @staticmethod
    def cf_883i(ac=FastIO()):
        # 模板：二分加双指针dp
        n, k = ac.read_ints()
        nums = sorted(ac.read_list_ints())

        def check(x):
            dp = [0] * (n + 1)
            dp[0] = 1
            j = 0
            for i in range(n):
                while nums[i] - nums[j] > x:
                    j += 1
                while not dp[j] and j < i - k + 1:
                    j += 1
                if dp[j] and i + 1 - j >= k:
                    dp[i + 1] = 1
            return dp[-1] == 1

        ans = BinarySearch().find_int_left(0, nums[-1] - nums[0], check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2237(ac=FastIO()):
        # 模板：脑筋急转弯排序后二分查找
        w, n = ac.read_ints()
        nums = [ac.read_str() for _ in range(w)]
        ind = list(range(w))
        ind.sort(key=lambda it: nums[it])
        nums.sort()
        for _ in range(n):
            k, s = ac.read_list_strs()
            k = int(k)
            x = bisect.bisect_left(nums, s) + k - 1
            if x < w and nums[x][:len(s)] == s:
                ac.st(ind[x] + 1)
            else:
                ac.st(-1)
        return

    @staticmethod
    def lg_p2810(ac=FastIO()):

        # 模板：二分加枚举
        n = ac.read_int()

        low = 0
        high = 10 ** 18

        def check2(x):
            k = 2
            res = 0
            while k * k * k <= x:
                res += x // (k * k * k)
                k += 1
            return res

        def check(x):
            k = 2
            res = 0
            while k * k * k <= x:
                res += x // (k * k * k)
                k += 1
            return res >= n

        ans = BinarySearch().find_int_left(low, high, check)
        if check2(ans) == n:
            ac.st(ans)
            return
        ac.st(-1)
        return

    @staticmethod
    def lg_p3718(ac=FastIO()):
        # 模板：二分加贪心
        n, k = ac.read_ints()
        s = ac.read_str()

        def check(x):
            if x == 1:
                # 特殊情况
                op1 = op2 = 0
                for i in range(n):
                    if i % 2:
                        op1 += 1 if s[i] == "N" else 0
                        op2 += 1 if s[i] == "F" else 0
                    else:
                        op1 += 1 if s[i] == "F" else 0
                        op2 += 1 if s[i] == "N" else 0
                return op1 <= k or op2 <= k

            op = 0
            pre = s[0]
            cnt = 1
            for w in s[1:]:
                if w == pre:
                    cnt += 1
                else:
                    # 对于相同状态连续区间断开的最少操作次数
                    op += cnt // (x + 1)
                    pre = w
                    cnt = 1
            op += cnt // (x + 1)
            return op <= k

        ans = BinarySearch().find_int_left(1, n, check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p3853(ac=FastIO()):
        # 模板：经典二分贪心题
        length, n, k = ac.read_ints()
        lst = ac.read_list_ints()
        lst.sort()

        def check(x):
            return sum((lst[i + 1] - lst[i] + x - 1) // x - 1 for i in range(n - 1)) <= k

        low = 1
        high = max(lst[i + 1] - lst[i] for i in range(n - 1))
        ans = BinarySearch().find_int_left(low, high, check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p4343(ac=FastIO()):
        # 模板：上下界二分加模拟
        l, k = ac.read_ints()
        lst = []
        for _ in range(l):
            lst.append(int(input().strip()))
        low = 1
        high = sum(abs(ls) for ls in lst)

        def compute(n):
            cnt = cur = 0
            for num in lst:
                cur += num
                if cur >= n:
                    cnt += 1
                    cur = 0
                cur = 0 if cur < 0 else cur
            return cnt

        def check1(n):
            return compute(n) >= k

        def check2(n):
            return compute(n) <= k

        # 经典的二分函数写法与出方案
        ceil = BinarySearch().find_int_right(low, high, check1)
        if compute(ceil) != k:
            ac.st(-1)
            return
        floor = BinarySearch().find_int_left(low, high, check2)
        ac.lst([floor, ceil])
        return

    @staticmethod
    def lg_p5844(ac=FastIO()):
        # 模板：经典中位数贪心与前缀和二分
        n, m, b = ac.read_ints()
        pos = [ac.read_int() for _ in range(n)]
        ans = j = 0
        pre = ac.accumulate(pos)

        def check(x, y):
            mid = (x + y) // 2
            left = (mid - x) * pos[mid] - (pre[mid] - pre[x])
            right = pre[y + 1] - pre[mid + 1] - (y - mid) * pos[mid]
            return left + right

        for i in range(n):
            # 枚举左端点二分右端点
            while j < n and check(i, j) <= b:
                j += 1
            ans = ac.max(ans, j - i)
        ac.st(ans)
        return

    @staticmethod
    def lg_p5878(ac=FastIO()):
        # 模板：使用二分加枚举计算
        n, m = ac.read_ints()
        nums = [ac.read_list_ints() for _ in range(n)]

        def check(num):
            cost = 0
            for x, y, sm, pm, sv, pv in nums:
                need = num * x - y
                if need <= 0:
                    continue
                cur = inf
                # 枚举小包装个数
                for i in range(need + 1):
                    rest = need - i * sm
                    if rest > 0:
                        cur = ac.min(cur, i * pm + math.ceil(rest / sv) * pv)
                    else:
                        cur = ac.min(cur, i * pm)
                        break
                cost += cur
                if cost > m:
                    return False

            return cost <= m

        ans = BinarySearch().find_int_right(0, m, check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6004(ac=FastIO()):
        # 模板：二分加并查集计算
        n, m = ac.read_ints()
        nums = ac.read_list_ints_minus_one()
        edges = [ac.read_list_ints() for _ in range(m)]
        edges.sort(key=lambda it: -it[2])

        def check(x):
            uf = UnionFind(n)
            for i, j, _ in edges[:x]:
                uf.union(i - 1, j - 1)
            group = uf.get_root_part()
            for g in group:
                cur = set([nums[i] for i in group[g]])
                if not all(i in cur for i in group[g]):
                    return False
            return True

        ans = BinarySearch().find_int_left(0, m, check)
        ac.st(-1 if not ans else edges[ans - 1][2])
        return

    @staticmethod
    def lg_p6058(ac=FastIO()):
        # 模板：使用深搜序与离线 LCA 计算相邻叶子之间距离并二分确定时间
        n, k = ac.read_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y, z = ac.read_ints_minus_one()
            dct[x].append([y, z + 1])
            dct[y].append([x, z + 1])
        if n == 1:
            ac.st(0)
            return
        for i in range(n):
            dct[i].sort(reverse=True)
        # 找出叶子
        stack = [[0, -1]]
        dis = [0] * n
        leaf = []
        while stack:
            i, fa = stack.pop()
            for j, w in dct[i]:
                if j != fa:
                    dis[j] = dis[i] + w
                    stack.append([j, i])
            if len(dct[i]) == 1 and i:
                leaf.append(i)
        c = len(leaf)
        pairs = [[leaf[i - 1], leaf[i]] for i in range(1, c)]
        edge = [[ls[0] for ls in lst] for lst in dct]
        # 计算叶子之间的距离
        ces = OfflineLCA().bfs_iteration(edge, pairs, 0)
        pairs_dis = [dis[leaf[i - 1]] + dis[leaf[i]] - 2 * dis[ces[i - 1]] for i in range(1, c)]
        pre = ac.accumulate(pairs_dis)

        def check(t):
            ii = 0
            part = 0
            while ii < c:
                post = -1
                for jj in range(ii, c):
                    # 当前节点最远能够到达的叶子距离
                    if pre[jj] - pre[ii] + dis[leaf[ii]] + dis[leaf[jj]] <= t:
                        post = jj
                    else:
                        break
                part += 1
                ii = post + 1
            return part <= k

        # 二分进行计算
        ans = BinarySearch().find_int_left(max(dis[i] * 2 for i in leaf), sum(dis[i] * 2 for i in leaf), check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6069(ac=FastIO()):
        # 模板：经典方差计算公式变形，使用二分加变量维护区间的方差值大小
        n, m = ac.read_ints()
        nums = ac.read_list_ints()
        nums.sort()

        def check(x):
            ss = s = 0
            for i in range(n):
                ss += nums[i]**2
                s += nums[i]
                if i >= x - 1:
                    # 方差变形公式转换为整数乘法
                    if x * ss - s * s <= x * m:
                        return True
                    ss -= nums[i - x + 1]**2
                    s -= nums[i - x + 1]
            return False

        ans = BinarySearch().find_int_right(1, n, check)
        ac.st(n - ans)
        return

    @staticmethod
    def lg_p6633(ac=FastIO()):
        # 模板：二分加STL进行 Check
        n, k = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda it: -it[1])

        def check(x):
            res = 0
            pre = LocalSortedList()
            for a, c in nums:
                res += pre.bisect_right(a*c-x*a)
                pre.add(-(a*c-x*a))
                if res >= k:
                    return True
            return res >= k

        ans = BinarySearch().find_float_right(0, nums[0][1], check, 1e-3)
        ac.st(ans)
        return

    @staticmethod
    def lc_2594(ranks: List[int], cars: int) -> int:
        #  模板：经典二分

        def check(x):
            res = 0
            for r in ranks:
                res += int((x / r) ** 0.5)
            return res >= cars

        return BinarySearch().find_int_left(0, ranks[0] * cars ** 2, check)

    @staticmethod
    def lc_2604(hens: List[int], grains: List[int]) -> int:
        # 模板：二分加指针贪心 check
        hens.sort()
        grains.sort()
        m, n = len(hens), len(grains)

        def check(x):
            i = 0
            for pos in hens:
                left = right = 0
                while i < n:
                    if grains[i] >= pos:
                        right = right if right > grains[i] - pos else grains[i] - pos
                    else:
                        left = left if left > pos - grains[i] else pos - grains[i]
                    if left * 2 + right <= x or right * 2 + left <= x:
                        i += 1
                    else:
                        break
                if i == n:
                    return True
            return False

        low = 0
        high = sum(abs(g - hens[0]) * 2 for g in grains)
        return BinarySearch().find_int_left(low, high, check)

    @staticmethod
    def lg_p8161(ac=FastIO()):
        # 模板：经典贪心加二分
        n, m = ac.read_ints()
        a = ac.read_list_ints()
        b = ac.read_list_ints()

        def check(x):
            res = 0
            for i in range(n):
                if a[i] < b[i]:
                    res += (x + b[i] - 1) // b[i]
                else:
                    if m * a[i] >= x:
                        res += (x + a[i] - 1) // a[i]
                    else:
                        res += m
                        res += (x - a[i] * m + b[i] - 1) // b[i]
                if res > m*n:
                    return False
            return res <= m * n

        low = 0
        high = 10**18
        ans = BinarySearch().find_int_right(low, high, check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p8198(ac=FastIO()):
        # 模板：经典二分加指针
        n, k = ac.read_ints()
        nums = ac.read_list_ints()

        def check(x):
            res = pre = 0
            for num in nums:
                if pre + num * num > x:
                    res += 1
                    pre = num * num
                else:
                    pre += num * num
            res += 1
            return res <= k

        low = max(nums)**2
        high = sum(num * num for num in nums)
        ans = BinarySearch().find_int_left(low, high, check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p9050(ac=FastIO()):
        # 模板：经典二分注意值域区间与贪心模拟
        n = ac.read_int()
        if n == 1:
            ac.st("T")
            return

        nums = ac.read_list_ints()
        lst = sorted(nums)

        def check(x):
            flag = 1
            res = lst[x]
            for w in lst:
                if w == lst[x] and flag:
                    flag = 0
                    continue
                if res <= w:
                    return False
                res += w
            return True

        floor = BinarySearch().find_int_left(0, n-1, check)
        if check(floor):
            ans = ["N" if nums[x] < lst[floor] else "T" for x in range(n)]
            ac.st("".join(ans))
        else:
            ac.st("N" * n)
        return

    @staticmethod
    def ac_3973(ac=FastIO()):
        # 模板：浮点数二分与滑动窗口双指针
        n, m = ac.read_ints()
        nums = ac.read_list_ints()
        pos = ac.read_list_ints()
        pos.sort()
        nums.sort()

        def check(x):
            i = 0
            for num in nums:
                while i < m and not (pos[i]-x <= num <= pos[i]+x):
                    i += 1
                if i == m:
                    return False
            return True

        ans = BinarySearch().find_float_left(0, 2*10**9, check)
        if ans - int(ans) >= 0.5:
            ac.st(int(ans)+1)
        else:
            ac.st(int(ans))
        return

    @staticmethod
    def ac_4683(ac=FastIO()):
        # 模板：经典二分加鸽巢原理
        for _ in range(ac.read_int()):
            ac.read_str()
            m, n = ac.read_ints()
            grid = [ac.read_list_ints() for _ in range(m)]

            def check(x):
                row = [0] * m
                col = [0] * n
                for i in range(m):
                    for j in range(n):
                        if grid[i][j] >= x:
                            row[i] += 1
                            col[j] = 1
                if any(x == 0 for x in col):
                    return False
                if m <= n - 1:
                    return True
                return max(row) >= 2

            ac.st(BinarySearch().find_int_right(0, 10**9, check))
        return

    @staticmethod
    def ac_5048(ac=FastIO()):
        # 模板：经典浮点数二分并求出具体方案
        n = ac.read_int()
        nums = ac.read_list_ints()
        nums.sort()

        def compute(r):
            pre = -inf
            res = []
            for num in nums:
                if num > pre:
                    res.append(num+r)
                    pre = num+2*r
                    if len(res) > 3:
                        break
            return res

        def check(r):
            return len(compute(r)) <= 3

        x = BinarySearch().find_float_left(0, nums[-1]-nums[0], check, 1e-6)
        ac.st(x)
        ans = compute(x)
        while len(ans) < 3:
            ans.append(ans[-1]+1)
        ac.lst(ans)
        return


class TestGeneral(unittest.TestCase):

    def test_binary_search(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
