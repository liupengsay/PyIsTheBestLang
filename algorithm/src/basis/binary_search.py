import bisect
import unittest
import random


"""

算法：二分查找
功能：利用单调性确定最优选择，通常可以使用SortedList、Bisect，还可以使用精度控制二分
题目：xx（xx）
===================================力扣===================================
4. 寻找两个正序数组的中位数（https://leetcode.cn/problems/median-of-two-sorted-arrays/）经典二分思想查找题
295. 数据流的中位数（https://leetcode.cn/problems/find-median-from-data-stream/）使用一个SortedList和三个变量维护左右两边与中间段的和
2468 根据限制分割消息（https://leetcode.cn/problems/split-message-based-on-limit/）根据长度限制进行二分
2426 满足不等式的数对数目（https://leetcode.cn/problems/number-of-pairs-satisfying-inequality/）根据不等式变换和有序集合进行二分查找
2179 统计数组中好三元组数目（https://leetcode.cn/problems/count-good-triplets-in-an-array/）维护区间范围内的个数
2141 同时运行 N 台电脑的最长时间（https://leetcode.cn/problems/maximum-running-time-of-n-computers/）贪心选择最大的 N 个电池作为基底，然后二分确定在其余电池的加持下可以运行的最长时间
2102 序列顺序查询（https://leetcode.cn/problems/sequentially-ordinal-rank-tracker/）使用有序集合维护优先级姓名实时查询

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
https://codeforces.com/problemset/problem/913/C（DP预处理最有单价，再二分加贪心进行模拟求解）
Teleporters (Hard Version)（https://codeforces.com/problemset/problem/1791/G2）贪心排序，前缀和枚举二分

D. Multiplication Table（https://codeforces.com/problemset/problem/448/D）经典二分查找计算n*m的乘法表第k大元素
D. Cleaning the Phone（https://codeforces.com/problemset/problem/1475/D）贪心排序+前缀和+枚举+二分
D. Odd-Even Subsequence（https://codeforces.com/problemset/problem/1370/D）利用单调性二分，再使用贪心check

参考：OI WiKi（xx）
"""


class BinarySearch:
    def __init__(self):
        return

    @staticmethod
    def bisect_left(nums, target):
        # 模板: 寻找target的插入位置1
        return bisect.bisect_left(nums, target)

    @staticmethod
    def bisect_right(nums, target):
        # 模板: 寻找target的插入位置2
        return bisect.bisect_right(nums, target)

    @staticmethod
    def find_int(low, high, check):
        # 模板: 整数范围内二分查找
        while low < high-1:
            mid = low+(high-low)//2
            if check(mid):
                high = mid
            else:
                low = mid
        return low if check(low) else high

    @staticmethod
    def find_float(low, high, check, error=1e-6):
        # 模板: 浮点数范围内二分查找
        while low < high-error:
            mid = low+(high-low)/2
            if check(mid):
                high = mid
            else:
                low = mid
        return low if check(low) else high

    @staticmethod
    def cf_448d(n, m, k):
        # 模板：计算 n*m 乘法矩阵内的第 k 大元素
        def check(num):
            res = 0
            for x in range(1, m + 1):
                res += min(n, num // x)
            return res

        # 初始化大小
        if m > n:
            m, n = n, m

        # 二分查找第k大
        low = 1
        high = m * n
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid) < k:
                low = mid
            else:
                high = mid

        ans = low if check(low) >= k else high
        return ans


class TestGeneral(unittest.TestCase):

    def test_binary_search(self):

        bs = BinarySearch()
        for _ in range(2):
            m = random.randint(10, 100)
            n = random.randint(10, 100)
            lst = []
            for i in range(1, m + 1):
                lst.extend([i*j for j in range(1, n+1)])
            lst.sort()
            for i, num in enumerate(lst):
                k = i+1
                assert bs.cf_448d(n, m, k) == num

        return


if __name__ == '__main__':
    unittest.main()
