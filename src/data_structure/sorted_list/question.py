
"""

算法：有序集合
功能：利用单调性确定最优选择，通常可以使用SortedList用于维护和查询有序集合信息

题目：xx（xx）

===================================力扣===================================
295. 数据流的中位数（https://leetcode.cn/problems/find-median-from-data-stream/）使用一个SortedList即可
2426 满足不等式的数对数目（https://leetcode.cn/problems/number-of-pairs-satisfying-inequality/）根据不等式变换和有序集合进行二分查找
2179 统计数组中好三元组数目（https://leetcode.cn/problems/count-good-triplets-in-an-array/）维护区间范围内的个数
2141 同时运行 N 台电脑的最长时间（https://leetcode.cn/problems/maximum-running-time-of-n-computers/）贪心选择最大的 N 个电池作为基底，然后二分确定在其余电池的加持下可以运行的最长时间
2102 序列顺序查询（https://leetcode.cn/problems/sequentially-ordinal-rank-tracker/）使用有序集合维护优先级姓名实时查询
2519. Count the Number of K-Big Indices（https://leetcode.cn/problems/count-the-number-of-k-big-indices/）使用有序集合维护计算数量
2276. 统计区间中的整数数目（https://leetcode.cn/problems/count-integers-in-intervals/）动态开点线段树模板题，维护区间并集的长度，也可使用SortedList
1912. 设计电影租借系统（https://leetcode.cn/problems/design-movie-rental-system/）典型SortedList应用
1825. 求出 MK 平均值（https://leetcode.cn/problems/finding-mk-average/）经典SortedList与deque应用
2250. 统计包含每个点的矩形数目（https://leetcode.cn/problems/count-number-of-rectangles-containing-each-point/）离线查询，指针排序二分

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
P7333 [JRKSJ R1] JFCA（https://www.luogu.com.cn/problem/P7333）经典排序预处理后，动态更新使用有序集合进行查询，注意是环形数组
P7391 「TOCO Round 1」自适应 PVZ（https://www.luogu.com.cn/problem/P7391）有序集合进行贪心模拟，延迟替换，类似课程表3
P7910 [CSP-J 2021] 插入排序（https://www.luogu.com.cn/problem/P7910）使用有序列表进行维护
P4375 [USACO18OPEN]Out of Sorts G（https://www.luogu.com.cn/problem/P4375）冒泡排序，使用有序列表维护
P1908 逆序对（https://www.luogu.com.cn/problem/P1908）经典问题求逆序对，可以使用归并排序
P1966 [NOIP2013 提高组] 火柴排队（https://www.luogu.com.cn/problem/P1966）逆序对经典贪心题目
P2161 [SHOI2009]会场预约（https://www.luogu.com.cn/problem/P2161）区间合并与删除处理
P1637 三元上升子序列（https://www.luogu.com.cn/problem/P1637）典型STL应用题，前后缀大小值计数
P2234 [HNOI2002]营业额统计（https://www.luogu.com.cn/problem/P2234）典型STL应用题
P2804 神秘数字（https://www.luogu.com.cn/problem/P2804）前缀和加 STL 计算平均值大于 m 的连续子数组个数
P3608 [USACO17JAN]Balanced Photo G（https://www.luogu.com.cn/problem/P3608）典型STL应用题
P5076 【深基16.例7】普通二叉树（简化版）（https://www.luogu.com.cn/problem/P5076）使用有序列表与有序集合进行名次模拟
P5149 会议座位（https://www.luogu.com.cn/problem/P5149）经典逆序对计算使用 bisect 实现
P5459 [BJOI2016]回转寿司（https://www.luogu.com.cn/problem/P5459）前缀和与有序列表二分查找
P6538 [COCI2013-2014#1] LOPOV（https://www.luogu.com.cn/problem/P6538）典型STL维护贪心
P7912 [CSP-J 2021] 小熊的果篮（https://www.luogu.com.cn/problem/P7912）经典 STL 应用模拟题
P8667 [蓝桥杯 2018 省 B] 递增三元组（https://www.luogu.com.cn/problem/P8667）典型STL应用题

================================CodeForces================================
D. Pashmak and Parmida's problem（https://codeforces.com/problemset/problem/459/D）使用有序集合进行大小计数查找
E. Enemy is weak（https://codeforces.com/problemset/problem/61/E）典型应用场景，前后缀大于小于值计数
D. Multiset（https://codeforces.com/problemset/problem/1354/D）有序列表的维护与查询
E2. Median on Segments（https://codeforces.com/contest/1005/problem/E2）经典特定中位数的连续子数组个数，使用容斥原理加前缀和有序列表二分
E. MEX and Increments（https://codeforces.com/contest/1619/problem/E）经典MEX贪心

参考：OI WiKi（xx）
"""


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lg_4375d(ac=FastIO()):
        # 模板：双向冒泡排序所需要的比较轮数
        n = ac.read_int()
        ans = 1
        nums = [ac.read_int() for _ in range(n)]
        tmp = sorted(nums)
        ind = {num: i+1 for i, num in enumerate(tmp)}
        lst = LocalSortedList()
        for i in range(n):
            lst.add(ind[nums[i]])
            ans = ac.max(ans, i+1-lst.bisect_right(i+1))
        ac.st(ans)
        return

    @staticmethod
    def cf_61e(ac=FastIO()):
        # 模板：典型计算 i < j < k 但是 nums[i] > nums[j] > nums[k] 的组合数
        n = ac.read_int()
        nums = ac.read_list_ints()
        pre = [0] * (n + 1)
        lst = LocalSortedList()
        for i in range(n):
            pre[i + 1] = i - lst.bisect_right(nums[i])
            lst.add(nums[i])

        post = [0] * (n + 1)
        lst = LocalSortedList()
        for i in range(n - 1, -1, -1):
            post[i] = lst.bisect_left(nums[i])
            lst.add(nums[i])

        ans = 0
        for i in range(1, n - 1):
            ans += pre[i + 1] * post[i]
        ac.st(ans)
        return

    @staticmethod
    def cf_1005e2(ac=FastIO()):
        # 模板：经典特定中位数的连续子数组个数，使用容斥原理加前缀和有序列表二分
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()

        def check(x):
            lst = [1 if num >= x else -1 for num in nums]
            pre = LocalSortedList([0])
            cur = res = 0
            for num in lst:
                cur += num
                res += pre.bisect_left(cur)
                pre.add(cur)
            return res
        ac.st(- check(m+1) + check(m))
        return

    @staticmethod
    def lc_2426(nums1: List[int], nums2: List[int], diff: int) -> int:
        # 模板：经典使用公式变换与有序集合二分查找进行计数
        n = len(nums1)
        ans = 0
        lst = SortedList([nums1[n - 1] - nums2[n - 1] + diff])
        for i in range(n - 2, -1, -1):
            k = lst.bisect_left(nums1[i] - nums2[i])
            ans += len(lst) - k
            lst.add(nums1[i] - nums2[i] + diff)
        return ans

    @staticmethod
    def lg_1966(ac=FastIO()):
        # 模板：逆序对经典贪心题目
        n = ac.read_int()
        ans = 0
        mod = 10**8-3
        nums1 = ac.read_list_ints()
        ind1 = list(range(n))
        ind1.sort(key=lambda it: nums1[it])

        nums2 = ac.read_list_ints()
        ind2 = list(range(n))
        ind2.sort(key=lambda it: nums2[it])

        q = [0]*n
        for i in range(n):
            q[ind1[i]] = ind2[i]
        lst = LocalSortedList()
        for num in q:
            ans += len(lst) - lst.bisect_right(num)
            lst.add(num)
        ac.st(ans % mod)
        return

    @staticmethod
    def ac_127(ac=FastIO()):
        # 模板：经典二维排序贪心
        n, m = ac.read_list_ints()
        machine = [ac.read_list_ints() for _ in range(n)]
        task = [ac.read_list_ints() for _ in range(m)]
        machine.sort(reverse=True)
        task.sort(reverse=True)
        lst = []
        ans = money = j = 0
        for i in range(m):
            tm, level = task[i]
            while j < n and machine[j][0] >= tm:
                insort_left(lst, machine[j][1])  # 使用bisect代替Sortedlist
                j += 1
            ind = bisect_left(lst, level)
            if ind < len(lst):
                lst.pop(ind)
                ans += 1
                money += 500*tm + 2*level
        ac.lst([ans, money])
        return

    @staticmethod
    def lg_p1637(ac=FastIO()):
        # 模板：典型STL应用题，前后缀大小值计数
        n = ac.read_int()
        nums = ac.read_list_ints()

        pre = [0] * n
        lst = LocalSortedList()
        for i in range(n):
            pre[i] = lst.bisect_left(nums[i])
            lst.add(nums[i])

        lst = LocalSortedList()
        post = [0] * n
        for i in range(n - 1, -1, -1):
            post[i] = n - i - 1 - lst.bisect_right(nums[i])
            lst.add(nums[i])
        ans = sum(pre[i] * post[i] for i in range(n))
        ac.st(ans)
        return

    @staticmethod
    def lg_p2234(ac=FastIO()):
        # 模板：典型STL应用题
        n = ac.read_int()
        ans = 0
        lst = LocalSortedList()
        for _ in range(n):
            x = ac.read_int()
            if not lst:
                ans += x
            else:
                i = lst.bisect_left(x)
                cur = inf
                for j in [i-1, i]:
                    if 0 <= j < len(lst):
                        cur = ac.min(cur, abs(lst[j]-x))
                ans += cur
            lst.add(x)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2804(ac=FastIO()):
        # 模板：前缀和加 STL 计算平均值大于 m 的连续子数组个数
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        mod = 92084931
        pre = 0
        lst = LocalSortedList()
        lst.add(0)
        ans = 0
        for num in nums:
            pre += num - m
            ans += lst.bisect_left(pre)
            lst.add(pre)
        ac.st(ans % mod)
        return

    @staticmethod
    def lg_p5076(ac=FastIO()):
        # 模板：使用有序列表与有序集合进行名次模拟
        q = ac.read_int()
        dct = set()
        lst = []
        lst_set = []
        for _ in range(q):
            op, x = ac.read_list_ints()
            if op == 1:
                i = bisect.bisect_left(lst, x)
                ac.st(i + 1)
            elif op == 2:
                ac.st(lst_set[x - 1])
            elif op == 3:
                i = bisect.bisect_left(lst, x)
                n = len(lst)
                ans = -2147483647
                if 0 <= i - 1 < n and lst_set[i - 1] < x:
                    ans = lst_set[i - 1]
                ac.st(ans)
            elif op == 4:
                i = bisect.bisect_right(lst, x)
                n = len(lst)
                if 0 <= i < n:
                    ac.st(lst[i])
                else:
                    ac.st(2147483647)
            else:
                bisect.insort_left(lst, x)
                if x not in dct:
                    bisect.insort_left(lst_set, x)
                    dct.add(x)
        return

    @staticmethod
    def lg_p5149(ac=FastIO()):
        # 模板：经典逆序对计算使用 bisect 实现
        ac.read_int()
        lst = ac.read_list_strs()
        ind = {st: i for i, st in enumerate(lst)}
        lst = [ind[s] for s in ac.read_list_strs()]
        ans = 0
        pre = []
        for num in lst:
            ans += len(pre)-bisect.bisect_left(pre, num)
            bisect.insort_left(pre, num)
        ac.st(ans)
        return

    @staticmethod
    def lg_p5459(ac=FastIO()):
        # 模板：前缀和与有序列表二分查找
        n, low, high = ac.read_list_ints()
        a = ac.read_list_ints()
        ans = 0
        lst = []
        s = sum(a)
        bisect.insort_left(lst, s)
        for i in range(n - 1, -1, -1):
            s -= a[i]
            x = bisect.bisect_left(lst, s + low)
            ans += bisect.bisect_right(lst, s + high) - x
            bisect.insort_left(lst, s)
        ac.st(ans)
        return

    @staticmethod
    def lg_p7912(ac=FastIO()):
        # 模板：经典 STL 应用模拟题使用 STL 模拟删除
        n = ac.read_int()
        nums = ac.read_list_ints()
        lst = [LocalSortedList([i+1 for i in range(n) if not nums[i]]), LocalSortedList([i+1 for i in range(n) if nums[i]])]

        while lst[0] and lst[1]:
            ans = []
            if lst[0][0] < lst[1][0]:
                i = 0
            else:
                i = 1
            ans.append(lst[i].pop(0))
            i = 1-i
            while lst[i] and lst[i][-1] > ans[-1]:
                j = lst[i].bisect_left(ans[-1])
                ans.append(lst[i].pop(j))
                i = 1-i
            ac.lst(ans)

        for a in lst[0]:
            ac.st(a)
        for a in lst[1]:
            ac.st(a)
        return


class SolutionLC2276:

    def __init__(self):
        self.lst = SortedList()
        self.sum = 0

    def add(self, left: int, right: int) -> None:
        x = self.lst.bisect_left([left, left])
        while x - 1 >= 0 and self.lst[x - 1][1] >= left:
            x -= 1

        while 0 <= x < len(self.lst) and not (self.lst[x][0] > right or self.lst[x][1] < left):
            a, b = self.lst.pop(x)
            left = left if left < a else a
            right = right if right > b else b
            self.sum -= b - a + 1
        self.sum += right - left + 1
        self.lst.add([left, right])

    def count(self) -> int:
        return self.sum
