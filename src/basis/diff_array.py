import math
import unittest
from collections import defaultdict, deque
from typing import List

from src.basis.binary_search import BinarySearch
from src.fast_io import FastIO
import bisect
from math import inf

"""

算法：差分数组与前缀和、后缀和、前缀最大子序列和、后缀最大子序列和、二维差分、离散化差分、三维差分
功能：用来解决一维数组或者二维数组的加和问题，以及前缀和计算，还有前缀和的前缀和
题目：

===================================力扣===================================
152. 乘积最大子数组（https://leetcode.cn/problems/maximum-product-subarray/）前缀最大最小乘积，计算最大连续子数组乘积
598. 范围求和 II（https://leetcode.cn/problems/range-addition-ii/）二维差分
2281. 巫师的总力量（https://leetcode.cn/problems/sum-of-total-strength-of-wizards/）枚举当前元素作为最小值的子数组和并使用前缀和的前缀和计算
2251. 花期内花的数目（https://leetcode.cn/problems/number-of-flowers-in-full-bloom/）离散化差分数组
2132. 用邮票贴满网格图（https://leetcode.cn/problems/stamping-the-grid/）用前缀和枚举可行的邮票左上端点，然后查看空白格点左上方是否有可行的邮票点，也可以使用经典的二维差分覆盖进行解决
1229. 安排会议日程（https://leetcode.cn/problems/meeting-scheduler/）离散化差分数组
6292. 子矩阵元素加 1（https://leetcode.cn/problems/increment-submatrices-by-one/)二维差分前缀和
2565. 最少得分子序列（https://leetcode.cn/problems/subsequence-with-the-minimum-score/）使用前后缀指针枚举匹配最长前后缀
644. 子数组最大平均数 II（https://leetcode.cn/problems/maximum-average-subarray-ii/）前缀和加二分计算不短于k的子数组最大平均值
1292. 元素和小于等于阈值的正方形的最大边长（https://leetcode.cn/problems/maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold/）经典O(mn)复杂度枚举
1674. 使数组互补的最少操作次数（https://leetcode.cn/problems/minimum-moves-to-make-array-complementary/）经典差分数组作用域计数
1714. 数组中特殊等间距元素的和（https://leetcode.cn/problems/sum-of-special-evenly-spaced-elements-in-array/）经典分组加前缀和
1738. 找出第 K 大的异或坐标值（https://leetcode.cn/problems/find-kth-largest-xor-coordinate-value/）经典二维前缀异或和
1895. 最大的幻方（https://leetcode.cn/problems/largest-magic-square/）二维前缀和枚举优化
1943. 描述绘画结果（https://leetcode.cn/problems/describe-the-painting/）经典离散化差分
2021. 街上最亮的位置（https://leetcode.cn/problems/brightest-position-on-street/）经典离散化差分

===================================洛谷===================================
P8772 [蓝桥杯 2022 省 A] 求和（https://www.luogu.com.cn/record/list?user=739032&status=12&page=15）后缀和计算
P2367 语文成绩（https://www.luogu.com.cn/problem/P2367）差分数组记录区间增减变化，最后还原计算最小值
P2280 [HNOI2003]激光炸弹（https://www.luogu.com.cn/problem/P2280）二维前缀和
P3138 [USACO16FEB]Load Balancing S（https://www.luogu.com.cn/problem/P3138）二维前缀和
P3406 海底高铁（https://www.luogu.com.cn/problem/P3406）差分数组计算每段路经过次数，再贪心新选择买卡还是单刷
P3655 不成熟的梦想家 (未熟 DREAMER)（https://www.luogu.com.cn/problem/P3655）差分数组变形模拟
P5542 [USACO19FEB]Painting The Barn S（https://www.luogu.com.cn/problem/P5542）二维差分，需要适配覆盖区间
P5686 [CSP-S2019 江西] 和积和（https://www.luogu.com.cn/problem/P5686）前缀和的前缀和
P6180 [USACO15DEC]Breed Counting S（https://www.luogu.com.cn/problem/P6180）前缀和计数
P6481 [COCI2006-2007] FIREFLY（https://www.luogu.com.cn/problem/P6481）区间更新后进行所有前缀查询
P2956 [USACO09OCT]The Robot Plow G（https://www.luogu.com.cn/problem/P2956）二维差分前缀和
P3397 地毯（https://www.luogu.com.cn/problem/P3397）二维差分前缀和
P1869 愚蠢的组合数（https://www.luogu.com.cn/problem/P1869）使用前缀和记录1-N的因子2的个数继而计算C(N,K)的奇偶性
P7667 [JOI2018] Art Exhibition（https://www.luogu.com.cn/problem/P7667）公式变换，排序后使用前缀和
P2671 [NOIP2015 普及组] 求和（https://www.luogu.com.cn/problem/P2671）前缀加和与前缀计数枚举，分奇偶性讨论
P1719 最大加权矩形（https://www.luogu.com.cn/problem/P1719）求最大子矩阵和，经典枚举矩阵上下边界并使用前缀和计算
P2882 [USACO07MAR]Face The Right Way G（https://www.luogu.com.cn/problem/P2882）贪心枚举加差分验证
P4552 [Poetize6] IncDec Sequence（https://www.luogu.com.cn/problem/P4552）差分数组经典题，明晰差分本质
P1627 [CQOI2009] 中位数（https://www.luogu.com.cn/problem/P1627）经典前后缀中位数大小值差值计数
P1895 数字序列（https://www.luogu.com.cn/problem/P1895）前缀和计数加二分查找，最多不超多10**5
P1982 [NOIP2013 普及组] 小朋友的数字（https://www.luogu.com.cn/problem/P1982）前缀最大连续子段和与前缀最大值
P2070 刷墙（https://www.luogu.com.cn/problem/P2070）哈希离散化差分数组计数
P2190 小Z的车厢（https://www.luogu.com.cn/problem/P2190）环形数组差分
P2352 队爷的新书（https://www.luogu.com.cn/problem/P2352）离散化差分
P2363 马农（https://www.luogu.com.cn/problem/P2363）二维前缀和与枚举
P2706 巧克力（https://www.luogu.com.cn/problem/P2706）不包含障碍点的最大子矩阵和
P2879 [USACO07JAN] Tallest Cow S（https://www.luogu.com.cn/problem/P2879）差分数组经典题与贪心
P3028 [USACO10OCT]Soda Machine G（https://www.luogu.com.cn/problem/P3028）离散化差分计算覆盖区间最多的点
P4030 [Code+#2]可做题1（https://www.luogu.com.cn/problem/P4030）脑筋急转弯加二维前缀和计算
P4440 [COCI2017-2018#3] Programiranje（https://www.luogu.com.cn/problem/P4440）典型前缀和计数
P4623 [COCI2012-2013#6] BUREK（https://www.luogu.com.cn/problem/P4623）离散化差分计数
P6032 选择客栈 加强版（https://www.luogu.com.cn/problem/P6032）经典前后缀计数
P6070 『MdOI R1』Decrease（https://www.luogu.com.cn/problem/P6070）经典二维差分贪心修改实时维护差分与计算前缀和即矩阵最新值
P6278 [USACO20OPEN]Haircut G（https://www.luogu.com.cn/problem/P6278）经典逆序对作用域与差分前缀和计算
P6537 [COCI2013-2014#1] RATAR（https://www.luogu.com.cn/problem/P6537）预处理前缀和加枚举
P6877 [JOI 2020 Final] 長いだけのネクタイ（https://www.luogu.com.cn/problem/P6877）排序贪心前后缀 DP 枚举
P6878 [JOI 2020 Final] JJOOII 2（https://www.luogu.com.cn/problem/P6878）前后缀枚举
P8081 [COCI2011-2012#4] ZIMA（https://www.luogu.com.cn/problem/P8081）差分计数计算作用域
P8033 [COCI2015-2016#7] Prozor（https://www.luogu.com.cn/problem/P8033）二维前缀和计数
P7992 [USACO21DEC] Convoluted Intervals S（https://www.luogu.com.cn/problem/P7992）经典桶计数与作用域差分计数
P7948 [✗✓OI R1] 前方之风（https://www.luogu.com.cn/problem/P7948）排序后预处理前后缀信息指针查询
P8343 [COCI2021-2022#6] Zemljište（https://www.luogu.com.cn/problem/P8343）经典子矩阵前缀和枚举与双指针
P8551 Bassline（https://www.luogu.com.cn/problem/P8551）差分数组经典灵活应用
P8666 [蓝桥杯 2018 省 A] 三体攻击（https://www.luogu.com.cn/problem/P8666）二分加三维差分经典题
P8715 [蓝桥杯 2020 省 AB2] 子串分值（https://www.luogu.com.cn/problem/P8715）前后缀贡献计数
P8783 [蓝桥杯 2022 省 B] 统计子矩阵（https://www.luogu.com.cn/problem/P8783）经典O(n^3)与双指针枚举计算子矩阵个数

================================CodeForces================================
https://codeforces.com/problemset/problem/33/C（前后缀最大变换和与分割点枚举，经典类型题目）
https://codeforces.com/problemset/problem/797/C（后缀最小值字典序模拟）
https://codeforces.com/problemset/problem/75/D（压缩数组最大子段和）
C. Count Triangles（https://codeforces.com/problemset/problem/1355/C）经典使用作用域差分计算，合法三角形边长个数
C. Tea Tasting（https://codeforces.com/problemset/problem/1795/C）前缀和二分后，经典使用差分计数模拟加和
D. Constant Palindrome Sum（https://codeforces.com/problemset/problem/1343/D）枚举x使用差分数组进行范围计数
E. Counting Rectangles（https://codeforces.com/problemset/problem/1722/E）根据数字取值范围使用二位前缀和计算
D. Absolute Sorting（https://codeforces.com/contest/1772/problem/D）离散差分作用域计数

================================AcWing===================================
99. 激光炸弹（https://www.acwing.com/problem/content/description/101/）二维前缀和
100. 增减序列（https://www.acwing.com/problem/content/102/）差分数组经典题目，明晰本质
101. 最高的牛（https://www.acwing.com/problem/content/103/）查分数组，贪心得出结论
102. 最佳牛围栏（https://www.acwing.com/problem/content/104/）前缀和加二分计算不短于k的子数组最大平均值
121. 赶牛入圈（https://www.acwing.com/problem/content/description/123/）经典离散化前缀和，双指针加二分
126. 最大的和（https://www.acwing.com/problem/content/128/）经典最大子矩形和

参考：OI WiKi（xx）
"""


class PreFixSumMatrix:
    def __init__(self, mat):
        self.mat = mat
        # 二维前缀和
        m, n = len(mat), len(mat[0])
        self.pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                self.pre[i + 1][j + 1] = self.pre[i][j + 1] + \
                    self.pre[i + 1][j] - self.pre[i][j] + mat[i][j]

    def query(self, xa, ya, xb, yb):
        # 二维子矩阵和查询，索引从 0 开始，左上角 [xa, ya] 右下角 [xb, yb]
        return self.pre[xb + 1][yb + 1] - self.pre[xb +
                                                   1][ya] - self.pre[xa][yb + 1] + self.pre[xa][ya]


class DiffArray:
    def __init__(self):
        return

    @staticmethod
    def get_diff_array(n, shifts):
        # 一维差分数组
        diff = [0] * n
        for i, j, d in shifts:
            if j + 1 < n:
                diff[j + 1] -= d
            diff[i] += d
        for i in range(1, n):
            diff[i] += diff[i - 1]
        return diff

    @staticmethod
    def get_array_prefix_sum(n, lst):
        # 一维前缀和
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + lst[i]
        return pre

    @staticmethod
    def get_array_range_sum(pre, left, right):
        # 区间元素和
        return pre[right + 1] - pre[left]


class DiffMatrix:
    def __init__(self):
        return

    @staticmethod
    def get_diff_matrix(m, n, shifts):
        # 二维差分数组
        diff = [[0] * (n + 2) for _ in range(m + 2)]
        # 索引从 1 开始，矩阵初始值为 0
        for xa, xb, ya, yb, d in shifts:  # 注意这里的行列索引范围，是从左上角到右下角
            diff[xa][ya] += d
            diff[xa][yb + 1] -= d
            diff[xb + 1][ya] -= d
            diff[xb + 1][yb + 1] += d

        for i in range(1, m + 2):
            for j in range(1, n + 2):
                diff[i][j] += diff[i - 1][j] + \
                    diff[i][j - 1] - diff[i - 1][j - 1]

        for i in range(1, m + 1):
            diff[i] = diff[i][1:n + 1]
        return diff[1: m + 1]

    @staticmethod
    def get_diff_matrix2(m, n, shifts):
        diff = [[0] * (n + 1) for _ in range(m + 1)]
        # 二维差分，索引从 0 开始， 注意这里的行列索引范围，是从左上角到右下角
        for xa, xb, ya, yb, d in shifts:
            diff[xa][ya] += d
            diff[xa][yb + 1] -= d
            diff[xb + 1][ya] -= d
            diff[xb + 1][yb + 1] += d

        res = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                res[i + 1][j + 1] = res[i + 1][j] + \
                    res[i][j + 1] - res[i][j] + diff[i][j]
        return [item[1:] for item in res[1:]]

    @staticmethod
    def get_matrix_prefix_sum(mat):
        # 二维前缀和
        m, n = len(mat), len(mat[0])
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                pre[i + 1][j + 1] = pre[i][j + 1] + \
                    pre[i + 1][j] - pre[i][j] + mat[i][j]
        return pre

    @staticmethod
    def get_matrix_range_sum(pre, xa, ya, xb, yb):
        # 二维子矩阵和
        return pre[xb + 1][yb + 1] - pre[xb + 1][ya] - \
            pre[xa][yb + 1] + pre[xa][ya]


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p3397(ac=FastIO()):
        # 模板：二维差分前缀和
        n, m = ac.read_ints()
        shifts = []
        for _ in range(m):
            x1, y1, x2, y2 = ac.read_ints()
            shifts.append([x1, x2, y1, y2, 1])
        ans = DiffMatrix().get_diff_matrix(n, n, shifts)
        for a in ans:
            ac.lst(a)
        return

    @staticmethod
    def lg_p4552(ac=FastIO()):
        # 模板：差分数组经典题，明晰差分本质
        n = ac.read_int()
        pre = -1
        pos = 0
        neg = 0
        for _ in range(n):
            num = ac.read_int()
            if pre != -1:
                if pre > num:
                    neg += pre - num
                else:
                    pos += num - pre
            pre = num
        ac.st(max(pos, neg))
        ac.st(abs(pos - neg) + 1)
        return

    @staticmethod
    def lg_p1719(ac=FastIO()):

        # 模板：求最大子矩阵和，经典枚举矩阵上下边界并使用前缀和计算
        n = ac.read_int()
        total = []
        while len(total) < n * n:
            total.extend(ac.read_list_ints())
        grid = []
        for i in range(n):
            grid.append(total[i * n: (i + 1) * n])
        del total

        ans = float("-inf")
        for i in range(n):
            lst = [0] * n
            for j in range(i, n):
                lst = [lst[k] + grid[j][k] for k in range(n)]
                floor = pre = 0
                for num in lst:
                    pre += num
                    ans = ac.max(ans, pre - floor)
                    floor = ac.min(floor, pre)
        ac.st(ans)
        return

    @staticmethod
    def cf_1722e(ac=FastIO()):
        # 模板：根据数字范围，使用二位前缀和，求解子矩阵元素和
        for _ in range(ac.read_int()):
            k, q = ac.read_ints()
            rec = [ac.read_list_ints() for _ in range(k)]
            qur = [ac.read_list_ints() for _ in range(q)]
            m = n = 1001
            dp = [[0] * n for _ in range(m)]
            for a, b in rec:
                dp[a][b] += a * b
            pre = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m):
                for j in range(n):
                    pre[i + 1][j + 1] = pre[i + 1][j] + \
                        pre[i][j + 1] - pre[i][j] + dp[i][j]

            for hs, ws, hb, wb in qur:
                hb -= 1
                wb -= 1
                hs += 1
                ws += 1
                ans = pre[hb + 1][wb + 1] - pre[hs][wb + 1] - \
                    pre[hb + 1][ws] + pre[hs][ws]
                ac.st(ans)
        return

    @staticmethod
    def lg_p2671(ac=FastIO()):
        # 模板：前后缀计数加和，分奇偶性讨论
        n, m = ac.read_ints()
        number = ac.read_list_ints()
        colors = ac.read_list_ints()
        mod = 10007
        # 转换为求相同奇偶性的 x 与 y 且颜色相同的和 x*ax+x*az+z*ax+z*az 即 (x+z)*(ax+az)
        ans = 0
        pre_sum = [[0, 0] for _ in range(m + 1)]
        pre_cnt = [[0, 0] for _ in range(m + 1)]
        for i in range(n):  # 枚举 z 计算 z*ax+z*az
            num, color = number[i], colors[i]
            k = i % 2
            z_ax = (i + 1) * pre_sum[color][k]
            z_az = (i + 1) * num * pre_cnt[color][k]
            ans += z_ax + z_az
            pre_sum[color][k] += num
            pre_cnt[color][k] += 1
            ans %= mod

        pre_sum = [[0, 0] for _ in range(m + 1)]
        pre_cnt = [[0, 0] for _ in range(m + 1)]
        for i in range(n - 1, -1, -1):  # 枚举 x 计算 x*ax+x*az
            num, color = number[i], colors[i]
            k = i % 2
            x_az = (i + 1) * pre_sum[color][k]
            x_ax = (i + 1) * num * pre_cnt[color][k]
            ans += x_ax + x_az
            pre_sum[color][k] += num
            pre_cnt[color][k] += 1
            ans %= mod

        ac.st(ans)
        return

    @staticmethod
    def cf_1795c(ac=FastIO()):
        # 模板：根据作用域进行差分数组计数
        for _ in range(ac.read_int()):
            n = ac.read_int()
            a = ac.read_list_ints()
            b = ac.read_list_ints()
            pre = [0] * (n + 1)
            for i in range(n):
                pre[i + 1] = pre[i] + b[i]

            ans = [0] * n
            diff = [0] * n
            for i in range(n):
                j = bisect.bisect_left(pre, pre[i] + a[i])
                if j == n + 1 or pre[j] > pre[i] + a[i]:
                    j -= 1
                diff[i] += 1
                if j < n:
                    diff[j] -= 1
                if pre[j] - pre[i] < a[i]:
                    if j < n:
                        ans[j] += a[i] - (pre[j] - pre[i])
            for i in range(1, n):
                diff[i] += diff[i - 1]

            for i in range(n):
                ans[i] += b[i] * diff[i]
            ac.lst(ans)
        return

    @staticmethod
    def cf_1355c(a, b, c, d):
        # 模板: 经典使用作用域差分计算，合法三角形边长个数
        diff = [0] * (b + c + 1)
        for x in range(a, b + 1):
            diff[x + b] += 1
            if x + c + 1 <= b + c:
                diff[x + c + 1] -= 1
        for i in range(1, b + c + 1):
            diff[i] += diff[i - 1]

        for i in range(1, b + c + 1):
            diff[i] += diff[i - 1]

        ans = 0
        for z in range(c, d + 1):
            ans += diff[-1] - diff[min(z, b + c)]
        return ans

    @staticmethod
    def lc_2281(strength: List[int]) -> int:
        # 模板：使用单调栈确定加和范围，再使用前缀和的前缀和进行计数
        n = len(strength)
        mod = 10 ** 9 + 7

        # 单调栈
        pre = [0] * n
        post = [n - 1] * n
        stack = []
        for i in range(n):
            while stack and strength[stack[-1]] >= strength[i]:
                post[stack.pop()] = i - 1
            if stack:
                pre[i] = stack[-1] + 1
            stack.append(i)

        # 前缀和
        s = [0] * (n + 1)
        for i in range(n):
            s[i + 1] = s[i] + strength[i]

        # 前缀和的前缀和
        ss = [0] * (n + 2)
        for i in range(n + 1):
            ss[i + 1] = ss[i] + s[i]

        # 遍历加和
        ans = 0
        for i in range(n):
            left, right = pre[i], post[i]
            ans += strength[i] * ((i - left + 1) * (ss[right + 2] -
                                  ss[i + 1]) - (right - i + 1) * (ss[i + 1] - ss[left]))
            ans %= mod
        return ans

    @staticmethod
    def lc_2565(s: str, t: str) -> int:
        # 模板：使用前后缀贪心枚举前后缀最长匹配
        m, n = len(s), len(t)
        pre = [0] * (m + 1)
        ind = 0
        for i in range(m):
            if ind < n and s[i] == t[ind]:
                ind += 1
            pre[i + 1] = ind
        if ind == n:
            return 0

        post = [0] * (m + 1)
        ind = 0
        for i in range(m - 1, -1, -1):
            if ind < n and s[i] == t[-ind - 1]:
                ind += 1
            post[i] = ind

        ans = min(n - (post[i] + pre[i]) for i in range(m + 1))
        return ans

    @staticmethod
    def lg_p2882(ac=FastIO()):

        # 模板：贪心枚举加差分验证
        n = ac.read_int()
        lst = [int(ac.read_str() == "F") for _ in range(n)]
        ans = [inf, 0]
        for k in range(1, n + 1):
            diff = [0] * n
            m = 0
            for i in range(n - k + 1):
                if i:
                    diff[i] += diff[i - 1]
                x = diff[i] + lst[i]
                if x % 2:
                    continue
                else:
                    m += 1
                    diff[i] += 1
                    if i + k < n:
                        diff[i + k] -= 1
            for i in range(n - k + 1, n):
                diff[i] += diff[i - 1]
                if (diff[i] + lst[i]) % 2 == 0:
                    break
            else:
                if [m, k] < ans:
                    ans = [m, k]
        ac.lst(ans[::-1])
        return

    @staticmethod
    def cf_1772d(ac=FastIO()):
        # 模板：离散化差分数组作用域计数
        ceil = 10 ** 9
        for _ in range(ac.read_int()):
            n = ac.read_int()
            diff = defaultdict(int)
            nums = ac.read_list_ints()
            for i in range(1, n):
                a, b = nums[i - 1], nums[i]
                if a == b:
                    diff[0] += 1
                    diff[ceil + 1] -= 1
                elif a < b:
                    mid = a + (b - a) // 2
                    diff[0] += 1
                    diff[mid + 1] -= 1
                else:
                    mid = b - (b - a) // 2
                    diff[mid] += 1
                    diff[ceil + 1] -= 1

            axis = sorted(list(diff.keys()))
            m = len(axis)
            for i in range(m):
                if i:
                    diff[axis[i]] += diff[axis[i - 1]]
                if diff[axis[i]] == n - 1:
                    ac.st(axis[i])
                    break
            else:
                ac.st(-1)
        return

    @staticmethod
    def ac_99(ac=FastIO()):
        # 模板：二维前缀和
        n, m = ac.read_ints()

        lst = [ac.read_list_ints() for _ in range(n)]
        length = max(max(ls[:-1]) for ls in lst) + 1
        grid = [[0] * length for _ in range(length)]
        for x, y, v in lst:
            grid[x][y] += v
        ans = 0
        dp = [[0] * (length + 1) for _ in range(length + 1)]
        for i in range(length):
            for j in range(length):
                dp[i + 1][j + 1] = dp[i][j + 1] + \
                    dp[i + 1][j] - dp[i][j] + grid[i][j]
                a, b = max(i - m + 1, 0), max(j - m + 1, 0)
                cur = dp[i + 1][j + 1] - dp[i + 1][b] - dp[a][j + 1] + dp[a][b]
                ans = ans if ans > cur else cur
        ac.st(ans)
        return

    @staticmethod
    def ac_102(ac=FastIO()):

        # 模板：前缀和加二分计算不短于k的子数组最大平均值
        n, f = ac.read_ints()
        nums = [ac.read_int() for _ in range(n)]

        def check(x):
            y = 0
            pre = [0] * (n + 1)
            for i in range(n):
                y += nums[i] * 1000 - x
                pre[i + 1] = pre[i] if pre[i] < y else y
                if i >= f - 1 and y - pre[i - f + 1] >= 0:
                    return True
            return False

        error = 1
        low = 0
        high = max(nums) * 1000
        while low < high - error:
            mid = low + (high - low) // 2
            if check(mid):
                low = mid
            else:
                high = mid
        ac.st(high if check(high) else low)
        return

    @staticmethod
    def ac_121(ac=FastIO()):
        # 模板：离散化前缀和，双指针加二分
        c, b = ac.read_ints()
        nums = [ac.read_list_ints() for _ in range(b)]
        lst_x = sorted(list(set([x for x, _ in nums])))
        lst_y = sorted(list(set([x for _, x in nums])))
        m = len(lst_x)
        n = len(lst_y)
        ind_x = {num: i for i, num in enumerate(lst_x)}
        ind_y = {num: i for i, num in enumerate(lst_y)}
        grid = [[0] * (n + 1) for _ in range(m + 1)]
        for x, y in nums:
            grid[ind_x[x] + 1][ind_y[y] + 1] += 1
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                grid[i][j] = grid[i - 1][j] + grid[i][j - 1] - \
                    grid[i - 1][j - 1] + grid[i][j]

        def check(xx):
            up = 0
            for i in range(m):
                while up < m and lst_x[up] - lst_x[i] <= xx - 1:
                    up += 1
                right = 0
                for j in range(n):
                    while right < n and lst_y[right] - lst_y[j] <= xx - 1:
                        right += 1
                    cur = grid[up][right] - grid[up][j] - \
                        grid[i][right] + grid[i][j]
                    if cur >= c:
                        return True

            return False

        low = 0
        high = 10000
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                high = mid
            else:
                low = mid
        ans = low if check(low) else high
        ac.st(ans)
        return

    @staticmethod
    def ac_126(ac=FastIO()):
        # 模板：经典最大子矩形和
        n = ac.read_int()
        nums = []
        while len(nums) < n * n:
            nums.extend(ac.read_list_ints())
        grid = [nums[i:i + n] for i in range(0, n * n, n)]
        del nums
        ans = grid[0][0]
        for i in range(n):
            pre = [0] * n
            for k in range(i, n):
                pre = [pre[j] + grid[k][j] for j in range(n)]
                floor = 0
                x = 0
                for j in range(n):
                    x += pre[j]
                    ans = ac.max(ans, x - floor)
                    floor = ac.min(floor, x)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1627(ac=FastIO()):
        # 模板：经典前后缀中位数大小值差值计数
        n, b = ac.read_ints()
        nums = ac.read_list_ints()
        i = nums.index(b)

        # 前缀差值计数
        pre = defaultdict(int)
        cnt = ans = 0
        for j in range(i - 1, -1, -1):
            num = nums[j]
            cnt += 1 if num > b else -1
            pre[cnt] += 1
            if cnt == 0:  # 只取前缀
                ans += 1

        # 后缀差值计数
        cnt = 0
        for j in range(i + 1, n):
            num = nums[j]
            cnt += 1 if num > b else -1
            ans += pre[-cnt]  # 取前后缀
            ans += 1 if not cnt else 0  # 只取后缀
        ans += 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p1895(ac=FastIO()):

        # 模板：前缀和计数加二分查找，最多不超多10**5
        n = 10**5
        dp = [0] * (n + 1)
        for i in range(1, n + 1):  # 序列1234..
            dp[i] = dp[i - 1] + len(str(i))

        # 序列1121231234..
        pre = [0] * (n + 1)
        for i in range(1, n + 1):
            pre[i] = pre[i - 1] + dp[i]

        def check(x):
            ii = bisect.bisect_left(pre, x)
            rest = x - pre[ii - 1]
            j = bisect.bisect_left(dp, rest)
            d = rest - dp[j - 1]
            return str(j)[d - 1]

        for _ in range(ac.read_int()):
            ac.st(check(ac.read_int()))

        return

    @staticmethod
    def lg_p1982(ac=FastIO()):

        # 模板：前缀最大连续子段和与前缀最大值
        n, p = ac.read_ints()
        nums = ac.read_list_ints()
        pre = 0
        for i in range(n):
            pre = pre if pre > 0 else 0
            pre += nums[i]
            nums[i] = pre
            if i:
                nums[i] = ac.max(nums[i], nums[i - 1])

        final = nums[0]
        pre = nums[0] * 2
        for i in range(1, n):
            final = ac.max(final, pre)
            pre = ac.max(pre, pre + nums[i])
        pos = 1 if final > 0 else -1
        ac.st(pos * (abs(final) % p))
        return

    @staticmethod
    def lg_p2070(ac=FastIO()):
        # 模板：哈希离散化差分数组计数
        n = ac.read_int()
        pos = 0
        diff = defaultdict(int)
        for _ in range(n):
            dis, op = ac.read_list_strs()
            dis = int(dis)
            if op == "L":
                diff[pos - dis] += 1
                diff[pos] -= 1
                pos -= dis
            else:
                diff[pos] += 1
                diff[pos + dis] -= 1
                pos += dis
        # 从小到大计算区间占有次数
        axis = sorted(diff.keys())
        m = len(axis)
        ans = 0
        for i in range(1, m):
            diff[axis[i]] += diff[axis[i - 1]]
            if diff[axis[i - 1]] >= 2:
                ans += axis[i] - axis[i - 1]
        ac.st(ans)
        return

    @staticmethod
    def lg_p2190(ac=FastIO()):
        # 模板：环形数组差分
        n, m = ac.read_ints()
        diff = [0] * n
        for _ in range(m):
            x, y, z = ac.read_ints()
            x -= 1
            y -= 1
            if x <= y:
                diff[x] += z
                diff[y] -= z
            else:
                diff[x] += z
                if y > 0:
                    diff[0] += z
                    diff[y] -= z
        for i in range(1, n):
            diff[i] += diff[i - 1]
        ac.st(math.ceil(max(diff) / 36))
        return

    @staticmethod
    def lg_p2352(ac=FastIO()):
        # 模板：离散化差分
        diff = defaultdict(int)
        for _ in range(ac.read_int()):
            a, b = ac.read_ints()
            diff[a] += 1
            diff[b + 1] -= 1
        axis = sorted(list(diff.keys()))
        m = len(axis)
        ans = 0
        for i in range(1, m):
            diff[axis[i]] += diff[axis[i - 1]]
            # 注意此时选择右端点
            ans = ac.max(ans, diff[axis[i - 1]] * (axis[i] - 1))
        ac.st(ans)
        return

    @staticmethod
    def lg_p2363(ac=FastIO()):
        # 模板：二维前缀和与枚举
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        pre = PreFixSumMatrix(nums)
        ans = 0
        for i in range(n):
            for j in range(n):
                dct = defaultdict(int)
                for x in range(i + 1):
                    for y in range(j + 1):
                        dct[pre.query(x, y, i, j)] += 1
                for p in range(i + 1, n):
                    for q in range(j + 1, n):
                        ans += dct[pre.query(i + 1, j + 1, p, q)]

                dct = defaultdict(int)
                for x in range(i + 1):
                    for y in range(j, n):
                        dct[pre.query(x, j, i, y)] += 1
                for p in range(i + 1, n):
                    for q in range(j):
                        ans += dct[pre.query(i + 1, q, p, j - 1)]
        ac.st(ans)
        return

    @staticmethod
    def lg_p2706(ac=FastIO()):
        # 模板：不包含障碍点的最大子矩阵和
        m, n = ac.read_ints()
        grid = []
        while len(grid) < m * n:
            grid.extend(ac.read_list_ints())
        ans = 0
        for i in range(m):
            lst = [0] * n
            for j in range(i, m):
                floor = pre = 0
                for k in range(n):
                    num = grid[j * n + k]
                    lst[k] += num if num != 0 else -inf
                    floor = 0 if floor < 0 else floor
                    pre = pre if pre > 0 else 0
                    num = lst[k]
                    pre += num
                    ans = ans if ans > pre - floor else pre - floor
        ac.st(ans)
        return

    @staticmethod
    def lg_p2879(ac=FastIO()):
        # 模板：差分数组经典题与贪心
        n, i, h, r = ac.read_ints()
        diff = [0] * n
        pre = set()
        for _ in range(r):
            a, b = ac.read_ints_minus_one()
            if (a, b) in pre:
                continue
            pre.add((a, b))
            if a < b:
                diff[a + 1] -= 1
                diff[b] += 1
            else:
                diff[b + 1] -= 1
                diff[a] += 1
        for i in range(1, n):
            diff[i] += diff[i - 1]
        gap = h - diff[i]
        for d in diff:
            ac.st(d + gap)
        return

    @staticmethod
    def lg_p3028(ac=FastIO()):
        # 模板：离散化差分计算覆盖区间最多的点
        n = ac.read_int()
        diff = defaultdict(int)
        for _ in range(n):
            a, b = ac.read_ints()
            if a > b:
                a, b = b, a
            diff[a] += 1
            diff[b + 1] -= 1
            # 增加右端点避免离散化带来重合
            diff[b] += 0
        axis = sorted(list(diff.keys()))
        ans = diff[axis[0]]
        m = len(axis)
        for i in range(1, m):
            # 当前点覆盖的区间数
            diff[axis[i]] += diff[axis[i - 1]]
            ans = ac.max(ans, diff[axis[i]])
        ac.st(ans)
        return

    @staticmethod
    def lg_p4030(ac=FastIO()):
        # 模板：脑筋急转弯加二维前缀和计算
        m, n, t = ac.read_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        mat = [[0] * n for _ in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                a, b = grid[i - 1][j - 1], grid[i - 1][j]
                c, d = grid[i][j - 1], grid[i][j]
                if a + d == b + c:
                    mat[i][j] = 1
        pm = PreFixSumMatrix(mat)
        for i in range(t):
            x, y, k = ac.read_ints()
            if k == 1:
                ac.st("Y")
                continue
            x -= 1
            y -= 1
            if pm.query(x + 1, y + 1, x + k - 1, y +
                        k - 1) == (k - 1) * (k - 1):
                ac.st("Y")
            else:
                ac.st("N")
        return

    @staticmethod
    def lg_p4440(ac=FastIO()):
        # 模板：典型前缀和计数
        s = ac.read_str()
        pre = []
        cnt = [0] * 26
        pre.append(cnt[:])
        for w in s:
            cnt[ord(w) - ord("a")] += 1
            pre.append(cnt[:])
        for _ in range(ac.read_int()):
            a, b, c, d = ac.read_ints_minus_one()
            if d - c != b - a:
                ac.st("NE")
                continue
            if all(pre[b + 1][j] - pre[a][j] == pre[d + 1]
                   [j] - pre[c][j] for j in range(26)):
                ac.st("DA")
            else:
                ac.st("NE")
        return

    @staticmethod
    def lg_p4623(ac=FastIO()):
        # 模板：离散化差分计数
        n = ac.read_int()
        m = 10**6 + 1
        diff_x = [0] * m
        diff_y = [0] * m
        for _ in range(n):
            x1, y1, x2, y2, x3, y3 = ac.read_ints()
            low_x = min(x1, x2, x3)
            high_x = max(x1, x2, x3)
            low_y = min(y1, y2, y3)
            high_y = max(y1, y2, y3)
            diff_x[low_x + 1] += 1
            diff_x[high_x] -= 1
            diff_y[low_y + 1] += 1
            diff_y[high_y] -= 1

        # 进行差分计算
        for i in range(1, m):
            diff_x[i] += diff_x[i - 1]
        for i in range(1, m):
            diff_y[i] += diff_y[i - 1]

        # 查询
        for _ in range(ac.read_int()):
            op, _, num = ac.read_list_strs()
            num = int(num)
            if op == "x":
                ac.st(diff_x[num])
            else:
                ac.st(diff_y[num])
        return

    @staticmethod
    def lg_p6032(ac=FastIO()):
        # 模板：经典前后缀计数
        n, k, p = ac.read_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        post = [0] * k
        for i in range(n):
            post[nums[i][0]] += 1

        pre = dict()
        ans = ss = 0
        for i in range(n):
            cc, pp = nums[i]
            if pp <= p:
                # 直接清空，前序ss个
                ans += ss + post[cc] - 1
                ss = 0
                pre = dict()
                post[cc] -= 1
                continue

            pre[cc] = pre.get(cc, 0) + 0
            ss -= pre.get(cc, 0)
            pre[cc] += 1

            post[cc] -= 1
            ss += post[cc]
        ac.st(ans)
        return

    @staticmethod
    def lg_p6070(ac=FastIO()):
        # 模板：经典二维差分贪心修改实时维护差分与计算前缀和即矩阵最新值
        n, m, k = ac.read_ints()
        grid = [[0] * n for _ in range(n)]
        for _ in range(m):
            x, y, z = ac.read_ints_minus_one()
            grid[x][y] = z + 1
        diff = [[0] * (n + 2) for _ in range(n + 2)]

        ans = 0
        for i in range(n):
            for j in range(n):
                diff[i + 1][j + 1] += diff[i + 1][j] + \
                    diff[i][j + 1] - diff[i][j]
                d = diff[i + 1][j + 1] + grid[i][j]
                # 二维差分，索引从 0 开始， 注意这里的行列索引范围，是从左上角到右下角
                if d:
                    if i + k + 1 > n + 1 or j + k + 1 > n + 1:
                        ac.st(-1)
                        return
                    diff[i + 1][j + 1] -= d
                    diff[i + 1][j + k + 1] += d
                    diff[i + k + 1][j + 1] += d
                    diff[i + k + 1][j + k + 1] -= d
                    ans += abs(d)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6278(ac=FastIO()):
        # 模板：经典逆序对作用域与差分前缀和计算
        n = ac.read_int()
        nums = ac.read_list_ints()
        diff = [0] * (n + 1)
        pre = []
        for num in nums:
            # num 作为最小值时的逆序对个数
            diff[num] += len(pre) - bisect.bisect_right(pre, num)
            bisect.insort_left(pre, num)
        diff = ac.accumulate(diff)
        for i in range(n):
            # 减少到 i 时前面小于 i 的对应逆序对不受改变
            ac.st(diff[i])
        return

    @staticmethod
    def lg_p6537(ac=FastIO()):
        # 模板：预处理前缀和加枚举
        n = ac.read_int()
        grid = [ac.read_list_ints() for _ in range(n)]
        pre = PreFixSumMatrix(grid)
        ans = 0
        for i in range(n):
            for j in range(n):
                # 左上右下
                dct = dict()
                for x in range(i + 1):
                    for y in range(j + 1):
                        val = pre.query(x, y, i, j)
                        dct[val] = dct.get(val, 0) + 1
                for x in range(i + 1, n):
                    for y in range(j + 1, n):
                        val = pre.query(i + 1, j + 1, x, y)
                        ans += dct.get(val, 0)
                # 左下右上
                dct = defaultdict(int)
                for x in range(i + 1):
                    for y in range(j, n):
                        val = pre.query(x, j, i, y)
                        dct[val] = dct.get(val, 0) + 1
                for x in range(i + 1, n):
                    for y in range(j):
                        val = pre.query(i + 1, y, x, j - 1)
                        ans += dct.get(val, 0)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6877(ac=FastIO()):
        # 模板：排序贪心前后缀 DP 枚举
        n = ac.read_int()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        aa = a[:]

        a.sort()
        b.sort()
        pre = [0] * (n + 2)
        for i in range(n):
            pre[i + 1] = ac.max(pre[i], a[i] - b[i])

        post = [0] * (n + 2)
        for i in range(n - 1, -1, -1):
            post[i] = ac.max(post[i + 1], a[i + 1] - b[i])

        ans = dict()
        for i in range(n + 1):
            ans[a[i]] = ac.max(pre[i], post[i])
        ac.lst([ans[x] for x in aa])
        return

    @staticmethod
    def lg_p6878(ac=FastIO()):
        # 模板：前后缀枚举
        n, k = ac.read_ints()
        s = ac.read_str()
        pre = [-1] * n
        stack = deque()
        for i in range(n):
            while len(stack) > k:
                stack.popleft()
            if s[i] == "J":
                stack.append(i)
            elif s[i] == "O":
                if len(stack) == k:
                    pre[i] = stack[0]

        post_o = [-1] * n
        post = [-1] * n
        stack = deque()
        stack_o = deque()
        for i in range(n - 1, -1, -1):
            while len(stack) > k:
                stack.popleft()
            if s[i] == "I":
                stack.append(i)
            if s[i] == "O":
                stack_o.append(i)
            while len(stack_o) > k:
                stack_o.popleft()
            if s[i] == "O":
                if len(stack) == k:
                    post[i] = stack[0]
                if len(stack_o) == k:
                    post_o[i] = stack_o[0]

        ans = inf
        for i in range(n):
            if pre[i] != -1 and post_o[i] != -1 and post[post_o[i]] != -1:
                ans = ac.min(ans, post[post_o[i]] - pre[i] + 1 - 3 * k)
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def lg_p8081(ac=FastIO()):
        # 模板：差分计数计算作用域
        n = ac.read_int()
        nums = ac.read_list_ints()
        diff = [0] * (n + 1)
        pre = 0
        ceil = 0
        for i in range(n):
            if nums[i] < 0:
                pre += 1
            else:
                if pre:
                    ceil = ac.max(ceil, pre)
                    low, high = ac.max(0, i - 3 * pre), i - pre - 1
                    if low <= high:
                        diff[low] += 1
                        diff[high + 1] -= 1
                pre = 0
        if pre:
            ceil = ac.max(ceil, pre)
            low, high = ac.max(0, n - 3 * pre), n - pre - 1
            if low <= high:
                diff[low] += 1
                diff[high + 1] -= 1

        diff = [int(num > 0) for num in ac.accumulate(diff)][1:n + 1]
        diff = ac.accumulate(diff)
        ans = diff[-1]
        pre = 0
        res = 0
        for i in range(n):
            if nums[i] < 0:
                pre += 1
            else:
                if pre == ceil:
                    low, high = i - 4 * pre, i - 3 * pre - 1
                    low = ac.max(low, 0)
                    if low <= high:
                        res = ac.max(res, high - low + 1 -
                                     diff[high + 1] + diff[low])
                pre = 0
        if pre == ceil:
            low, high = n - 4 * pre, n - 3 * pre - 1
            low = ac.max(low, 0)
            if low <= high:
                res = ac.max(res, high - low + 1 - diff[high + 1] + diff[low])
        ac.st(ans + res)
        return

    @staticmethod
    def lg_p8033(ac=FastIO()):
        # 模板：二维前缀和计数
        m, n, k = ac.read_list_ints()
        grid = [list(ac.read_str()) for _ in range(m)]
        mat = [[int(w == "*") for w in lst] for lst in grid]
        pre = PreFixSumMatrix(mat)
        val = 0
        ans = []

        for i in range(k - 1, m):
            for j in range(k - 1, n):
                cur = pre.query(i - k + 2, j - k + 2, i - 1, j - 1)
                if cur > val:
                    val = cur
                    ans = [i, j]
        i, j = ans
        x1, y1, x2, y2 = i - k + 1, j - k + 1, i, j
        grid[x1][y1] = grid[x1][y2] = "+"
        grid[x2][y1] = grid[x2][y2] = "+"
        for i in [x1, x2]:
            for j in range(y1 + 1, y2):
                grid[i][j] = "-"
        for j in [y1, y2]:
            for i in range(x1 + 1, x2):
                grid[i][j] = "|"
        ac.st(val)
        for g in grid:
            ac.st("".join(g))
        return

    @staticmethod
    def lg_p7992(ac=FastIO()):
        # 模板：经典桶计数与作用域差分计数
        n, m = ac.read_ints()
        a = [0] * (m + 1)
        b = [0] * (m + 1)
        diff = [0] * (2 * m + 2)
        for _ in range(n):
            x, y = ac.read_ints()
            a[x] += 1
            b[y] += 1
        for i in range(m + 1):
            for j in range(m + 1):
                diff[i + j] += a[i] * a[j]
                diff[i + j + 1] -= b[i] * b[j]
        for i in range(2 * m + 1):
            if i:
                diff[i] += diff[i - 1]
            ac.st(diff[i])
        return

    @staticmethod
    def lg_p7948(ac=FastIO()):
        # 模板：排序后预处理前后缀信息指针查询
        for _ in range(ac.read_int()):
            n, q = ac.read_ints()
            a = ac.read_list_ints()
            b = ac.read_list_ints()
            a.sort(reverse=True)
            pre = [0] * n
            s = 0
            for i in range(n):
                s += a[i]
                pre[i] = (i+1)*a[i] - s

            ind = list(range(q))
            ind.sort(key=lambda it: -b[it])
            ans = [-1] * q
            j = n - 1
            for i in ind:
                k = b[i]
                while j >= 0 and pre[j] < -k*(j+1):
                    j -= 1
                ans[i] = j + 1
            ac.lst(ans)
        return

    @staticmethod
    def lg_p8343(ac=FastIO()):
        # 模板：经典子矩阵前缀和枚举与双指针
        m, n, a, b = ac.read_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        if a > b:
            a, b = b, a
        pre = PreFixSumMatrix(grid)
        ans = inf
        for i in range(m):
            for k in range(i, m):
                lst = [0]
                ind_a = ind_b = 0
                for j in range(n):
                    cur = pre.query(i, 0, k, j)
                    lst.append(cur)
                    while ind_a + 1 < j+1 and cur - lst[ind_a] >= a:
                        ans = ac.min(ans, abs(cur - lst[ind_a] - a) + abs(cur - lst[ind_a] - b))
                        ind_a += 1

                    while ind_b + 1 < j+1 and cur-lst[ind_b] <= b:
                        ans = ac.min(ans, abs(cur - lst[ind_b] - a) + abs(cur - lst[ind_b] - b))
                        ind_b += 1

                    ans = ac.min(ans, abs(cur-lst[ind_a]-a)+abs(cur-lst[ind_a]-b))
                    ans = ac.min(ans, abs(cur - lst[ind_b] - a) + abs(cur - lst[ind_b] - b))
                    if ans == b-a:
                        ac.st(ans)
                        return
        ac.st(ans)
        return

    @staticmethod
    def lg_p8551(ac=FastIO()):
        # 模板：差分数组经典灵活应用
        n = ac.read_int()
        m = 3 * 10**5 + 1
        diff = [0] * (m + 2)
        point = [0] * (m + 2)
        for _ in range(n):
            a, b = ac.read_ints()
            diff[a] += 1
            diff[b + 1] -= 1
            point[a - 1] = 1
            point[b] = 1
        ans = 0
        pre = inf
        for i in range(1, m + 2):
            diff[i] += diff[i - 1]
            if point[i]:
                ans = ac.max(ans, diff[i] * (i - pre))
                pre = i + 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p8666(ac=FastIO()):
        # 模板：二分加三维差分经典题
        a, b, c, m = ac.read_ints()
        grid = [[[0] * (c + 1) for _ in range(b + 1)] for _ in range(a + 1)]
        nums = ac.read_list_ints()
        for i in range(1, a + 1):
            for j in range(1, b + 1):
                for k in range(1, c + 1):
                    grid[i][j][k] = nums[((i - 1) * b + (j - 1)) * c + k - 1]
        del nums
        lst = [ac.read_list_ints() for _ in range(m)]

        def check(x):
            # 模板：三位差分计算
            diff = [[[0] * (c + 2) for _ in range(b + 2)] for _ in range(a + 2)]
            for i1, i2, j1, j2, k1, k2, h in lst[:x]:

                # 差分值更新索引从 1 开始
                diff[i1][j1][k1] += h

                diff[i2 + 1][j1][k1] -= h
                diff[i1][j2 + 1][k1] -= h
                diff[i1][j1][k2 + 1] -= h

                diff[i2 + 1][j2 + 1][k1] += h
                diff[i1][j2 + 1][k2 + 1] += h
                diff[i2 + 1][j1][k2 + 1] += h

                diff[i2 + 1][j2 + 1][k2 + 1] -= h

            for i1 in range(1, a + 1):
                for j1 in range(1, b + 1):
                    for k1 in range(1, c + 1):
                        # 前缀和计算索引从 1 开始
                        diff[i1][j1][k1] += diff[i1 - 1][j1][k1] + diff[i1][j1 - 1][k1] + diff[i1][j1][k1 - 1] - \
                            diff[i1][j1 - 1][k1 - 1] - diff[i1 - 1][j1][k1 - 1] - diff[i1 - 1][j1 - 1][k1] +\
                            diff[i1 - 1][j1 - 1][k1 - 1]
                        if diff[i1][j1][k1] > grid[i1][j1][k1]:
                            return True

            return False

        ans = BinarySearch().find_int_left(1, m, check)
        ac.st(ans)
        return

    @staticmethod
    def lc_1292(mat: List[List[int]], threshold: int) -> int:
        # 模板：经典O(mn)复杂度枚举
        m, n = len(mat), len(mat[0])
        ans = 0
        pre = PreFixSumMatrix(mat)
        for i in range(m):
            for j in range(n):
                r = n-j if n-j < m-i else m-i
                for d in range(ans+1, r+1):
                    cur = pre.query(i, j, i+d-1, j+d-1)
                    if cur > threshold:
                        break
                    ans = d
        return ans

    @staticmethod
    def lc_1738(matrix: List[List[int]], k: int) -> int:

        # 模板：经典二维前缀异或和
        m, n = len(matrix), len(matrix[0])
        # 原地进行异或运算
        for i in range(1, m):
            matrix[i][0] = matrix[i][0] ^ matrix[i - 1][0]
        for j in range(1, n):
            matrix[0][j] = matrix[0][j] ^ matrix[0][j - 1]
        for i in range(1, m):
            for j in range(1, n):
                matrix[i][j] = matrix[i - 1][j - 1] ^ matrix[i - 1][j] ^ matrix[i][j - 1] ^ matrix[i][j]

        # 排序后返回结果
        lst = []
        for i in range(m):
            lst.extend(matrix[i])
        lst.sort()
        return lst[-k]


class TestGeneral(unittest.TestCase):

    def test_diff_array_range(self):
        dar = DiffArray()
        n = 3
        shifts = [[0, 1, 1], [1, 2, -1]]
        diff = dar.get_diff_array(n, shifts)
        assert diff == [1, 0, -1]

        n = 3
        shifts = [1, 2, 3]
        pre = dar.get_array_prefix_sum(n, shifts)
        assert pre == [0, 1, 3, 6]

        left = 1
        right = 2
        assert dar.get_array_range_sum(pre, left, right) == 5
        return

    def test_diff_array_matrix(self):
        dam = DiffMatrix()
        m = 3
        n = 3
        # 索引从1开始
        shifts = [[1, 2, 1, 2, 1], [2, 3, 2, 3, 1],
                  [2, 2, 2, 2, 2], [1, 1, 3, 3, 3]]
        diff = [[1, 1, 3], [1, 4, 1], [0, 1, 1]]
        assert dam.get_diff_matrix(m, n, shifts) == diff

        shifts = [[1, 2, 1, 2, 1], [2, 3, 2, 3, 1],
                  [2, 2, 2, 2, 2], [1, 1, 3, 3, 3]]
        shifts = [[x - 1 for x in ls[:-1]] + [ls[-1]] for ls in shifts]
        assert dam.get_diff_matrix2(m, n, shifts) == diff

        pre = dam.get_matrix_prefix_sum(diff)
        assert pre == [[0, 0, 0, 0], [0, 1, 2, 5],
                       [0, 2, 7, 11], [0, 2, 8, 13]]

        xa, ya, xb, yb = 1, 1, 2, 2
        assert dam.get_matrix_range_sum(pre, xa, ya, xb, yb) == 7
        return


if __name__ == '__main__':
    unittest.main()