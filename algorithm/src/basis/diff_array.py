import unittest
from typing import List

from algorithm.src.fast_io import FastIO
import bisect

"""

算法：差分数组与前缀和、后缀和、前缀最大子序列和、后缀最大子序列和、二维差分、离散化差分
功能：用来解决一维数组或者二维数组的加和问题，以及前缀和计算，还有前缀和的前缀和
题目：

===================================力扣===================================
2281. 巫师的总力量（https://leetcode.cn/problems/sum-of-total-strength-of-wizards/）枚举当前元素作为最小值的子数组和并使用前缀和的前缀和计算
2251. 花期内花的数目（https://leetcode.cn/problems/number-of-flowers-in-full-bloom/）离散化差分数组
2132. 用邮票贴满网格图（https://leetcode.cn/problems/stamping-the-grid/）用前缀和枚举可行的邮票左上端点，然后查看空白格点左上方是否有可行的邮票点
1229. 安排会议日程（https://leetcode.cn/problems/meeting-scheduler/）离散化差分数组
6292. 子矩阵元素加 1（https://leetcode.cn/problems/increment-submatrices-by-one/)二维差分前缀和

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
P3397 地毯（https://www.luogu.com.cn/problem/P3397#submit）
P1869 愚蠢的组合数（https://www.luogu.com.cn/problem/P1869）使用前缀和记录1-N的因子2的个数继而计算C(N,K)的奇偶性
P7667 [JOI2018] Art Exhibition（https://www.luogu.com.cn/problem/P7667）公式变换，排序后使用前缀和

================================CodeForces================================
https://codeforces.com/problemset/problem/33/C（前后缀最大变换和与分割点枚举，经典类型题目）
https://codeforces.com/problemset/problem/797/C（后缀最小值字典序模拟）
https://codeforces.com/problemset/problem/75/D（压缩数组最大子段和）
C. Count Triangles（https://codeforces.com/problemset/problem/1355/C）经典使用作用域差分计算，合法三角形边长个数
C. Tea Tasting（https://codeforces.com/problemset/problem/1795/C）前缀和二分后，经典使用差分计数模拟加和

参考：OI WiKi（xx）
"""


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
                diff[i][j] += diff[i - 1][j] + diff[i][j - 1] - diff[i - 1][j - 1]

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
                res[i + 1][j + 1] = res[i + 1][j] + res[i][j + 1] - res[i][j] + diff[i][j]
        return [item[1:] for item in res[1:]]

    @staticmethod
    def get_matrix_prefix_sum(mat):
        # 二维前缀和
        m, n = len(mat), len(mat[0])
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                pre[i + 1][j + 1] = pre[i][j + 1] + pre[i + 1][j] - pre[i][j] + mat[i][j]
        return pre

    @staticmethod
    def get_matrix_range_sum(pre, xa, ya, xb, yb):
        # 二维子矩阵和
        return pre[xb + 1][yb + 1] - pre[xb + 1][ya] - pre[xa][yb + 1] + pre[xa][ya]


class Solution:
    def __init__(self):
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
            ans += strength[i] * ((i - left + 1) * (ss[right + 2] - ss[i + 1]) - (right - i + 1) * (ss[i + 1] - ss[left]))
            ans %= mod
        return ans


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
        shifts = [[x-1 for x in ls[:-1]]+[ls[-1]] for ls in shifts]
        assert dam.get_diff_matrix2(m, n, shifts) == diff

        pre = dam.get_matrix_prefix_sum(diff)
        assert pre == [[0, 0, 0, 0], [0, 1, 2, 5], [0, 2, 7, 11], [0, 2, 8, 13]]

        xa, ya, xb, yb = 1, 1, 2, 2
        assert dam.get_matrix_range_sum(pre, xa, ya, xb, yb) == 7
        return


if __name__ == '__main__':
    unittest.main()
