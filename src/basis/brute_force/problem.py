"""
算法：暴力枚举、旋转矩阵、螺旋矩阵、brute_force、贡献法
功能：根据题意，在复杂度有限的情况下，进行所有可能情况的枚举
题目：

===================================力扣===================================
670. 最大交换（https://leetcode.cn/problems/maximum-swap/）看似贪心，在复杂度允许的情况下使用枚举暴力保险
395. 至少有 K 个重复字符的最长子串（https://leetcode.cn/problems/longest-substring-with-at-least-k-repeating-characters/）经典枚举分治
1330. 翻转子数组得到最大的数组值（https://leetcode.cn/problems/reverse-subarray-to-maximize-array-value/）经典枚举绝对值正负数
2488. 统计中位数为 K 的子数组（https://leetcode.cn/problems/count-subarrays-with-median-k/）利用中位数的定义枚举前后子序列中满足大于 K 和小于 K 的数个数相等的子数组
2484. 统计回文子序列数目（https://leetcode.cn/problems/count-palindromic-subsequences/）利用前后缀哈希计数枚举当前索引作为回文中心的回文子串的前后缀个数
2322. 从树中删除边的最小分数（https://leetcode.cn/problems/minimum-score-after-removals-on-a-tree/）枚举删除的第一条边后使用树形递归再枚举第二条边计算连通块异或差值最小分数
2321. 拼接数组的最大分数（https://leetcode.cn/problems/maximum-score-of-spliced-array/）借助经典的最大最小子数组和枚举交换的连续子序列
2306. 公司命名（https://leetcode.cn/problems/naming-a-company/）利用字母个数有限的特点枚举首字母交换对
2272. 最大波动的子字符串（https://leetcode.cn/problems/substring-with-largest-variance/）利用字母个数有限的特点进行最大字符与最小字符枚举
2183. 统计可以被 K 整除的下标对数目（https://leetcode.cn/problems/count-array-pairs-divisible-by-k/）按照最大公约数进行分组枚举
2151. 基于陈述统计最多好人数（https://leetcode.cn/problems/maximum-good-people-based-on-statements/）使用状态压缩进行枚举并判定是否合法
2147. 分隔长廊的方案数（https://leetcode.cn/problems/number-of-ways-to-divide-a-long-corridor/）根据题意进行分隔点枚举计数
2122. 还原原数组（https://leetcode.cn/problems/recover-the-original-array/）枚举差值 k 判断是否合法
2468 根据限制分割消息（https://leetcode.cn/problems/split-message-based-on-limit/）根据长度限制进行二分
2417. 最近的公平整数（https://leetcode.cn/problems/closest-fair-integer/）按照位数贪心枚举加和
2681. 英雄的力量（https://leetcode.cn/problems/power-of-heroes/）按照贡献法枚举计数
1625. 执行操作后字典序最小的字符串（https://leetcode.cn/problems/lexicographically-smallest-string-after-applying-operations/）经典枚举计算最小的字典序
1819. 序列中不同最大公约数的数目（https://leetcode.cn/problems/number-of-different-subsequences-gcds/）经典调和级数枚举
1862. 向下取整数对和（https://leetcode.cn/submissions/detail/371754298/）枚举商并利用调和级数的复杂度进行计算
2014. 重复 K 次的最长子序列（https://leetcode.cn/problems/longest-subsequence-repeated-k-times/）经典利用数据范围进行枚举，贪心使用permutations
2077. 殊途同归（https://leetcode.cn/problems/paths-in-maze-that-lead-to-same-room/）经典使用位运算枚举
2081. k 镜像数字的和（https://leetcode.cn/problems/sum-of-k-mirror-numbers/）经典回文串进制数据枚举
2170. 使数组变成交替数组的最少操作数（https://leetcode.cn/problems/minimum-operations-to-make-the-array-alternating/）经典枚举，运用最大值与次大值技巧
1215. 步进数（https://leetcode.cn/problems/stepping-numbers/）经典根据数据范围使用回溯枚举所有满足条件的数
2245. 转角路径的乘积中最多能有几个尾随零（https://leetcode.cn/problems/maximum-trailing-zeros-in-a-cornered-path/）经典四个方向的前缀和与两两组合枚举
1878. 矩阵中最大的三个菱形和（https://leetcode.cn/problems/get-biggest-three-rhombus-sums-in-a-grid/）经典两个方向上的前缀和计算与边长枚举
2018. 判断单词是否能放入填字游戏内（https://leetcode.cn/problems/check-if-word-can-be-placed-in-crossword/description/）经典枚举空挡位置与矩阵行列取数
2591. 将钱分给最多的儿童（https://leetcode.cn/problems/distribute-money-to-maximum-children/）经典枚举考虑边界条件
910. 最小差值 II（https://leetcode.cn/problems/smallest-range-ii/description/）经典枚举操作的范围，计算最大值与最小值
1131. 绝对值表达式的最大值（https://leetcode.cn/problems/maximum-of-absolute-value-expression/description/）经典曼哈顿距离计算，枚举可能的符号组合
1761. 一个图中连通三元组的最小度数（https://leetcode.cn/problems/minimum-degree-of-a-connected-trio-in-a-graph/description/?envType=daily-question&envId=2023-08-31）经典无向图转为有向图进行枚举
1178. 猜字谜（https://leetcode.cn/problems/number-of-valid-words-for-each-puzzle/）典型哈希计数枚举，使用位运算
1638. 统计只差一个字符的子串数目（https://leetcode.cn/problems/count-substrings-that-differ-by-one-character/description/）枚举子字符串对开头位置也可使用DP枚举
2212. 射箭比赛中的最大得分（https://leetcode.cn/problems/maximum-points-in-an-archery-competition/）位运算枚举或者回溯计算
2749. 得到整数零需要执行的最少操作数（https://leetcode.cn/problems/minimum-operations-to-make-the-integer-zero/）枚举操作次数使用位运算计算可行性
2094. 找出 3 位偶数（https://leetcode.cn/problems/finding-3-digit-even-numbers/description/）脑筋急转弯枚举，有技巧地缩小范围
842. 将数组拆分成斐波那契序列（https://leetcode.cn/problems/split-array-into-fibonacci-sequence/description/）脑筋急转弯枚举数列前两项也可以使用DFS回溯计算
2122. 还原原数组（https://leetcode.cn/problems/recover-the-original-array/）枚举间隔哈希模拟可行性
1782. 统计点对的数目（https://leetcode.cn/problems/count-pairs-of-nodes/description/）使用枚举所有的点对

===================================洛谷===================================
P1548 棋盘问题（https://www.luogu.com.cn/problem/P1548）枚举正方形与长方形的右小角计算个数
P1632 点的移动（https://www.luogu.com.cn/problem/P1632）枚举横坐标和纵坐标的所有组合移动距离
P2128 赤壁之战（https://www.luogu.com.cn/problem/P2128）枚举完全图的顶点组合，平面图最多四个点
P2191 小Z的情书（https://www.luogu.com.cn/problem/P2191）逆向思维旋转矩阵
P2699 【数学1】小浩的幂次运算（https://www.luogu.com.cn/problem/P2699）分类讨论、暴力枚举模拟
P1371 NOI元丹（https://www.luogu.com.cn/problem/P1371）前后缀枚举计数
P1369 矩形（https://www.luogu.com.cn/problem/P1369）矩阵DP与有贪心枚举在矩形边上的最多点个数
P1158 [NOIP2010 普及组] 导弹拦截（https://www.luogu.com.cn/problem/P1158）按照距离的远近进行预处理排序枚举，使用后缀最大值进行更新
P8928 「TERRA-OI R1」你不是神，但你的灵魂依然是我的盛宴（https://www.luogu.com.cn/problem/P8928）枚举取模计数的组合
P8892 「UOI-R1」磁铁（https://www.luogu.com.cn/problem/P8892）枚举字符串的分割点，经典查看一个字符串是否为另一个字符串的子序列
P8799 [蓝桥杯 2022 国 B] 齿轮（https://www.luogu.com.cn/problem/P8799）暴力枚举是否有组合相除得到查询值
P3142 [USACO16OPEN]Field Reduction S（https://www.luogu.com.cn/problem/P3142）暴力枚举临近左右上下边界的点进行最小面积计算
P3143 [USACO16OPEN] Diamond Collector S（https://www.luogu.com.cn/problem/P3143）枚举前缀与后缀序列的最多个数相加
P3670 [USACO17OPEN]Bovine Genomics S（https://www.luogu.com.cn/problem/P3670）哈希枚举计数
P3799 妖梦拼木棒（https://www.luogu.com.cn/problem/P3799）枚举正三角形的木棒边长
P3910 纪念邮票（https://www.luogu.com.cn/problem/P3910）结合因数分解枚举可行的连续数组和为目标数字
P4086 [USACO17DEC]My Cow Ate My Homework S（https://www.luogu.com.cn/problem/P4086）利用后缀进行倒序枚举
P4596 [COCI2011-2012#5] RAZBIBRIGA（https://www.luogu.com.cn/problem/P4596）枚举可行的正方形单词与方案数
P4759 [CERC2014]Sums（https://www.luogu.com.cn/problem/P4759）结合因数分解枚举可行的连续数组和为目标数字
P6267 [SHOI2002]N的连续数拆分（https://www.luogu.com.cn/problem/P6267）结合因数分解枚举可行的连续数组和为目标数字
P5077 Tweetuzki 爱等差数列（https://www.luogu.com.cn/problem/P5077）结合因数分解枚举可行的连续数组和为目标数字
P4960 血小板与凝血因子（https://www.luogu.com.cn/problem/P4960）按照题意模拟枚举
P4994 终于结束的起点（https://www.luogu.com.cn/problem/P4994）暴力模拟，皮萨诺周期可以证明pi(n)<=6n
P5190 [COCI2009-2010#5] PROGRAM（https://www.luogu.com.cn/problem/P5190）使用埃氏筛的思想进行计数与前缀和计算查询，复杂度为调和级数O(nlogn)
P5614 [MtOI2019]膜Siyuan（https://www.luogu.com.cn/problem/P5614）根据题意枚举其中两个数，计算满足条件的另一个数的个数
P6014 [CSGRound3]斗牛（https://www.luogu.com.cn/problem/P6014）使用哈希计算整体取模与每个单个数字确定互补取模计数
P6067 [USACO05JAN]Moo Volume S（https://www.luogu.com.cn/problem/P6067）排序后使用前后缀和进行枚举计算
P6248 准备战斗，选择你的英雄（https://www.luogu.com.cn/problem/P6248）计算可能性进行暴力枚举
P6355 [COCI2007-2008#3] DEJAVU（https://www.luogu.com.cn/problem/P6355）枚举直角三角形的直角点进行计数
P6365 [传智杯 #2 初赛] 众数出现的次数（https://www.luogu.com.cn/problem/P6365）使用容斥原理进行枚举计数
P6439 [COCI2011-2012#6] ZAGRADE（https://www.luogu.com.cn/problem/P6439）枚举删除的位置组合，使用几集合进行去重
P6686 混凝土数学（https://www.luogu.com.cn/problem/P6686）枚举等腰三角形的边长计数
P2666 [USACO07OCT]Bessie's Secret Pasture S（https://www.luogu.com.cn/problem/P2666）枚举计数，计算将n拆解为4个数的平方和的方案数
P2705 小球（https://www.luogu.com.cn/problem/P2705）枚举红色小球放在蓝色盒子的数量计算
P5690 [CSP-S2019 江西] 日期（https://www.luogu.com.cn/problem/P5690）对于日期，典型地暴力进行枚举确认
P7076 [CSP-S2020] 动物园（https://www.luogu.com.cn/problem/P7076）位运算枚举计数
P7094 [yLOI2020] 金陵谣（https://www.luogu.com.cn/problem/P7094）变换公式根据，数据范围进行枚举
P7273 ix35 的等差数列（https://www.luogu.com.cn/problem/P7273）经典公差枚举，计算使得首项相同的个数，贪心选择最佳
P7286 「EZEC-5」人赢（https://www.luogu.com.cn/problem/P7286）排序后枚举最小值，选择最优结果计数
P7626 [COCI2011-2012#1] MATRIX（https://www.luogu.com.cn/problem/P7626）枚举正方形子矩阵的主对角线与副对角线之差
P7799 [COCI2015-2016#6] PIANINO（https://www.luogu.com.cn/problem/P7799）哈希枚举计数
P1018 [NOIP2000 提高组] 乘积最大（https://www.luogu.com.cn/problem/P1018）枚举乘号位置
P1311 [NOIP2011 提高组] 选择客栈（https://www.luogu.com.cn/problem/P1311）线性枚举计数，每次重置避免重复计数
P2119 [NOIP2016 普及组] 魔法阵（https://www.luogu.com.cn/problem/P2119）枚举差值，并计算前后缀个数
P2652 同花顺（https://www.luogu.com.cn/problem/P2652）枚举花色与双指针计算长度
P2994 [USACO10OCT]Dinner Time S（https://www.luogu.com.cn/problem/P2994）按照座位枚举分配人员
P3985 不开心的金明（https://www.luogu.com.cn/problem/P3985）看似背包实则枚举
P4181 [USACO18JAN]Rental Service S（https://www.luogu.com.cn/problem/P4181）贪心枚举与后缀和
P6149 [USACO20FEB] Triangles S（https://www.luogu.com.cn/problem/P6149）经典枚举三角形的直角点使用前缀和与二分计算距离和
P6393 隔离的日子（https://www.luogu.com.cn/problem/P6393）经典利用值域范围进行枚举计算
P6767 [BalticOI 2020/2012 Day0] Roses（https://www.luogu.com.cn/problem/P6767）
P8270 [USACO22OPEN] Subset Equality S（https://www.luogu.com.cn/problem/P8270）经典脑筋急转弯枚举，转换为两两字母比较
P8587 新的家乡（https://www.luogu.com.cn/problem/P8587）桶计数枚举
P8663 [蓝桥杯 2018 省 A] 倍数问题（https://www.luogu.com.cn/problem/P8663）桶计数枚举
P8672 [蓝桥杯 2018 国 C] 交换次数（https://www.luogu.com.cn/problem/P8672）字符串枚举与经典置换环计数
P8712 [蓝桥杯 2020 省 B1] 整数拼接（https://www.luogu.com.cn/problem/P8712）整数长度枚举
P8749 [蓝桥杯 2021 省 B] 杨辉三角形（https://www.luogu.com.cn/problem/P8749）利用杨辉三角形特点进行枚举
P8808 [蓝桥杯 2022 国 C] 斐波那契数组（https://www.luogu.com.cn/problem/P8808）利用斐波那契数组的特点进行枚举
P8809 [蓝桥杯 2022 国 C] 近似 GCD（https://www.luogu.com.cn/problem/P8809）枚举加贡献计数
P9076 [PA2018] PIN（https://www.luogu.com.cn/problem/P9076）根据数字的因数进行枚举
P9008 [入门赛 #9] 大碗宽面 (Hard Version)（https://www.luogu.com.cn/problem/P9008）经典朋友敌人陌生人容斥枚举计数
P9006 [入门赛 #9] 神树大人挥动魔杖 (Hard Version)（https://www.luogu.com.cn/problem/P9006）经典枚举取模计数
P8948 [YsOI2022]NOIp和省选（https://www.luogu.com.cn/problem/P8948）预处理和枚举所有情况
P8894 「UOI-R1」求和（https://www.luogu.com.cn/problem/P8894）按照区间范围值进行枚举前后缀计数
P8872 [传智杯 #5 初赛] D-莲子的物理热力学（https://www.luogu.com.cn/problem/P8872）排序后前后缀移动次数枚举

================================CodeForces================================
https://codeforces.com/problemset/problem/1426/F（分类枚举中间的b计数两边的?ac，并使用快速幂进行求解）
D. Zigzags（https://codeforces.com/problemset/problem/1400/D）枚举+二分
D. Moscow Gorillas（https://codeforces.com/contest/1793/problem/D）枚举计数
D. Dima and Lisa（https://codeforces.com/problemset/problem/584/D）确定一个质数3，枚举第二三个质数，小于 10**9 的任何数都可以分解为最多三个质数的和
D. Three Integers（https://codeforces.com/problemset/problem/1311/D）根据题意，确定一个上限值，贪心枚举
C. Flag（https://codeforces.com/problemset/problem/1181/C）按列进行枚举
B. Maximum Value（https://codeforces.com/problemset/problem/484/B）排序后进行枚举，并使用二分查找进行确认
C. Arithmetic Progression（https://codeforces.com/problemset/problem/382/C）分类讨论
D - Remainder Reminder（https://atcoder.jp/contests/abc090/tasks/arc091_b）典型枚举
D - Katana Thrower（https://atcoder.jp/contests/abc085/tasks/abc085_d）典型枚举
E. Divisibility by 25（https://codeforces.com/contest/988/problem/E）思维题贪心枚举
B. Getting Zero（https://codeforces.com/contest/1661/problem/B）经典枚举

================================AtCoder================================
D - Digit Sum（https://atcoder.jp/contests/abc044/tasks/arc060_b）经典进制计算与分情况枚举因子
D - Menagerie （https://atcoder.jp/contests/abc055/tasks/arc069_b）思维题脑筋急转弯枚举
C - Sequence（https://atcoder.jp/contests/abc059/tasks/arc072_a）枚举前缀和的符号贪心增减
C - Chocolate Bar（https://atcoder.jp/contests/abc062/tasks/arc074_a）枚举切割方式
C - Sugar Water（https://atcoder.jp/contests/abc074/tasks/arc083_a）经典枚举系数利用公式计算边界

================================Acwing===================================
95. 费解的开关（https://www.acwing.com/problem/content/description/97/）枚举第一行的开关按钮使用状态

参考：OI WiKi（xx）
"""
import bisect
import math
from collections import defaultdict, deque
from functools import reduce, lru_cache
from itertools import combinations, permutations
from math import inf
from operator import mul, or_
from typing import List

from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1311d(ac=FastIO()):
        # 模板：根据贪心策略 a=b=1 时显然满足条件，因此枚举不会超过这个代价的范围就行
        for _ in range(ac.read_int()):
            a, b, c = ac.read_list_ints()
            ans = inf
            res = []
            for x in range(1, 2 * a + 1):
                for y in range(x, 2 * b + 1, x):
                    if y % x == 0:
                        for z in [(c // y) * y, (c // y) * y + y]:
                            cost = abs(a - x) + abs(b - y) + abs(c - z)
                            if cost < ans:
                                ans = cost
                                res = [x, y, z]
            ac.st(ans)
            ac.lst(res)
        return

    @staticmethod
    def cf_584d(ac=FastIO()):
        # 模板：将 n 分解为最多三个质数的和
        def is_prime4(x):
            if x == 1:
                return False
            if (x == 2) or (x == 3):
                return True
            if (x % 6 != 1) and (x % 6 != 5):
                return False
            for ii in range(5, int(math.sqrt(x)) + 1, 6):
                if (x % ii == 0) or (x % (ii + 2) == 0):
                    return False
            return True

        # 模板：将正整数分解为最多三个质数的和
        n = ac.read_int()
        assert 3 <= n < 10 ** 9

        if is_prime4(n):
            ac.st(1)
            ac.st(n)
            return

        for i in range(2, 10 ** 5):
            j = n - 3 - i
            if is_prime4(i) and is_prime4(j):
                ac.st(3)
                ac.lst([3, i, j])
                return
        return

    @staticmethod
    def lc_670(num: int) -> int:
        # 模板：在复杂度有限的情况下有限采用枚举的方式计算而不是贪心

        def check():  # 贪心
            lst = list(str(num))
            n = len(lst)
            post = list(range(n))
            # 从后往前遍历，对每个数位，记录其往后最大且最靠后的比它大的数位位置，再从前往后交换第一个有更大的靠后值得数位
            j = n - 1
            for i in range(n - 2, -1, -1):
                if lst[i] > lst[j]:
                    j = i
                if lst[j] > lst[i]:
                    post[i] = j

            for i in range(n):
                if post[i] != i:
                    lst[i], lst[post[i]] = lst[post[i]], lst[i]
                    return int("".join(lst))
            return int("".join(lst))

        def check2():  # 枚举
            lst = list(str(num))
            n = len(lst)
            ans = num
            for item in combinations(list(range(n)), 2):
                cur = lst[:]
                i, j = item
                cur[i], cur[j] = cur[j], cur[i]
                x = int("".join(cur))
                ans = ans if ans > x else x
            return ans

        check()
        return check2()

    @staticmethod
    def cf_484b(ac=FastIO()):
        # 模板：查询数组中两两取模运算的最大值（要求较小值作为取模数）
        ac.read_int()
        nums = sorted(list(set(ac.read_list_ints())))
        n = len(nums)
        ceil = nums[-1]

        dp = [0] * (ceil + 1)
        i = 0
        for x in range(1, ceil + 1):
            dp[x] = dp[x - 1]
            while i < n and nums[i] <= x:
                dp[x] = nums[i]
                i += 1

        ans = 0
        for num in nums:
            x = 1
            while x * num <= ceil:
                x += 1
                for a in [x * num - 1]:
                    ans = ac.max(ans, dp[ac.min(a, ceil)] % num)
        ac.st(ans)
        return

    @staticmethod
    def cf_382c(ac=FastIO()):

        # 2023年3月29日·灵茶试炼·分类讨论
        n = ac.read_int()
        nums = sorted(ac.read_list_ints())

        # 只有一种情况有无穷多个
        if n == 1:
            ac.st(-1)
            return

        # 计算排序后相邻项差值最大值与最小值以及不同差值
        diff = [nums[i] - nums[i - 1] for i in range(1, n)]
        high = max(diff)
        low = min(diff)
        cnt = len(set(diff))

        # 1. 大于等于3个不同差值显然没有
        if cnt >= 3:
            ac.st(0)
            return
        elif cnt == 2:
            # 2. 有2个不同差值存在合理情况当且仅当 high=2*low 且 count(high)==1
            if high != 2 * low or diff.count(high) != 1:
                ac.st(0)
                return

            for i in range(1, n):
                if nums[i] - nums[i - 1] == high:
                    ac.st(1)
                    ac.st(nums[i - 1] + low)
                    return
        else:
            # 3.有1个差值时分为0与不为0，不为0分 n大于2 与等于2
            if low == high == 0:
                ac.st(1)
                ac.st(nums[0])
                return
            if n == 2:
                if low % 2 == 0:
                    ac.st(3)
                    ac.lst([nums[0] - low, nums[0] + low // 2, nums[1] + low])
                else:
                    ac.st(2)
                    ac.lst([nums[0] - low, nums[1] + low])
            else:
                ac.st(2)
                ac.lst([nums[0] - low, nums[-1] + low])
        return

    @staticmethod
    def abc_44d(ac=FastIO()):
        # 模板：经典进制计算与分情况枚举因子
        def check():
            lst = []
            num = n
            while num:
                lst.append(num % b)
                num //= b
            return sum(lst) == s

        n = ac.read_int()
        s = ac.read_int()
        if s > n:
            ac.st(-1)
        elif s == n:
            ac.st(n + 1)
        else:
            # (n-s) % (b-1) == 0
            ans = inf
            for x in range(1, n - s + 1):
                if x * x > n - s:
                    break
                if (n - s) % x == 0:
                    # 枚举 b-1 的值为 n-s 的因子
                    y = (n - s) // x
                    b = x + 1
                    if check():
                        ans = b if ans > b else ans
                    b = y + 1
                    if check():
                        ans = b if ans > b else ans
            ac.st(-1 if ans == inf else ans)
        return

    @staticmethod
    def abc_59c(ac=FastIO()):
        # 模板：枚举前缀和的符号贪心增减
        n = ac.read_int()
        nums = ac.read_list_ints()
        ans1 = 0
        pre = 0
        for i in range(n):
            pre += nums[i]
            if i % 2 == 0:
                if pre <= 0:
                    ans1 += 1 - pre
                    pre = 1
            else:
                if pre >= 0:
                    ans1 += pre + 1
                    pre = -1
        ans2 = 0
        pre = 0
        for i in range(n):
            pre += nums[i]
            if i % 2 == 1:
                if pre <= 0:
                    ans2 += 1 - pre
                    pre = 1
            else:
                if pre >= 0:
                    ans2 += pre + 1
                    pre = -1
        ac.st(ac.min(ans1, ans2))
        return

    @staticmethod
    def abc_62c(ac=FastIO()):
        # 模板：枚举切割方式
        m, n = ac.read_list_ints()

        def check1():
            nonlocal ans
            for x in range(1, m):
                lst = [x * n, (m - x) * (n // 2), (m - x) * (n // 2 + n % 2)]
                cur = max(lst) - min(lst)
                if cur < ans:
                    ans = cur
            return

        def check2():
            nonlocal ans
            for x in range(1, m - 1):
                lst = [x * n, ((m - x) // 2) * n, ((m - x) // 2 + (m - x) % 2) * n]
                cur = max(lst) - min(lst)
                if cur < ans:
                    ans = cur
            return

        ans = inf
        check1()
        check2()
        m, n = n, m
        check1()
        check2()
        ac.st(ans)
        return

    @staticmethod
    def abc_74c(ac=FastIO()):
        # 模板：经典枚举系数利用公式计算边界
        res = 0
        a, b, c, d, e, f = ac.read_list_ints()
        ans = [100 * a, 0]
        for p in range(3001):
            if p * a * 100 > f:
                break
            for q in range(3001):
                if p * a * 100 + q * b * 100 > f:
                    break
                if p == q == 0:
                    continue
                ceil = (p * a + q * b) * e

                for x in range(3001):
                    if x * c > ceil:
                        break
                    y1 = (ceil - x * c) // d
                    y2 = (f - p * a * 100 - q * b * 100 - x * c) // d
                    y1 = y1 if y1 < y2 else y2
                    if y1 < 0:
                        continue
                    y = y1
                    percent = 100 * (x * c + y * d) / (p * a * 100 + q * b * 100 + x * c + y * d)
                    if percent > res:
                        res = percent
                        ans = [p * a * 100 + q * b * 100 + x * c + y * d, x * c + y * d]
        ac.lst(ans)
        return

    @staticmethod
    def ac_95(ac=FastIO()):
        # 模板：枚举第一行状态
        n = ac.read_int()

        for _ in range(n):
            grid = [[int(w) for w in ac.read_str()] for _ in range(5)]
            ac.read_str()

            ans = -1
            for state in range(1 << 5):
                lst = [x for x in range(5) if state & (1 << x)]
                temp = [g[:] for g in grid]
                cur = len(lst)
                for x in lst:
                    i, j = 0, x
                    temp[i][j] = 1 - temp[i][j]
                    for a, b in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                        if 0 <= a < 5 and 0 <= b < 5:
                            temp[a][b] = 1 - temp[a][b]

                for r in range(1, 5):
                    for j in range(5):
                        if temp[r - 1][j] == 0:
                            i, j = r, j
                            temp[i][j] = 1 - temp[i][j]
                            cur += 1
                            for a, b in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                                if 0 <= a < 5 and 0 <= b < 5:
                                    temp[a][b] = 1 - temp[a][b]
                if all(all(x == 1 for x in g) for g in temp):
                    ans = ans if ans < cur and ans != -1 else cur
            ac.st(ans if ans <= 6 else -1)
        return

    @staticmethod
    def lg_p1018(ac=FastIO()):
        # 模板：枚举乘号的位置
        n, k = ac.read_list_ints()
        nums = ac.read_list_str()

        ans = 0
        for item in combinations(list(range(1, n)), k):
            cur = nums[:]
            for i in item:
                cur[i] = "*" + cur[i]
            res = [int(w) for w in ("".join(cur)).split("*")]
            cur = reduce(mul, res)
            ans = ac.max(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1311(ac=FastIO()):
        # 模板：线性枚举计数，每次重置避免重复计数
        n, k, p = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        cnt = [0] * k
        for i in range(n):
            cnt[nums[i][0]] += 1
        pre = [0] * k
        ans = 0
        for i in range(n):
            c = nums[i][0]
            pre[c] += 1
            if nums[i][1] <= p:
                for j in range(k):
                    if j != c:
                        ans += pre[j] * (cnt[j] - pre[j])
                    else:
                        ans += pre[j] - 1
                        ans += cnt[j] - pre[j]
                        ans += (pre[j] - 1) * (cnt[j] - pre[j])
                    cnt[j] -= pre[j]
                pre = [0] * k
        ac.st(ans)
        return

    @staticmethod
    def lg_p2119(ac=FastIO()):

        # 模板：枚举差值，并计算前后缀个数
        n, m = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(m)]

        cnt = [0] * (n + 1)
        for num in nums:
            cnt[num] += 1

        aa = [0] * (n + 1)
        bb = [0] * (n + 1)
        cc = [0] * (n + 1)
        dd = [0] * (n + 1)

        # 枚举b-a=x
        for x in range(1, n // 9 + 1):
            if 1 + 9 * x + 1 > n:
                break

            # 前缀ab计数
            pre_ab = [0] * (n + 1)
            for b in range(2 * x + 1, n + 1):
                pre_ab[b] = pre_ab[b - 1]
                pre_ab[b] += cnt[b] * cnt[b - 2 * x]

            # 作为cd
            for c in range(n - x, -1, -1):
                if c - 6 * x - 1 >= 1:
                    cc[c] += pre_ab[c - 6 * x - 1] * cnt[c + x]
                    dd[c + x] += pre_ab[c - 6 * x - 1] * cnt[c]
                else:
                    break

            # 后缀cd
            post_cd = [0] * (n + 2)
            for c in range(n - x, -1, -1):
                post_cd[c] = post_cd[c + 1]
                post_cd[c] += cnt[c] * cnt[c + x]

            # 作为ab计数
            for b in range(2 * x + 1, n + 1):
                if b + 6 * x + 1 <= n:
                    aa[b - 2 * x] += post_cd[b + 6 * x + 1] * cnt[b]
                    bb[b] += post_cd[b + 6 * x + 1] * cnt[b - 2 * x]
                else:
                    break

        for x in nums:
            ac.lst([aa[x], bb[x], cc[x], dd[x]])
        return

    @staticmethod
    def lg_p2652(ac=FastIO()):

        # 模板：枚举花色与双指针计算长度
        n = ac.read_int()
        dct = defaultdict(set)
        for _ in range(n):
            a, b = ac.read_list_ints()
            dct[a].add(b)
        ans = n
        for a in dct:
            lst = sorted(list(dct[a]))
            m = len(lst)
            j = 0
            for i in range(m):
                while j < m and lst[j] - lst[i] <= n - 1:
                    j += 1
                ans = ac.min(ans, n - (j - i))
        ac.st(ans)
        return

    @staticmethod
    def lg_p2994(ac=FastIO()):

        # 模板：按照座位枚举分配人员
        def dis():
            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        n, m = ac.read_list_ints()
        cow = [ac.read_list_ints() for _ in range(n)]
        pos = [ac.read_list_ints() for _ in range(n)]
        visit = [0] * n
        for j in range(m):
            ceil = inf
            ind = 0
            x1, y1 = pos[j]
            for i in range(n):
                if visit[i]:
                    continue
                x2, y2 = cow[i]
                cur = dis()
                if cur < ceil:
                    ceil = cur
                    ind = i
            if ceil < inf:
                visit[ind] = 1
        ans = [i + 1 for i in range(n) if not visit[i]]
        if not ans:
            ac.st(0)
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lg_p4181(ac=FastIO()):
        # 模板：贪心枚举与后缀和
        n, m, r = ac.read_list_ints()
        cow = [ac.read_int() for _ in range(n)]
        cow.sort()
        nums1 = [ac.read_list_ints()[::-1] for _ in range(m)]
        nums1.sort(key=lambda it: -it[0])
        nums2 = [ac.read_int() for _ in range(r)]
        nums2.sort(reverse=True)
        # 预处理后缀和
        ind = 0
        post = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            cur = 0
            while ind < m and cow[i]:
                if nums1[ind][1] == 0:
                    ind += 1
                    continue
                x = ac.min(nums1[ind][1], cow[i])
                cow[i] -= x
                nums1[ind][1] -= x
                cur += nums1[ind][0] * x
            post[i] = post[i + 1] + cur
        # 枚举
        ans = post[0]
        pre = 0
        for i in range(ac.min(r, n)):
            pre += nums2[i]
            ans = ac.max(ans, pre + post[i + 1])
        ac.st(ans)
        return

    @staticmethod
    def lg_p6149(ac=FastIO()):
        # 模板：经典枚举三角形的直角点使用前缀和与二分计算距离和
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        dct_x = defaultdict(list)
        dct_y = defaultdict(list)
        for x, y in nums:
            dct_x[x].append(y)
            dct_y[y].append(x)
        pre_x = defaultdict(list)
        for x in dct_x:
            dct_x[x].sort()
            pre_x[x] = ac.accumulate(dct_x[x])
        pre_y = defaultdict(list)
        for y in dct_y:
            dct_y[y].sort()
            pre_y[y] = ac.accumulate(dct_y[y])

        ans = 0
        mod = 10 ** 9 + 7
        for x, y in nums:
            # 二分找到中间点 xi 计算两侧距离
            xi = bisect.bisect_left(dct_y[y], x)
            left_x = (xi + 1) * x - pre_y[y][xi + 1]
            right_x = pre_y[y][-1] - pre_y[y][xi + 1] - (len(dct_y[y]) - xi - 1) * x

            yi = bisect.bisect_left(dct_x[x], y)
            left_y = (yi + 1) * y - pre_x[x][yi + 1]
            right_y = pre_x[x][-1] - pre_x[x][yi + 1] - (len(dct_x[x]) - yi - 1) * y
            ans += (left_x + right_x) * (left_y + right_y)
            ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def lg_p6393(ac=FastIO()):
        # 模板：经典利用值域范围进行枚举计算
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        dct = dict()
        for i in range(n):
            a, b = nums[i]
            if b not in dct:
                dct[b] = dict()
            if a not in dct[b]:
                dct[b][a] = deque()
            dct[b][a].append(i)
        for i in range(n):
            a, b = nums[i]
            ind = -2
            for bb in dct:
                if (b * b) % bb == 0:
                    # 寻找符合条件的最小值
                    aa = a + b * b // bb + b
                    if aa in dct[bb]:
                        while dct[bb][aa] and dct[bb][aa][0] <= i:
                            dct[bb][aa].popleft()
                        if dct[bb][aa]:
                            j = dct[bb][aa][0]
                            if ind == -2 or j < ind:
                                ind = j
                        else:
                            del dct[bb][aa]
                            if not dct[bb]:
                                del dct[bb]
            ac.st(ind + 1)
        return

    @staticmethod
    def lc_2591(money: int, children: int) -> int:
        # 模板：经典枚举考虑边界条件
        ans = -1
        for x in range(children + 1):
            if x * 8 > money:
                break
            rest_money = money - x * 8
            rest_people = children - x
            if rest_money < rest_people:
                continue
            if not rest_people and rest_money:
                continue
            if rest_people == 1 and rest_money == 4:
                continue
            ans = x
        return ans

    @staticmethod
    def lc_2681(nums: List[int]) -> int:
        # 模板：按照贡献法枚举计数
        mod = 10 ** 9 + 7
        nums.sort()
        ans = pre = 0
        for num in nums:
            ans += num * num * pre
            ans += num * num * num
            pre %= mod
            ans %= mod
            pre *= 2
            pre += num
        return ans

    @staticmethod
    def lg_p6767(ac=FastIO()):
        # 模板：贪心枚举性价比较低的数量
        n, a, b, c, d = ac.read_list_ints()
        if b * c > a * d:
            a, b, c, d = c, d, a, b

        ans = inf
        for x in range(10 ** 5 + 1):
            cur = x * d + b * ac.max(math.ceil((n - x * c) / a), 0)
            ans = ac.min(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p8270(ac=FastIO()):
        # 模板：经典脑筋急转弯枚举，转换为两两字母比较
        s = ac.read_str()
        t = ac.read_str()
        lst = sorted(list("abcdefghijklmnopqr"))
        m = len(lst)
        pre = set()
        for i in range(m):
            for j in range(i, m):
                cur = {lst[i], lst[j]}
                ss = ""
                tt = ""
                for w in s:
                    if w in cur:
                        ss += w
                for w in t:
                    if w in cur:
                        tt += w
                if ss == tt:
                    pre.add(lst[i] + lst[j])
                    pre.add(lst[j] + lst[i])
        ans = ""
        for _ in range(ac.read_int()):
            st = ac.read_str()
            m = len(st)
            flag = True
            for i in range(m):
                for j in range(i, m):
                    if st[i] + st[j] not in pre:
                        flag = False
                        break
                if not flag:
                    break
            ans += "Y" if flag else "N"
        ac.st(ans)
        return

    @staticmethod
    def lg_p8672(ac=FastIO()):
        # 模板：字符串枚举与经典置换环计数
        s = ac.read_str()
        n = len(s)
        dct = dict()
        dct["B"] = s.count("B")
        dct["A"] = s.count("A")
        dct["T"] = s.count("T")
        ans = inf
        for item in permutations("BAT", 3):
            t = ""
            for w in item:
                t += dct[w] * w
            cnt = defaultdict(int)
            for i in range(n):
                if s[i] != t[i]:
                    cnt[s[i] + t[i]] += 1
            cur = 0
            for w in item:
                for p in item:
                    if w != p:
                        x = ac.min(cnt[w + p], cnt[p + w])
                        cur += x
                        cnt[w + p] -= x
                        cnt[p + w] -= x
            rest = sum(cnt.values())
            cur += rest * 2 // 3
            ans = ac.min(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p9076(ac=FastIO()):
        # 模板：根据数字的因数进行枚举
        n = ac.read_int()
        ans = 0
        pre = set()
        for a in range(1, int(n ** 0.5) + 1):
            if n % a == 0:
                for bc in [n // a - 1, a - 1]:
                    if bc in pre:
                        continue
                    pre.add(bc)
                    for x in range(2, bc + 1):
                        if bc % x == 0:
                            y = bc // x - 1
                            if y > 1:
                                ans += 1
                        if bc // x <= 2:
                            break
        ac.st(ans)
        return

    @staticmethod
    def lg_p9008(ac=FastIO()):
        # 模板：经典朋友敌人陌生人容斥枚举计数
        n, p, q = ac.read_list_ints()
        friend = defaultdict(set)
        for _ in range(p):
            u, v = ac.read_list_ints()
            friend[u].add(v)
            friend[v].add(u)
        ans = n * (n - 1) // 2
        rem = set()
        for _ in range(q):
            u, v = ac.read_list_ints()
            rem.add((u, v) if u < v else (v, u))
            for x in friend[u]:
                if x not in friend[v]:
                    rem.add((x, v) if x < v else (v, x))
            for y in friend[v]:
                if y not in friend[u]:
                    rem.add((y, u) if y < u else (u, y))
        ac.st(ans - len(rem))
        return

    @staticmethod
    def lg_p9006(ac=FastIO()):
        # 模板：经典枚举取模计数
        mod = 100000007
        n, k = ac.read_list_ints()
        num = 9 * 10 ** (n - 1)
        x = num // k
        x %= mod
        ans = [x] * k
        for y in range(10 ** (n - 1) + x * k, 10 ** (n - 1) + x * k + num % k):
            ans[y % k] += 1
        ac.lst([x % mod for x in ans])
        return

    @staticmethod
    def lg_p8948(ac=FastIO()):
        # 模板：预处理和枚举所有情况
        dct = dict()
        dct[2000] = [400, 600]
        for i in range(401):
            for j in range(601):
                x = (3 * i + 2 * j) * 10 / 12
                x = int(x) + int(x - int(x) >= 0.5)
                if 10 <= x <= 1990:
                    dct[x] = [i, j]
        for _ in range(ac.read_int()):
            ac.lst(dct[ac.read_int()])
        return

    @staticmethod
    def lg_p8894(ac=FastIO()):
        # 模板：按照区间范围值进行枚举前后缀计数
        n = ac.read_int()
        mod = 998244353
        nums = [ac.read_list_ints() for _ in range(n)]
        ceil = max(q for _, q in nums)
        low = min(p for p, _ in nums)
        ans = 0
        for s in range(low, ceil + 1):
            pre = [0] * (n + 1)
            pre[0] = 1
            for i in range(n):
                p, q = nums[i]
                if p > s:
                    pre[i + 1] = 0
                    break
                else:
                    pre[i + 1] = pre[i] * (ac.min(s, q) - p + 1) % mod

            post = [0] * (n + 1)
            post[n] = 1
            for i in range(n - 1, -1, -1):
                p, q = nums[i]
                if p >= s:
                    post[i] = 0
                    break
                else:
                    post[i] = post[i + 1] * (ac.min(q, s - 1) - p + 1) % mod
            for i in range(n):
                p, q = nums[i]
                if p <= s <= q:
                    ans += pre[i] * post[i + 1] * s
                    ans %= mod
                if pre[i + 1] == 0:
                    break
        ac.st(ans)
        return

    @staticmethod
    def lg_p8872(ac=FastIO()):
        # 模板：排序后前后缀移动次数枚举
        n, m = ac.read_list_ints()
        nums = sorted(ac.read_list_ints())
        ans = inf
        for i in range(n):
            if i > m:
                break
            right = (m - i) // 2
            if right >= n - i - 1:
                ac.st(0)
                return
            cur = nums[-right - 1] - nums[i]
            ans = ac.min(ans, cur)

        for i in range(n - 1, -1, -1):
            if n - i - 1 > m:
                break
            left = (m - n + i + 1) // 2
            if left >= i:
                ac.st(0)
                return
            cur = nums[i] - nums[left]
            ans = ac.min(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lc_2018(board: List[List[str]], word: str) -> bool:
        # 模板：经典枚举空挡位置与矩阵行列取数
        k = len(word)

        def check(cur):
            if len(cur) != len(word):
                return False
            return all(cur[i] == " " or cur[i] == word[i] for i in range(k))

        def compute(lst):
            length = len(lst)
            pre = 0
            for i in range(length):
                if lst[i] == "#":
                    if check([lst[x] for x in range(pre, i)]):
                        return True
                    pre = i + 1
            if check([lst[x] for x in range(pre, length)]):
                return True
            return False

        for tmp in board:
            if compute(tmp[:]) or compute(tmp[::-1]):
                return True

        for tmp in zip(*board):
            if compute(tmp[:]) or compute(tmp[::-1]):
                return True
        return False

    @staticmethod
    def lc_2170(nums: List[int]) -> int:
        # 模板：经典枚举，运用最大值与次大值技巧
        odd = defaultdict(int)
        even = defaultdict(int)
        n = len(nums)
        odd_cnt = 0
        even_cnt = 0
        for i in range(n):
            if i % 2 == 0:
                even[nums[i]] += 1
                even_cnt += 1
            else:
                odd[nums[i]] += 1
                odd_cnt += 1

        # 最大值与次大值计算
        a = b = 0
        for num in even:
            if even[num] >= a:
                a, b = even[num], a
            elif even[num] >= b:
                b = even[num]

        # 枚举奇数位置的数
        ans = odd_cnt + even_cnt - a
        for num in odd:
            cur = odd_cnt - odd[num]
            if even[num] == a:
                x = b
            else:
                x = a
            cur += even_cnt - x
            if cur < ans:
                ans = cur

        return ans

    @staticmethod
    def lc_910(nums: List[int], k: int) -> int:
        # 模板：经典枚举操作的范围，计算最大值与最小值
        nums.sort()
        ans = nums[-1] - nums[0]
        n = len(nums)
        for i in range(n - 1):
            a, b = nums[n - 1] - k, nums[i] + k
            a = a if a > b else b
            c, d = nums[0] + k, nums[i + 1] - k
            c = c if c < d else d
            if a - c < ans:
                ans = a - c
        return ans

    @staticmethod
    def lc_1178(words: List[str], puzzles: List[str]) -> List[int]:
        # 模板：典型哈希计数枚举，使用位运算
        dct = defaultdict(int)
        for word in words:
            cur = set(word)
            lst = [ord(w) - ord("a") for w in cur]
            state = reduce(or_, [1 << x for x in lst])
            if len(cur) <= 7:
                dct[state] += 1
        ans = []
        for word in puzzles:
            lst = [ord(w) - ord("a") for w in word]
            n = len(lst)
            cur = 0
            for i in range(1 << (n - 1)):
                i *= 2
                i += 1
                s = sum(1 << lst[j] for j in range(n) if i & (1 << j))
                cur += dct[s]
            ans.append(cur)
        return ans

    @staticmethod
    def lc_1215(low: int, high: int) -> List[int]:

        # 模板：经典根据数据范围使用回溯枚举所有满足条件的数

        def dfs():
            nonlocal num, ceil
            if num > ceil:
                return
            ans.append(num)
            last = num % 10
            for x in [last - 1, last + 1]:
                if 0 <= x <= 9:
                    num = num * 10 + x
                    dfs()
                    num //= 10
            return

        ceil = 2 * 10 ** 9
        ans = [0]
        for i in range(1, 10):
            num = i
            dfs()
        ans.sort()
        i, j = bisect.bisect_left(ans, low), bisect.bisect_right(ans, high)
        return ans[i:j]

    @staticmethod
    def lc_1131(arr1: List[int], arr2: List[int]) -> int:
        # 模板：经典曼哈顿距离计算，枚举可能的符号组合
        n = len(arr1)
        ans = 0
        for x in [1, -1]:
            for y in [1, -1]:
                for z in [1, -1]:
                    a1 = max(x * arr1[i] + y * arr2[i] + z * i for i in range(n))
                    a2 = min(x * arr1[i] + y * arr2[i] + z * i for i in range(n))
                    if a1 - a2 > ans:
                        ans = a1 - a2
        return ans

    @staticmethod
    def lc_1638_1(s: str, t: str) -> int:
        # 模板：枚举子字符串对开头位置也可使用DP枚举
        m, n = len(s), len(t)
        ans = 0
        for i in range(m):
            for j in range(n):
                cur = int(s[i] != t[j])
                x, y = i, j
                while cur <= 1 and x < m and y < n:
                    ans += cur == 1
                    x += 1
                    y += 1
                    if x == m or y == n:
                        break
                    cur += int(s[x] != t[y])
        return ans

    @staticmethod
    def lc_1638_2(s: str, t: str) -> int:
        # 模板：枚举子字符串对开头位置也可使用DP枚举
        m = len(s)
        n = len(t)
        cnt = [[0] * (n + 1) for _ in range(m + 1)]
        same = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if s[i] == t[j]:
                    same[i + 1][j + 1] = same[i][j] + 1  # 以i,j为结尾的最长连续子串长度
                    cnt[i + 1][j + 1] = cnt[i][j]  # 以i,j为结尾的子串对数
                else:
                    same[i + 1][j + 1] = 0  # 转移可以使用对角线方向转移则只需要O(1)空间
                    cnt[i + 1][j + 1] = same[i][j] + 1
        return sum(sum(d) for d in cnt)

    @staticmethod
    def lc_1761(n: int, edges: List[List[int]]) -> int:
        # 模板：经典无向图转为有向图进行枚举
        edges = [[i - 1, j - 1] for i, j in edges]
        degree = [0] * n
        dct = [set() for _ in range(n)]
        directed = [set() for _ in range(n)]
        for i, j in edges:
            dct[i].add(j)
            degree[i] += 1
            degree[j] += 1
            dct[j].add(i)
        for i, j in edges:
            if degree[i] < degree[j] or (degree[i] == degree[j] and i < j):
                directed[i].add(j)
            else:
                directed[j].add(i)
        ans = inf
        for i in range(n):
            for j in directed[i]:
                for k in directed[j]:
                    if k in dct[i]:
                        x = degree[i] + degree[j] + degree[k] - 6
                        if x < ans:
                            ans = x
        return ans if ans < inf else -1

    @staticmethod
    def lc_1878(grid: List[List[int]]) -> List[int]:
        # 模板：经典两个方向上的前缀和计算与边长枚举

        m, n = len(grid), len(grid[0])

        @lru_cache(None)
        def left_up(p, q):

            if p < 0 or q < 0:
                return 0
            res = grid[p][q]
            if p and q:
                res += left_up(p - 1, q - 1)

            return res

        @lru_cache(None)
        def right_up(p, q):
            if p < 0 or q < 0:
                return 0
            res = grid[p][q]
            if p and q + 1 < n:
                res += right_up(p - 1, q + 1)

            return res

        ans = set()
        k = max(m, n)
        for i in range(m):
            for j in range(n):
                ans.add(grid[i][j])

                for x in range(1, k + 1):
                    up_point = [i - x, j]
                    down_point = [i + x, j]
                    left_point = [i, j - x]
                    right_point = [i, j + x]
                    if not all(0 <= a < m and 0 <= b < n for a, b in [up_point, down_point, left_point, right_point]):
                        break
                    cur = left_up(right_point[0], right_point[1]) - left_up(up_point[0], up_point[1])
                    cur += left_up(down_point[0], down_point[1]) - left_up(left_point[0], left_point[1])

                    cur += right_up(left_point[0], left_point[1]) - right_up(up_point[0], up_point[1])
                    cur += right_up(down_point[0], down_point[1]) - right_up(right_point[0], right_point[1])
                    cur -= grid[down_point[0]][down_point[1]]
                    cur += grid[up_point[0]][up_point[1]]
                    ans.add(cur)
        ans = list(ans)
        ans.sort(reverse=True)
        return ans[:3]

    @staticmethod
    def lc_2212(x: int, y: List[int]) -> List[int]:
        # 模板：位运算枚举或者回溯计算
        n = len(y)
        ans = [0] * n
        ans[0] = x
        res = 0
        for i in range(1 << n):
            lst = [0] * n
            cur = 0
            for j in range(n):
                if i & (1 << j):
                    lst[j] = y[j] + 1
                    cur += j
            s = sum(lst)
            if s <= x:
                lst[0] += x - s
                if cur > res:
                    res = cur
                    ans = lst[:]
        return ans

    @staticmethod
    def lc_2245(grid: List[List[int]]) -> int:
        # 模板：经典四个方向的前缀和与两两组合枚举

        def check(num, f):
            res = 0
            while num % f == 0:
                res += 1
                num //= f
            return res

        m, n = len(grid), len(grid[0])

        cnt = [[[check(grid[i][j], 2), check(grid[i][j], 5)] for j in range(n)] for i in range(m)]

        @lru_cache(None)
        def up(i, j):
            cur = cnt[i][j][:]
            if i:
                nex = up(i - 1, j)
                cur[0] += nex[0]
                cur[1] += nex[1]
            return cur

        @lru_cache(None)
        def down(i, j):
            cur = cnt[i][j][:]
            if i + 1 < m:
                nex = down(i + 1, j)
                cur[0] += nex[0]
                cur[1] += nex[1]
            return cur

        @lru_cache(None)
        def left(i, j):
            cur = cnt[i][j][:]
            if j:
                nex = left(i, j - 1)
                cur[0] += nex[0]
                cur[1] += nex[1]
            return cur

        @lru_cache(None)
        def right(i, j):
            cur = cnt[i][j][:]
            if j + 1 < n:
                nex = right(i, j + 1)
                cur[0] += nex[0]
                cur[1] += nex[1]
            return cur

        ans = 0
        for i in range(m):
            for j in range(n):
                lst = [up(i, j), down(i, j), left(i, j), right(i, j)]
                for ls in lst:
                    x = ls[0] if ls[0] < ls[1] else ls[1]
                    if x > ans:
                        ans = x
                tmp = cnt[i][j]
                for item in combinations(lst, 2):
                    ls1, ls2 = item
                    x = ls1[0] + ls2[0] - tmp[0] if ls1[0] + ls2[0] - tmp[0] < ls1[1] + ls2[1] - tmp[1] \
                        else ls1[1] + ls2[1] - tmp[1]
                    if x > ans:
                        ans = x
        return ans
