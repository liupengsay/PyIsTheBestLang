"""

Algorithm：greedy、reverse_thinking、抽屉原理、pigeonhole、inclusion_exclusion、define_sort、思维、brain_teaser、construction
Function：各种可证明不可证明的头脑风暴

====================================LeetCode====================================
134（https://leetcode.com/problems/gas-station/）greedy
330（https://leetcode.com/problems/patching-array/）greedy
1199（https://leetcode.com/problems/minimum-time-to-build-blocks/）哈夫曼树Huffman Treegreedy从小到大合并，类似合并果子
2499（https://leetcode.com/problems/minimum-total-cost-to-make-arrays-unequal/）利用pigeonholegreedy最小代价
2449（https://leetcode.com/problems/minimum-total-cost-to-make-arrays-unequal/）转换题意sorting后用奇偶数greedy变换得到
2448（https://leetcode.com/problems/minimum-cost-to-make-array-equal/）利用median的特点变换到带权重广义下median的位置是最优的greedy增减
2412（https://leetcode.com/problems/minimum-money-required-before-transactions/）根据交易增长特点define_sort
2366（https://leetcode.com/problems/minimum-replacements-to-sort-the-array/）reverse_order|greedy不断分解得到满足要求且尽可能大的值
2350（https://leetcode.com/problems/shortest-impossible-sequence-of-rolls/）brain_teaser本质上是求全排列出现的轮数
2344（https://leetcode.com/problems/minimum-deletions-to-make-array-divisible/）利用最大公约数greedy删除最少的元素
2136（https://leetcode.com/problems/earliest-possible-day-of-full-bloom/）greedy安排成长时间最长的先种
2071（https://leetcode.com/problems/maximum-number-of-tasks-you-can-assign/）greedy|binary_search极值判断
517（https://leetcode.com/problems/super-washing-machines/）类似上题，最小的左右移动次数以及往左右的移动次数
1798（https://leetcode.com/problems/maximum-number-of-consecutive-values-you-can-make/）看似背包实则greedy
625（https://leetcode.com/problems/minimum-factorization/）greedy因式分解，类似质factorization|
2568（https://leetcode.com/problems/minimum-impossible-or/）brain_teasergreedy，可以根据打表观察规律
6361（https://leetcode.com/problems/minimum-score-by-changing-two-elements/）brain_teasergreedy
6316（https://leetcode.com/contest/weekly-contest-336/problems/rearrange-array-to-maximize-prefix-score/）greedy，|prefix_sum
2436（https://leetcode.com/problems/minimum-split-into-subarrays-with-gcd-greater-than-one/）greedy
1029（https://leetcode.com/problems/two-city-scheduling/）greedy题目，可举例两个、再归纳确定sorting规则
1353（https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended/）brute_forcegreedy
1402（https://leetcode.com/problems/reducing-dishes/）prefix_sumgreedy
1665（https://leetcode.com/problems/minimum-initial-energy-to-finish-tasks/）greedy不同项比较公式sortingimplemention，同CF1203F
1675（https://leetcode.com/problems/minimize-deviation-in-array/）brain_teaserbrain_teaser|greedy
1686（https://leetcode.com/problems/stone-game-vi/）greedy采用列式子确定sorting方式
1808（https://leetcode.com/problems/maximize-number-of-nice-divisors/）按照模3的因子个数greedy处理，将和拆分成最大乘积
1953（https://leetcode.com/problems/maximum-number-of-weeks-for-which-you-can-work/）greedy只看最大值的影响
2856（https://leetcode.com/problems/minimum-array-length-after-pair-removals/）greedy只看最大值的影响
858（https://leetcode.com/problems/mirror-reflection/description/）brain_teaserbrain_teaser|
1927（https://leetcode.com/problems/sum-game/description/）博弈brain_teaser|classification_discussion
2592（https://leetcode.com/problems/maximize-greatness-of-an-array/）典型greedysorting后two_pointers
1503（https://leetcode.com/problems/last-moment-before-all-ants-fall-out-of-a-plank/）brain_teaser题目，相撞不影响结果
991（https://leetcode.com/problems/broken-calculator/）逆向greedy，偶数除2奇数|1
2745（https://leetcode.com/problems/construct-the-longest-new-string/）brain_teasergreedybrain_teaser|
1657（https://leetcode.com/problems/determine-if-two-strings-are-close/description/）brain_teasergreedybrain_teaser|
2561（https://leetcode.com/problems/rearranging-fruits/）brain_teaser|greedy交换
843（https://leetcode.com/problems/guess-the-word/）brain_teaser|greedyimplemention交互
1946（https://leetcode.com/problems/largest-number-after-mutating-substring/description/）易错greedy
1540（https://leetcode.com/problems/can-convert-string-in-k-moves/）greedybrain_teaser|，pointer记录
1121（https://leetcode.com/problems/divide-array-into-increasing-sequences/description/）brain_teaser|greedy，只考虑最大值的分组影响

=====================================LuoGu======================================
1031（https://www.luogu.com.cn/problem/P1031）greedy每个点的prefix_sum流量，需要补齐或者输出时counter
1684（https://www.luogu.com.cn/problem/P1684）线性greedy满足条件即增|counter
1658（https://www.luogu.com.cn/problem/P1658）看似背包实则greedy
2001（https://www.luogu.com.cn/problem/P2001）看似背包实则greedy
1620（https://www.luogu.com.cn/problem/P1620）classification_discussiongreedy
2773（https://www.luogu.com.cn/problem/P2773）classification_discussiongreedy
2255（https://www.luogu.com.cn/problem/P2255）两个pointergreedy
2327（https://www.luogu.com.cn/problem/P2327）brain_teaserbrute_force
2777（https://www.luogu.com.cn/problem/P2777）greedybrute_force最佳得分组合，|prefix_suffix记录最大值
2649（https://www.luogu.com.cn/problem/P2649）greedy，输的时候输最惨，赢的时候微弱优势
1367（https://www.luogu.com.cn/problem/P1367）brain_teaser，蚂蚁的相对移动位置sorting还是不变
1362（https://www.luogu.com.cn/problem/P1362）找规律之后，广度优先搜索brute_force
1090（https://www.luogu.com.cn/record/list?user=739032&status=12&page=11）从小到大greedy合并
1334（https://www.luogu.com.cn/problem/P1334）reverse_thinking的合并果子，从小到大合并
1325（https://www.luogu.com.cn/problem/P1325）sorting后greedy修建更新
1250（https://www.luogu.com.cn/problem/P1250）区间的greedy题，segment_tree|修改区间与查询和，以及binary_search
1230（https://www.luogu.com.cn/problem/P1230）sorting后选取greedy
1159（https://www.luogu.com.cn/problem/P1159）队列greedyimplemention
1095（https://www.luogu.com.cn/problem/P1095）greedyimplemention也可以理解为动态规划转移
1056（https://www.luogu.com.cn/record/list?user=739032&status=12&page=14）根据题意countersortinggreedy选择
8847（https://www.luogu.com.cn/problem/P8847）classification_discussion和greedy
8845（https://www.luogu.com.cn/problem/solution/P8845）brain_teaser，只有2是偶数质数
2772（https://www.luogu.com.cn/problem/P2772）按照两个维度sorting，再按照其中一个维度顺序比较最大值
2878（https://www.luogu.com.cn/problem/P2878）greedy题目，可举例两个、再归纳确定sorting规则
2920（https://www.luogu.com.cn/problem/P2920）sorting后greedy
2983（https://www.luogu.com.cn/problem/P2983）看起来是背包其实是greedy优先选择最便宜的奶牛满足
3173（https://www.luogu.com.cn/problem/P3173）从大到小sortinggreedy
5098（https://www.luogu.com.cn/problem/P5098）greedy按照一个维度sorting后再按照另一个维度classification_discussion，记录前缀最小值
5159（https://www.luogu.com.cn/problem/P5159）利用异或的特点brute_forcecounter并fast_power|
5497（https://www.luogu.com.cn/problem/P5497）抽屉原理classification_discussion
5682（https://www.luogu.com.cn/problem/P5682）brain_teasersorting后greedybrute_force确定
5804（https://www.luogu.com.cn/problem/P5804）sortinggreedybrute_force和binary_search优化
5963（https://www.luogu.com.cn/problem/P5963）greedy题目，可举例两个、再归纳确定sorting规则
6023（https://www.luogu.com.cn/problem/P6023）可证明集中在某天是最佳结果，然后pointerimplemention
6243（https://www.luogu.com.cn/problem/P6243）greedy举例之后优先级比较，再define_sort
6179（https://www.luogu.com.cn/problem/list?difficulty=3&page=13）greedy
6380（https://www.luogu.com.cn/problem/P6380）greedyimplemention赋值
6446（https://www.luogu.com.cn/problem/P6446）greedy操作，使得数组所有值相等的最少操作次数变形题目，每次操作可以使得连续区间|1或者减1
5019（https://www.luogu.com.cn/problem/P5019）greedy操作，使得数组所有值相等的最少操作次数变形题目，每次操作可以使得连续区间|1或者减1
6462（https://www.luogu.com.cn/problem/P6462）greedyclassification_discussion|
6549（https://www.luogu.com.cn/problem/P6549）reverse_thinking，插入sorting的思想implemention
6785（https://www.luogu.com.cn/problem/P6785）brain_teaser，条件判断与greedycounter
6851（https://www.luogu.com.cn/problem/P6851）greedyimplemention，均从大到小sorting，先选择赢的牌，再输的牌
7176（https://www.luogu.com.cn/problem/P7176）greedy策略，结论题
7228（https://www.luogu.com.cn/problem/P7228）brain_teasergreedy|树形dfs
7260（https://www.luogu.com.cn/problem/P7260）greedy与动态规划，使得数组所有值从0变化等于给定升序数组的最少操作次数，每次操作可以使得连续区间|1
7319（https://www.luogu.com.cn/problem/P7319）公式变形后sorting不等式greedy
7412（https://www.luogu.com.cn/problem/P7412）greedy，将问题转换为去掉最长的k-1个非零距离
7522（https://www.luogu.com.cn/problem/P7522）classification_discussion|greedy讨论
7633（https://www.luogu.com.cn/problem/P7633）埃氏筛法思想，implementiongreedy
7714（https://www.luogu.com.cn/problem/P7714）子序列sorting使得整体有序，前缀最大值与pointercounter确认子数组分割点
7787（https://www.luogu.com.cn/problem/P7787）brain_teaser，借助完全二叉树的思想
7813（https://www.luogu.com.cn/problem/P7813）greedy最大选取值
1031（https://www.luogu.com.cn/problem/P1031）线性均分纸牌问题
2512（https://www.luogu.com.cn/problem/P2512）线性环形均分纸牌问题
1080（https://www.luogu.com.cn/problem/P1080）greedy，举例两项确定sorting公式
1650（https://www.luogu.com.cn/problem/P1650）greedy，优先上对上其次下对下最后下对上
2088（https://www.luogu.com.cn/problem/P2088）greedy，取空闲的，或者下一个离得最远的
2816（https://www.luogu.com.cn/problem/P2816）sorting后从小到大greedy放置，STL维护当前积木列高度
3819（https://www.luogu.com.cn/problem/P3819）mediangreedy题
3918（https://www.luogu.com.cn/problem/P3918）brain_teasergreedy
4025（https://www.luogu.com.cn/problem/P4025）greedy血量与增幅define_sort
4266（https://www.luogu.com.cn/problem/P4266）后缀最大值greedyimplemention
4447（https://www.luogu.com.cn/problem/P4447）greedy队列使得连续值序列最少的分组长度最大
4575（https://www.luogu.com.cn/problem/P4575）brain_teaser|状压运算
4653（https://www.luogu.com.cn/problem/P4653）看似binary_searchpointergreedy选取
5093（https://www.luogu.com.cn/problem/P5093）brain_teaser集合确定轮数
5425（https://www.luogu.com.cn/problem/P5425）看似最小生成树，实则brain_teasergreedy距离
5884（https://www.luogu.com.cn/problem/P5884）brain_teaser
5948（https://www.luogu.com.cn/problem/P5948）greedyimplemention
6196（https://www.luogu.com.cn/problem/P6196）greedy 1 分段代价
6874（https://www.luogu.com.cn/problem/P6874）变换公式转为mediangreedy
8050（https://www.luogu.com.cn/problem/P8050）brain_teaser黑白染色法任意操作不改变黑白元素和的差值
7935（https://www.luogu.com.cn/problem/P7935）brain_teaser
8109（https://www.luogu.com.cn/problem/P8109）STLgreedy分配求解
8669（https://www.luogu.com.cn/problem/P8669）greedy选取 k 个数乘积最大
8709（https://www.luogu.com.cn/problem/P8709）brain_teaserimplemention
8732（https://www.luogu.com.cn/problem/P8732）greedybrute_force两项优先级公式
8887（https://www.luogu.com.cn/problem/P8887）brain_teasergreedy

===================================CodeForces===================================
1186D（https://codeforces.com/problemset/problem/1186/D）greedy取floor，再根据|和为0的特质补充|1成为ceil
792C（https://codeforces.com/contest/792/problem/C）classification_discussion|greedy取数比较，取最长的返回结果
166E（https://codeforces.com/problemset/problem/166/E）思维implementionDP
1025C（https://codeforces.com/problemset/problem/1025/C）brain_teaser
1042C（https://codeforces.com/problemset/problem/1042/C）greedyclassification_discussion|implemention
439C（https://codeforces.com/problemset/problem/439/C）greedyclassification_discussion
1283E（https://codeforces.com/problemset/problem/1283/E）greedyclassification_discussion
1092C（https://codeforces.com/contest/1092/problem/C）brain_teaser思维classification_discussion|题
1280B（https://codeforces.com/problemset/problem/1280/B）brain_teaser思维classification_discussion|题
723C（https://codeforces.com/problemset/problem/723/C）greedyimplementionconstruction
712C（https://codeforces.com/problemset/problem/712/C）reverse_thinking反向implemention
747D（https://codeforces.com/problemset/problem/747/D）greedyimplemention求解
1148D（https://codeforces.com/problemset/problem/1148/D）greedy，define_sort选择construction
792C（https://codeforces.com/contest/792/problem/C）classification_discussion|greedy讨论
830A（https://codeforces.com/problemset/problem/830/A）按照影响区间sorting，然后greedy分配时间
478C（https://codeforces.com/problemset/problem/478/C）greedy结论题a<=b<=c则有min((a+b+c)//3, a+b)
1329A（https://codeforces.com/problemset/problem/1329/A）greedy+pointer+implemention
1401D（https://codeforces.com/problemset/problem/1401/D）greedydfsbrute_force经过边的路径counter
600C（https://codeforces.com/problemset/problem/600/C）palindrome_substringcountergreedy
1038D（https://codeforces.com/problemset/problem/1038/D）greedyimplemention，classification_discussion
349B（https://codeforces.com/problemset/problem/349/B）greedyimplemention
1370C（https://codeforces.com/problemset/problem/1370/C）greedyimplemention必胜态
1822E（https://codeforces.com/contest/1822/problem/E）greedyimplementioncounter
1005E2（https://codeforces.com/contest/1005/problem/E2）特定median的连续子数组个数，inclusion_exclusion|prefix_sumsorted_listbinary_search，同LC2488
1512E（https://codeforces.com/contest/1512/problem/E）brain_teaser|从大到小greedy

====================================AtCoder=====================================
C - AtCoDeer and Election Report（https://atcoder.jp/contests/abc046/tasks/arc062_a）brain_teaser|，不等式greedy
D - Wide Flip（https://atcoder.jp/contests/abc083/tasks/arc088_b）brain_teaser|greedy
D - Various Sushi（https://atcoder.jp/contests/abc116/tasks/abc116_d）brain_teaser|greedy
D - Summer Vacation（https://atcoder.jp/contests/abc137/tasks/abc137_d）逆序brain_teaser|greedy

=====================================AcWing=====================================
104（https://www.acwing.com/problem/content/106/）mediangreedy
1536（https://www.acwing.com/problem/content/description/1538/）greedy均分纸牌
105（https://www.acwing.com/problem/content/description/1538/）环形均分纸牌问题
110（https://www.acwing.com/problem/content/112/）greedy匹配最多组合
123（https://www.acwing.com/problem/content/description/125/）mediangreedy扩展问题
125（https://www.acwing.com/problem/content/127/）greedy思路，邻项交换
127（https://www.acwing.com/problem/content/description/129/）二维sortinggreedy
145（https://www.acwing.com/problem/content/147/）heapq|greedy
122（https://www.acwing.com/problem/content/124/）线性环形均分纸牌问题
4204（https://www.acwing.com/problem/content/description/4207/）construction
4307（https://www.acwing.com/problem/content/description/4310/）lexicographical_orderbrute_forcegreedy
4313（https://www.acwing.com/problem/content/4316/）满二叉树树形DPgreedy（同LC2673）
4426（https://www.acwing.com/problem/content/4429/）brain_teaser|brain_teaser，等价于末尾两位数字可以被4整除
4427（https://www.acwing.com/problem/content/4430/）树形greedyconstruction
4429（https://www.acwing.com/problem/content/description/4432/）邻项公式greedysorting，prefix_suffixbrute_force
4430（https://www.acwing.com/problem/content/description/4433/）括号匹配brute_force，prefix_suffix遍历
4492（https://www.acwing.com/problem/content/description/4495/）brain_teaser分为奇数与偶数讨论
4623（https://www.acwing.com/problem/content/description/4626/）greedyimplemention

"""

import heapq
import math
from bisect import insort_left, bisect_left
from collections import Counter, deque, defaultdict
from typing import List

from sortedcontainers import SortedList

from src.data_structure.sorted_list.template import LocalSortedList
from src.mathmatics.number_theory.template import NumberTheory
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1005e2(ac=FastIO()):
        # 特定median的连续子数组个数，inclusion_exclusion|prefix_sumsorted_listbinary_search
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()

        def check(x):
            cur = res = s = 0
            dct = defaultdict(int)
            dct[cur ^ ac.random_seed] = 1
            for num in nums:
                if num >= x:
                    s += dct[cur ^ ac.random_seed]
                    cur += 1
                else:
                    cur -= 1
                    s -= dct[cur ^ ac.random_seed]
                res += s
                dct[cur ^ ac.random_seed] += 1
            return res

        ac.st(check(m) - check(m + 1))
        return

    @staticmethod
    def cf_1038d(ac=FastIO()):
        # classification_discussiongreedyimplemention
        n = ac.read_int()
        nums = ac.read_list_ints()
        if n == 1:
            ac.st(nums[0])
            return
        if n == 2:
            ac.st(ac.max(nums[0] - nums[1], nums[1] - nums[0]))
            return
        zero = nums.count(0)
        if zero >= 2:
            ac.st(sum(abs(num) for num in nums))
        elif zero == 1:
            ac.st(sum(abs(num) for num in nums))
        else:
            if all(num > 0 for num in nums):
                ac.st(sum(nums) - 2 * min(nums))
            elif all(num < 0 for num in nums):
                ac.st(sum(abs(num) for num in nums) - 2 * min(abs(num) for num in nums))
            else:
                ac.st(sum(abs(num) for num in nums))
        return

    @staticmethod
    def main(ac=FastIO()):
        s = ac.read_str()
        cnt = Counter(s)
        n = len(s)
        double = []
        single = []
        for w in cnt:
            if cnt[w] % 2 == 0:
                x = cnt[w] // 2
                double.append([w, x])
            else:
                x = cnt[w] // 2
                if x:
                    double.append([w, x])
                single.append(w)
        if n % 2 == 0:
            single.sort()
            m = len(single)
            for i in range(m // 2):
                double.append([single[i], 1])
            double.sort(key=lambda it: it[0])
            ans = ""
            for w, c in double:
                ans += w * c
            ac.st(ans + ans[::-1])

        else:
            single.sort()
            m = len(single)
            for i in range(m // 2):
                double.append([single[i], 1])
            double.sort(key=lambda it: it[0])
            ans = ""
            for w, c in double:
                ans += w * c
            ac.st(ans + single[m // 2] + ans[::-1])
        return

    @staticmethod
    def lg_p2512(ac=FastIO()):
        # 环形均分纸牌问题
        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]
        m = sum(nums) // n
        x = 0
        pre = []
        for i in range(n):
            x += m - nums[i]
            pre.append(x)
        pre.sort()
        y = pre[n // 2]
        ans = sum(abs(num - y) for num in pre)
        ac.st(ans)
        return

    @staticmethod
    def abc_46b(ac=FastIO()):
        # brain_teaser|，不等式greedy
        n = ac.read_int()
        a = b = 1
        for _ in range(n):
            x, y = ac.read_list_ints()
            z1 = a // x + int(a % x > 0)
            z2 = b // y + int(b % y > 0)
            z = ac.max(z1, z2)
            a = z * x
            b = z * y
        ac.st(a + b)
        return

    @staticmethod
    def ac_105(ac=FastIO()):

        def check(nums):
            # 环形均分纸牌
            nn = len(nums)
            s = sum(nums)
            if s % nn:
                return -1
            mm = s // nn
            x = 0
            pre = []
            for i in range(nn):
                x += mm - nums[i]
                pre.append(x)
            pre.sort()
            y = pre[nn // 2]
            ans = sum(abs(num - y) for num in pre)
            return ans

        m, n, t = ac.read_list_ints()
        row = [0] * m
        col = [0] * n
        for _ in range(t):
            xx, yy = ac.read_list_ints_minus_one()
            row[xx] += 1
            col[yy] += 1
        ans1 = check(row)
        ans2 = check(col)
        if ans1 != -1 and ans2 != -1:
            ac.lst(["both", ans1 + ans2])
        elif ans1 != -1:
            ac.lst(["row", ans1])
        elif ans2 != -1:
            ac.lst(["column", ans2])
        else:
            ac.st("impossible")
        return

    @staticmethod
    def ac_123(ac=FastIO()):
        # mediangreedy扩展问题，连续相邻sorting减去下标后再sorting
        n = ac.read_int()
        lst_x = []
        lst_y = []
        for _ in range(n):
            x, y = ac.read_list_ints()
            lst_y.append(y)
            lst_x.append(x)
        lst_y.sort()
        mid = lst_y[n // 2]
        ans = sum(abs(pos - mid) for pos in lst_y)

        lst_x.sort()
        lst_x = [lst_x[i] - i for i in range(n)]
        lst_x.sort()
        mid = lst_x[n // 2]
        ans += sum(abs(pos - mid) for pos in lst_x)
        ac.st(ans)
        return

    @staticmethod
    def ac_125(ac=FastIO()):
        # greedy思路，邻项交换
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda it: it[0] + it[1])
        ans = -math.inf
        pre = 0
        for w, s in nums:
            ans = ac.max(ans, pre - s)
            pre += w
        ac.st(ans)
        return

    @staticmethod
    def ac_127(ac=FastIO()):
        # 二维sortinggreedy
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
                insort_left(lst, machine[j][1])
                j += 1
            ind = bisect_left(lst, level)
            if ind < len(lst):
                lst.pop(ind)
                ans += 1
                money += 500 * tm + 2 * level
        ac.lst([ans, money])
        return

    @staticmethod
    def ac_145(ac=FastIO()):
        # heapq|greedy
        lst = []
        cnt = 0
        while cnt < 10000:
            cur = ac.read_list_ints()
            if not cur:
                cnt += 1
            lst.extend(cur)

        lst = deque(lst)
        while lst:
            n = lst.popleft()
            cur = []
            for _ in range(n):
                p, d = lst.popleft(), lst.popleft()
                cur.append([p, d])
            cur.sort(key=lambda it: it[1])
            stack = []
            for p, d in cur:
                heapq.heappush(stack, p)
                if len(stack) == d + 1:
                    heapq.heappop(stack)
            ac.st(sum(stack))
        return

    @staticmethod
    def lc_2745(x: int, y: int, z: int) -> int:
        # brain_teasergreedybrain_teaser|
        return z * 2 + min(x, y) * 4 + int((max(x, y) - min(x, y)) > 0) * 2

    @staticmethod
    def lg_p1080(ac=FastIO()):
        # greedy，举例两项确定sorting公式
        n = ac.read_int()
        a, b = ac.read_list_ints()
        lst = [ac.read_list_ints() for _ in range(n)]
        lst.sort(key=lambda x: x[0] * x[1] - x[1])
        ans = 0
        pre = a
        for a, b in lst:
            ans = ac.max(ans, pre // b)
            pre *= a
        ac.st(ans)
        return

    @staticmethod
    def lg_p1650(ac=FastIO()):
        # greedy，优先上对上其次下对下最后下对上
        ac.read_int()
        a = deque(sorted(ac.read_list_ints(), reverse=True))
        b = deque(sorted(ac.read_list_ints(), reverse=True))

        ans = 0
        while a and b:
            # 上对上
            if a[0] > b[0]:
                a.popleft()
                b.popleft()
                ans += 200
            # 下对下
            elif a[-1] > b[-1]:
                a.pop()
                b.pop()
                ans += 200
            # 下对上
            else:
                x = a.pop()
                y = b.popleft()
                if x > y:
                    ans += 200
                elif x < y:
                    ans -= 200
        ac.st(ans)
        return

    @staticmethod
    def lg_p2088(ac=FastIO()):
        # 队列集合greedy，取空闲的，或者下一个离得最远的
        ans = 0
        k, n = ac.read_list_ints()
        nums = []
        while len(nums) < n:
            nums.extend(ac.read_list_ints())

        busy = set()
        post = defaultdict(deque)
        for i in range(n):
            post[nums[i]].append(i)
        for x in post:
            post[x].append(n)

        for i in range(n):
            if nums[i] in busy:
                continue
            if len(busy) < k:
                busy.add(nums[i])
                continue
            nex = -1
            for x in busy:
                while post[x] and post[x][0] < i:
                    post[x].popleft()
                if nex == -1 or post[x][0] >= post[nex][0]:
                    nex = x
            busy.discard(nex)
            busy.add(nums[i])
            ans += 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p2816(ac=FastIO()):
        # sorting后从小到大greedy放置，STL维护当前积木列高度
        lst = LocalSortedList()
        ac.read_int()
        nums = ac.read_list_ints()
        nums.sort()
        for num in nums:
            i = lst.bisect_left(num)
            if (0 <= i < len(lst) and lst[i] > num) or i == len(lst):
                i -= 1
            if 0 <= i < len(lst):
                lst.add(lst.pop(i) + 1)
            else:
                lst.add(1)
        ac.st(len(lst))
        return

    @staticmethod
    def lg_p3819(ac=FastIO()):
        # mediangreedy题
        length, n = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        s = sum(x for _, x in nums)
        nums.sort()
        pre = 0
        for pos, x in nums:
            pre += x
            if pre >= s // 2:
                ans = sum(abs(x - pos) * y for x, y in nums)
                ac.st(ans)
                break
        return

    @staticmethod
    def lg_p4025(ac=FastIO()):
        # greedy血量与增幅define_sort
        n, z = ac.read_list_ints()
        pos = []
        neg = []
        for i in range(n):
            d, a = ac.read_list_ints()
            if d < a:
                pos.append([d, a, i])
            else:
                neg.append([d, a, i])
        pos.sort(key=lambda it: it[0])
        neg.sort(key=lambda it: -it[1])
        ans = []
        for d, a, i in pos + neg:
            z -= d
            if z <= 0:
                ac.st("NIE")
                return
            z += a
            ans.append(i + 1)
        ac.st("TAK")
        ac.lst(ans)
        return

    @staticmethod
    def lg_p4266(ac=FastIO()):
        # 后缀最大值greedyimplemention
        length, n, rf, rb = ac.read_list_ints()
        nums = [[0, 0]] + [ac.read_list_ints() for _ in range(n)]
        n += 1
        # 记录后缀最大值序列
        post = [n - 1] * n
        ind = n - 1
        for i in range(n - 2, -1, -1):
            post[i] = ind
            if nums[i][1] > nums[ind][1]:
                ind = i
        path = [post[0]]
        while post[path[-1]] != path[-1]:
            path.append(post[path[-1]])
        # implemention
        ans = t = pre = 0
        for i in path:
            cur = nums[i][0]
            c = nums[i][1]
            ans += (cur - pre) * (rf - rb) * c
            t += (cur - pre) * rf
            pre = cur
        ac.st(ans)
        return

    @staticmethod
    def lg_p4447(ac=FastIO()):
        # greedy队列使得连续值序列最少的分组长度最大
        ac.read_int()
        lst = ac.read_list_ints()
        lst.sort()
        # 记录末尾值为 num 的连续子序列长度
        cnt = defaultdict(list)
        for num in lst:
            if cnt[num - 1]:
                # 将最小的长度取出添上
                val = heapq.heappop(cnt[num - 1])
                heapq.heappush(cnt[num], val + 1)
            else:
                # 单独成为一个序列
                heapq.heappush(cnt[num], 1)
        ac.st(min(min(cnt[k]) for k in cnt if cnt[k]))
        return

    @staticmethod
    def lg_p4575(ac=FastIO()):
        # brain_teaser|状压运算
        for _ in range(ac.read_int()):
            m = ac.read_int()
            k = ac.read_int()
            dct = [set() for _ in range(m)]
            for _ in range(k):
                i, j = ac.read_list_ints()
                dct[i].add(j)

            dp = [sum((1 << j) for j in dct[i]) for i in range(m)]
            ans = True
            for i in range(m):
                if not ans:
                    break
                for j in range(i + 1, m):
                    if dp[i] & dp[j] and dp[i] ^ dp[j]:
                        ans = False
                        break
            ac.st("Yes" if ans else "No")
        return

    @staticmethod
    def lg_p4653(ac=FastIO()):

        # 看似binary_searchpointergreedy选取
        n = ac.read_int()
        nums1 = []
        nums2 = []
        for _ in range(n):
            x, y = ac.read_list_floats()
            nums1.append(x)
            nums2.append(y)
        nums1.sort(reverse=True)
        nums2.sort(reverse=True)
        # two_pointers选择
        ans = i = j = a = b = 0
        light_a = light_b = 0
        while i < n or j < n:
            if i < n and (a - light_b < b - light_a or j == n):
                a += nums1[i] - 1
                i += 1
                light_a += 1
            else:
                b += nums2[j] - 1
                j += 1
                light_b += 1
            ans = ac.max(ans, ac.min(a - light_b, b - light_a))
        ac.st("%.4f" % ans)
        return

    @staticmethod
    def lg_p5093(ac=FastIO()):
        # brain_teaser集合确定轮数
        n, k = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(n)]
        pre = set()
        ans = 1
        for num in nums:
            pre.add(num)
            if len(pre) == k:
                ans += 1
                pre = set()
        ac.st(ans)
        return

    @staticmethod
    def lg_p5425(ac=FastIO()):
        # 看似最小生成树，实则brain_teasergreedy距离
        n, k = ac.read_list_ints()
        ans = (2019201913 * (k - 1) + 2019201949 * n) % 2019201997
        ac.st(ans)
        return

    @staticmethod
    def lg_p5884(ac=FastIO()):
        # brain_teaser
        n = ac.read_int()
        degree = [0] * n
        edge = []
        while True:
            lst = ac.read_list_ints()
            if not lst:
                break
            i, j = lst
            if i > j:
                i, j = j, i
            degree[i] += 1
            edge.append([i, j])
        # 只有一个节点是最后一条边才确认
        for i, j in edge:
            degree[i] -= 1
            if not degree[i]:
                ac.st(1)
            else:
                ac.st(0)
        return

    @staticmethod
    def lg_p6196(ac=FastIO()):
        # greedy 1 分段代价
        ac.read_int()
        nums = ac.read_list_ints()
        ans = 0
        lst = []
        for num in nums:
            if num == 1:
                if lst:
                    m = len(lst)
                    ans += min(lst)
                    for i in range(1, m):
                        ans += lst[i - 1] * lst[i]
                ans += 1
                lst = []
            else:
                lst.append(num)
        if lst:
            m = len(lst)
            ans += min(lst)
            for i in range(1, m):
                ans += lst[i - 1] * lst[i]
        ac.st(ans)
        return

    @staticmethod
    def lg_p6874(ac=FastIO()):
        # 变换公式转为mediangreedy
        n = ac.read_int()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        for i in range(n):
            w = abs(i - n // 2)
            a[i] -= w
            b[i] -= w
        a.extend(b)
        del b
        a.sort()
        x = ac.max(0, a[n])
        ac.st(sum(abs(x - num) for num in a))
        return

    @staticmethod
    def lg_p8050(ac=FastIO()):
        # brain_teaser黑白染色法任意操作不改变黑白元素和的差值
        m1, n1, m2, n2, k = ac.read_list_ints()
        black = white = cnt = state = 0
        for i in range(m1 + m2):
            lst = ac.read_list_ints()
            for j in range(len(lst)):
                if lst[j] != 999999:
                    if (i + j) % 2:
                        black += lst[j]
                        cnt += 1
                    else:
                        white += lst[j]
                else:
                    state = (i + j) % 2
                    cnt += (i + j) % 2
        ans = (2 * cnt - m1 * n1 - m2 * n2) * k - black + white
        if not state:
            ans = -ans
        ac.st(ans)
        return

    @staticmethod
    def lg_p8732(ac=FastIO()):
        # greedybrute_force两项优先级公式
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda it: sum(it))
        ans = pre = 0
        for s, a, e in nums:
            pre += s + a
            ans += pre
            pre += e
        ac.st(ans)
        return

    @staticmethod
    def ac_4307(ac=FastIO()):
        # lexicographical_orderbrute_forcegreedy
        a = [int(w) for w in str(ac.read_int())]
        b = [int(w) for w in str(ac.read_int())]
        a.sort()
        if len(a) < len(b):
            res = a[::-1]
        else:
            n = len(a)
            res = []
            for x in range(n):
                for i in range(n - 1 - x, -1, -1):
                    tmp = a[:]
                    tmp.pop(i)
                    if res + [a[i]] + tmp <= b:
                        res.append(a[i])
                        a.pop(i)
                        break
        ac.st("".join(str(x) for x in res))
        return

    @staticmethod
    def ac_4313(ac=FastIO()):
        # 满二叉树树形DPgreedy
        n = ac.read_int()
        m = 2 ** (n + 1)
        dp = [0] * m
        nums = ac.read_list_ints()
        ans = 0
        for i in range(m // 2 - 1, 0, -1):
            left = dp[i * 2] + nums[i * 2 - 2]
            right = dp[i * 2 + 1] + nums[i * 2 - 1]
            x = ac.max(left, right)
            dp[i] = x
            ans += x * 2 - left - right
        ac.st(ans)
        return

    @staticmethod
    def ac_4426(ac=FastIO()):

        # brain_teaser|brain_teaser，等价于末尾两位数字可以被4整除
        s = ac.read_str()
        ans = 0
        n = len(s)
        for i in range(n):
            if i - 1 >= 0 and int(s[i - 1:i + 1]) % 4 == 0:
                ans += i  # 两位数及以上
            if int(s[i]) % 4 == 0:
                ans += 1  # 一位数
        ac.st(ans)
        return

    @staticmethod
    def ac_4427(ac=FastIO()):
        # 树形greedyconstruction
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        parent = ac.read_list_ints_minus_one()
        for i in range(n - 1):
            dct[parent[i]].append(i + 1)
        s = ac.read_list_ints()
        ans = [0] * n

        stack = [[0, 0, 0]]
        while stack:
            x, pre, ss = stack.pop()
            pre += 1
            if pre % 2:  # 奇数位没得选
                ans[x] = s[x] - ss
            else:
                lst = []  # 偶数位greedy取最大值
                for y in dct[x]:
                    lst.append(s[y])
                if lst:
                    ans[x] = min(lst) - ss

            for y in dct[x]:
                stack.append([y, pre, ss + ans[x]])

        ac.st(sum(ans) if all(x >= 0 for x in ans) else -1)
        return

    @staticmethod
    def ac_4429(ac=FastIO()):
        # 邻项公式greedysorting，prefix_suffixbrute_force
        n, x1, y1, x2, y2 = ac.read_list_ints()
        pos = [ac.read_list_ints() for _ in range(n)]
        dis1 = [(x - x1) * (x - x1) + (y - y1) * (y - y1) for x, y in pos]
        dis2 = [(x - x2) * (x - x2) + (y - y2) * (y - y2) for x, y in pos]
        # sorting
        ind = list(range(n))
        ind.sort(key=lambda it: dis1[it] - dis2[it])
        # 后缀最大值
        post = [0] * (n + 1)
        ceil = 0
        for i in range(n - 1, -1, -1):
            ceil = ac.max(dis2[ind[i]], ceil)
            post[i] = ceil

        # brute_force前缀
        ans = post[0]
        pre = 0
        for i in range(n):
            pre = ac.max(pre, dis1[ind[i]])
            if pre + post[i + 1] < ans:
                ans = pre + post[i + 1]
        ac.st(ans)
        return

    @staticmethod
    def ac_4430(ac=FastIO()):
        # 括号匹配brute_force，prefix_suffix遍历
        n = ac.read_int()
        s = ac.read_str()
        ans = 0

        # 左变右
        post = [-1] * (n + 1)
        post[n] = 0
        right = 0
        for i in range(n - 1, -1, -1):
            if s[i] == ")":
                right += 1
            else:
                if not right:
                    break
                right -= 1
            post[i] = right

        left = 0
        for i in range(n):
            if s[i] == ")":
                if not left:
                    break
                left -= 1
            else:
                if post[i + 1] + 1 == left and post[i + 1] != -1:
                    ans += 1
                left += 1

        # 右变左
        pre = [-1] * (n + 1)
        pre[0] = 0
        left = 0
        for i in range(n):
            if s[i] == "(":
                left += 1
            else:
                if not left:
                    break
                left -= 1
            pre[i + 1] = left

        right = 0
        for i in range(n - 1, -1, -1):
            if s[i] == "(":
                if not right:
                    break
                right -= 1
            else:
                if pre[i] + 1 == right and pre[i] != -1:
                    ans += 1
                right += 1
        ac.st(ans)
        return

    @staticmethod
    def ac_4492(ac=FastIO()):
        # brain_teaser分为奇数与偶数讨论
        n = ac.read_int()
        if n % 2 == 0:
            ac.st(n // 2)
        else:
            lst = NumberTheory().get_prime_factor(n)
            x = lst[0][0]
            ac.st(1 + (n - x) // 2)
        return

    @staticmethod
    def ac_4623(ac=FastIO()):
        # greedyimplemention
        n, t = ac.read_list_ints()
        a = ac.read_list_ints()
        ans = 0
        while a:
            s = sum(a)
            ans += (t // s) * len(a)
            t %= s
            b = []
            for num in a:
                if t >= num:
                    t -= num
                    ans += 1
                    b.append(num)
            a = b[:]
        ac.st(ans)
        return

    @staticmethod
    def lc_858(p: int, q: int) -> int:

        # brain_teaserbrain_teaser|

        g = math.gcd(p, q)
        # 求解等式 k*p = m*q

        # 左右次数k合计为偶数
        k = p // g
        if k % 2 == 0:
            return 2

        # 上下次数m合计为偶数
        m = q // g
        if m % 2 == 0:
            return 0
        return 1

    @staticmethod
    def lc_991(start: int, target: int) -> int:
        # 逆向greedy，偶数除2奇数|1
        ans = 0
        while target > start:
            if target % 2:
                target += 1
            else:
                target //= 2
            ans += 1
        return ans + start - target

    @staticmethod
    def lc_1503(n: int, left: List[int], right: List[int]) -> int:
        # 
        ans = 0
        for x in left:
            if x > ans:
                ans = x
        for x in right:
            if n - x > ans:
                ans = n - x
        return ans

    @staticmethod
    def lc_1675(nums: List[int]) -> int:
        # brain_teaserbrain_teaser|greedy
        lst = SortedList([num if num % 2 == 0 else num * 2 for num in nums])
        ans = lst[-1] - lst[0]
        while True:
            cur = lst[-1] - lst[0]
            ans = ans if ans < cur else cur
            if lst[-1] % 2:
                break
            lst.add(lst.pop() // 2)
        return ans

    @staticmethod
    def lc_1808(prime_factors: int) -> int:
        # 按照模3的因子个数greedy处理，将和拆分成最大乘积
        mod = 10 ** 9 + 7
        if prime_factors <= 2:
            return prime_factors
        if prime_factors % 3 == 0:
            return pow(3, prime_factors // 3, mod)
        elif prime_factors % 3 == 1:
            return (4 * pow(3, prime_factors // 3 - 1, mod)) % mod
        else:
            return (2 * pow(3, prime_factors // 3, mod)) % mod

    @staticmethod
    def lc_1927(num: str) -> bool:
        # 博弈brain_teaser|classification_discussion
        def check(s):
            res = 0
            cnt = 0
            for w in s:
                if w.isnumeric():
                    res += int(w)
                else:
                    cnt += 1
            return [res, cnt]

        # 左右两边的数字和以及问号个数
        n = len(num)
        a, x = check(num[:n // 2])
        b, y = check(num[n // 2:])

        # Alice把宝压在右边
        b_add = 9 * (y // 2 + y % 2)
        if y % 2 == 0:
            a_add = 9 * (x // 2)
        else:
            a_add = 9 * (x // 2 + x % 2)
        if a + a_add < b + b_add:
            return True

        # Alice把宝压在左边
        a_add = 9 * (x // 2 + x % 2)
        if x % 2 == 0:
            b_add = 9 * (y // 2)
        else:
            b_add = 9 * (y // 2 + y % 2)
        if b + b_add < a + a_add:
            return True

        # 左右都不能获胜
        return False

    @staticmethod
    def lc_2592(nums: List[int]) -> int:
        # 典型greedysorting后two_pointers
        n = len(nums)
        nums.sort()
        j = 0
        ans = 0
        for i in range(n):
            while j < n and nums[i] >= nums[j]:
                j += 1
            if j < n:
                ans += 1
                j += 1
        return ans

    @staticmethod
    def lc_2568(nums: List[int]) -> int:
        # brain_teasergreedy，可以根据打表观察规律
        dct = set(nums)
        for i in range(34):
            if 1 << i not in dct:
                return 1 << i
        return -1