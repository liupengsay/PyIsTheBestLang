import heapq
from bisect import insort_left, bisect_left
from collections import Counter, deque, defaultdict
from typing import List
from math import inf

from sortedcontainers import SortedList

from src.data_structure.sorted_list import LocalSortedList
from src.fast_io import FastIO
from src.mathmatics.number_theory import NumberTheory

"""

算法：贪心、逆向思维、抽屉原理、鸽巢原理、容斥原理、自定义排序、思维、脑筋急转弯、构造
功能：各种可证明不可证明的头脑风暴
题目：

===================================力扣===================================
134. 加油站（https://leetcode.cn/problems/gas-station/）经典贪心
330. 按要求补齐数组（https://leetcode.cn/problems/patching-array/）经典贪心
1199. 建造街区的最短时间（https://leetcode.cn/problems/minimum-time-to-build-blocks/）经典哈夫曼树Huffman Tree贪心从小到大合并，类似合并果子
2499. 让数组不相等的最小总代价（https://leetcode.cn/problems/minimum-total-cost-to-make-arrays-unequal/）利用鸽巢原理贪心计算最小代价
2449. 使数组相似的最少操作次数（https://leetcode.cn/problems/minimum-total-cost-to-make-arrays-unequal/）转换题意进行排序后用奇偶数贪心变换得到
2448. 使数组相等的最小开销（https://leetcode.cn/problems/minimum-cost-to-make-array-equal/）利用中位数的特点变换到带权重广义下中位数的位置是最优的贪心进行增减
2412. 完成所有交易的初始最少钱数（https://leetcode.cn/problems/minimum-money-required-before-transactions/）根据交易增长特点进行自定义排序
2366. 将数组排序的最少替换次数（https://leetcode.cn/problems/minimum-replacements-to-sort-the-array/）倒序贪心不断分解得到满足要求且尽可能大的值
2350. 不可能得到的最短骰子序列（https://leetcode.cn/problems/shortest-impossible-sequence-of-rolls/）脑筋急转弯本质上是求全排列出现的轮数
2344. 使数组可以被整除的最少删除次数（https://leetcode.cn/problems/minimum-deletions-to-make-array-divisible/）利用最大公约数贪心删除最少的元素
2136. 全部开花的最早一天（https://leetcode.cn/problems/earliest-possible-day-of-full-bloom/）贪心安排成长时间最长的先种
2071. 你可以安排的最多任务数目（https://leetcode.cn/problems/maximum-number-of-tasks-you-can-assign/）使用贪心加二分进行极值判断
517. 超级洗衣机（https://leetcode.cn/problems/super-washing-machines/）类似上题，计算最小的左右移动次数以及往左右的移动次数
1798. 你能构造出连续值的最大数目（https://leetcode.cn/problems/maximum-number-of-consecutive-values-you-can-make/）看似背包实则贪心
625. 最小因式分解（https://leetcode.cn/problems/minimum-factorization/）贪心进行因式分解，类似质因数分解
2568. 最小无法得到的或值（https://leetcode.cn/problems/minimum-impossible-or/）脑筋急转弯贪心，可以根据暴力打表观察规律
6361. 修改两个元素的最小分数（https://leetcode.cn/problems/minimum-score-by-changing-two-elements/）脑筋急转弯贪心
6316. 重排数组以得到最大前缀分数（https://leetcode.cn/contest/weekly-contest-336/problems/rearrange-array-to-maximize-prefix-score/）贪心，加前缀和
2436. 使子数组最大公约数大于一的最小分割数（https://leetcode.cn/problems/minimum-split-into-subarrays-with-gcd-greater-than-one/）贪心计算
1029. 两地调度（https://leetcode.cn/problems/two-city-scheduling/）经典贪心题目，可使用举例两个计算、再进行归纳确定排序规则
1353. 最多可以参加的会议数目（https://leetcode.cn/problems/maximum-number-of-events-that-can-be-attended/）经典枚举贪心
1402. 做菜顺序（https://leetcode.cn/problems/reducing-dishes/）经典前缀和贪心
1665. 完成所有任务的最少初始能量（https://leetcode.cn/problems/minimum-initial-energy-to-finish-tasks/）经典贪心不同项比较公式排序模拟
1675. 数组的最小偏移量（https://leetcode.cn/problems/minimize-deviation-in-array/）脑筋急转弯思维题贪心
1686. 石子游戏 VI（https://leetcode.cn/problems/stone-game-vi/）经典贪心采用列式子确定排序方式
1808. 好因子的最大数目（https://leetcode.cn/problems/maximize-number-of-nice-divisors/）按照模3的因子个数贪心处理，经典将和拆分成最大乘积
1953. 你可以工作的最大周数（https://leetcode.cn/problems/maximum-number-of-weeks-for-which-you-can-work/）经典贪心只看最大值的影响
858. 镜面反射（https://leetcode.cn/problems/mirror-reflection/description/）经典脑筋急转弯思维题
1927. 求和游戏（https://leetcode.cn/problems/sum-game/description/）经典博弈思维题分类讨论
2592. 最大化数组的伟大值（https://leetcode.cn/problems/maximize-greatness-of-an-array/）典型贪心排序后使用双指针计算

===================================洛谷===================================
P1031 均分纸牌（https://www.luogu.com.cn/problem/P1031）贪心计算每个点的前缀和流量，需要补齐或者输出时进行计数
P1684 考验（https://www.luogu.com.cn/problem/P1684）线性贪心满足条件即增加计数
P1658 购物（https://www.luogu.com.cn/problem/P1658）看似背包实则贪心
P2001 硬币的面值（https://www.luogu.com.cn/problem/P2001）看似背包实则贪心
P1620 漂亮字串（https://www.luogu.com.cn/problem/P1620）分类讨论进行贪心
P2773 漂亮字串（https://www.luogu.com.cn/problem/P2773）分类讨论进行贪心
P2255 [USACO14JAN]Recording the Moolympics S（https://www.luogu.com.cn/problem/P2255）两个指针进行贪心
P2327 [SCOI2005]扫雷（https://www.luogu.com.cn/problem/P2327）脑筋急转弯进行枚举
P2777 [AHOI2016初中组]自行车比赛（https://www.luogu.com.cn/problem/P2777）贪心枚举最佳得分组合，加前后缀记录最大值
P2649 游戏预言（https://www.luogu.com.cn/problem/P2649）贪心，输的时候输最惨，赢的时候微弱优势
P1367 蚂蚁（https://www.luogu.com.cn/problem/P1367）脑筋急转弯，蚂蚁的相对移动位置排序还是不变
P1362 兔子数（https://www.luogu.com.cn/problem/P1362）找规律之后，进行广度优先搜索枚举
P1090 [NOIP2004 提高组] 合并果子 / [USACO06NOV] Fence Repair G（https://www.luogu.com.cn/record/list?user=739032&status=12&page=11）从小到大贪心合并
P1334 瑞瑞的木板（https://www.luogu.com.cn/problem/P1334）逆向思维的合并果子，从小到大合并
P1325 雷达安装（https://www.luogu.com.cn/problem/P1325）排序后进行贪心修建更新
P1250 种树（https://www.luogu.com.cn/problem/P1250）区间的贪心题，使用线段树修改区间与查询和，以及二分进行计算
P1230 智力大冲浪（https://www.luogu.com.cn/problem/P1230）排序后进行选取贪心
P1159 排行榜（https://www.luogu.com.cn/problem/P1159）使用队列贪心进行模拟
P1095 [NOIP2007 普及组] 守望者的逃离（https://www.luogu.com.cn/problem/P1095）贪心模拟也可以理解为动态规划转移
P1056 [NOIP2008 普及组] 排座椅（https://www.luogu.com.cn/record/list?user=739032&status=12&page=14）根据题意进行计数排序贪心选择
P8847 [JRKSJ R5] 1-1 A（https://www.luogu.com.cn/problem/P8847）分类讨论和贪心进行
P8845 [传智杯 #4 初赛] 小卡和质数（https://www.luogu.com.cn/problem/solution/P8845）脑筋急转弯，只有2是偶数质数
P2772 寻找平面上的极大点（https://www.luogu.com.cn/problem/P2772）按照两个维度排序，再按照其中一个维度顺序比较最大值
P2878 [USACO07JAN] Protecting the Flowers S（https://www.luogu.com.cn/problem/P2878）经典贪心题目，可使用举例两个计算、再进行归纳确定排序规则
P2920 [USACO08NOV]Time Management S（https://www.luogu.com.cn/problem/P2920）排序后进行贪心计算
P2983 [USACO10FEB]Chocolate Buying S（https://www.luogu.com.cn/problem/P2983）看起来是背包其实是贪心优先选择最便宜的奶牛满足
P3173 [HAOI2009]巧克力（https://www.luogu.com.cn/problem/P3173）从大到小排序进行贪心计算
P5098 [USACO04OPEN]Cave Cows 3（https://www.luogu.com.cn/problem/P5098）贪心按照一个维度排序后再按照另一个维度分类讨论，记录前缀最小值
P5159 WD与矩阵（https://www.luogu.com.cn/problem/P5159）利用异或的特点枚举计数并进行快速幂计算
P5497 [LnOI2019SP]龟速单项式变换(SMT)（https://www.luogu.com.cn/problem/P5497）抽屉原理进行分类讨论
P5682 [CSP-J2019 江西] 次大值（https://www.luogu.com.cn/problem/P5682）脑筋急转弯进行排序后贪心枚举确定
P5804 [SEERC2019]Absolute Game（https://www.luogu.com.cn/problem/P5804）排序贪心枚举和二分查找优化
P5963 [BalticOI ?] Card 卡牌游戏【来源请求】（https://www.luogu.com.cn/problem/P5963）经典贪心题目，可使用举例两个计算、再进行归纳确定排序规则
P6023 走路（https://www.luogu.com.cn/problem/P6023）可证明集中在某天是最佳结果，然后使用指针进行模拟计算
P6243 [USACO06OPEN]The Milk Queue G（https://www.luogu.com.cn/problem/P6243）经典贪心举例之后进行优先级比较，再自定义排序
P6179 [USACO15DEC]High Card Wins S（https://www.luogu.com.cn/problem/list?difficulty=3&page=13）经典贪心
P6380 『MdOI R2』Mayuri（https://www.luogu.com.cn/problem/P6380）贪心模拟进行赋值
P6446 [COCI2010-2011#1] TABOVI（https://www.luogu.com.cn/problem/P6446）贪心进行操作，经典使得数组所有值相等的最少操作次数变形题目，每次操作可以使得连续区间加1或者减1
P5019 [NOIP2018 提高组] 铺设道路（https://www.luogu.com.cn/problem/P5019）贪心进行操作，经典使得数组所有值相等的最少操作次数变形题目，每次操作可以使得连续区间加1或者减1
P6462 [传智杯 #2 决赛] 补刀（https://www.luogu.com.cn/problem/P6462）贪心进行分类计算
P6549 [COCI2010-2011#2] KNJIGE（https://www.luogu.com.cn/problem/P6549）逆向思维，使用插入排序的思想进行模拟
P6785 「EZEC-3」排列（https://www.luogu.com.cn/problem/P6785）脑筋急转弯进行，条件判断与贪心计数
P6851 onu（https://www.luogu.com.cn/problem/P6851）贪心模拟，均从大到小排序，先选择赢的牌，再计算输的牌
P7176 [COCI2014-2015#4] PRIPREME（https://www.luogu.com.cn/problem/P7176）贪心策略，结论题
P7228 [COCI2015-2016#3] MOLEKULE（https://www.luogu.com.cn/problem/P7228）脑筋急转弯贪心加树形dfs计算
P7260 [COCI2009-2010#3] RAZGOVOR（https://www.luogu.com.cn/problem/P7260）贪心与动态规划，经典使得数组所有值从0变化等于给定升序数组的最少操作次数，每次操作可以使得连续区间加1
P7319 「PMOI-4」生成树（https://www.luogu.com.cn/problem/P7319）公式变形后使用排序不等式进行贪心计算
P7412 [USACO21FEB] Year of the Cow S（https://www.luogu.com.cn/problem/P7412）贪心，将问题转换为去掉最长的k-1个非零距离
P7522 ⎌ Nurture ⎌（https://www.luogu.com.cn/problem/P7522）进行分类贪心讨论
P7633 [COCI2010-2011#5] BRODOVI（https://www.luogu.com.cn/problem/P7633）使用埃氏筛法思想，进行模拟贪心计算
P7714 「EZEC-10」排列排序（https://www.luogu.com.cn/problem/P7714）经典子序列排序使得整体有序，使用前缀最大值与指针计数确认子数组分割点
P7787 [COCI2016-2017#6] Turnir（https://www.luogu.com.cn/problem/P7787）脑筋急转弯，借助完全二叉树的思想
P7813 谜（https://www.luogu.com.cn/problem/P7813）贪心计算最大选取值
P1031 [NOIP2002 提高组] 均分纸牌（https://www.luogu.com.cn/problem/P1031）经典线性均分纸牌问题
P2512 [HAOI2008]糖果传递（https://www.luogu.com.cn/problem/P2512）经典线性环形均分纸牌问题
P1080 [NOIP2012 提高组] 国王游戏（https://www.luogu.com.cn/problem/P1080）经典贪心，举例两项确定排序公式
P1650 田忌赛马（https://www.luogu.com.cn/problem/P1650）经典贪心，优先上对上其次下对下最后下对上
P2088 果汁店的难题（https://www.luogu.com.cn/problem/P2088）贪心，取空闲的，或者下一个离得最远的使用
P2816 宋荣子搭积木（https://www.luogu.com.cn/problem/P2816）排序后从小到大贪心放置，使用STL维护当前积木列高度
P3819 松江 1843 路（https://www.luogu.com.cn/problem/P3819）经典中位数贪心题
P3918 [国家集训队]特技飞行（https://www.luogu.com.cn/problem/P3918）脑筋急转弯贪心
P4025 [PA2014]Bohater（https://www.luogu.com.cn/problem/P4025）经典贪心血量与增幅自定义排序
P4266 [USACO18FEB]Rest Stops SP4266 [USACO18FEB]Rest Stops S（https://www.luogu.com.cn/problem/P4266）后缀最大值贪心模拟
P4447 [AHOI2018初中组]分组（https://www.luogu.com.cn/problem/P4447）经典贪心队列使得连续值序列最少的分组长度最大
P4575 [CQOI2013]图的逆变换（https://www.luogu.com.cn/problem/P4575）脑筋急转弯加状压运算
P4653 [CEOI2017] Sure Bet（https://www.luogu.com.cn/problem/P4653）看似二分使用指针贪心选取
P5093 [USACO04OPEN]The Cow Lineup（https://www.luogu.com.cn/problem/P5093）经典脑筋急转弯使用集合确定轮数
P5425 [USACO19OPEN]I Would Walk 500 Miles G（https://www.luogu.com.cn/problem/P5425）看似最小生成树，实则脑筋急转弯贪心计算距离
P5884 [IOI2014]game 游戏（https://www.luogu.com.cn/problem/P5884）脑筋急转弯
P5948 [POI2003]Chocolate（https://www.luogu.com.cn/problem/P5948）贪心模拟进行计算
P6196 [EER1]代价（https://www.luogu.com.cn/problem/P6196）贪心使用 1 进行分段计算代价
P6874 [COCI2013-2014#6] KOCKICE（https://www.luogu.com.cn/problem/P6874）经典变换公式转为中位数贪心
P8050 [ZYOI Round1] Chessboard game/棋盘游戏（https://www.luogu.com.cn/problem/P8050）脑筋急转弯黑白染色法任意操作不改变黑白元素和的差值
P7935 [COCI2007-2008#5] AVOGADRO（https://www.luogu.com.cn/problem/P7935）脑筋急转弯
P8109 [Cnoi2021]幻想乡程序设计大赛（https://www.luogu.com.cn/problem/P8109）经典STL贪心分配求解
P8669 [蓝桥杯 2018 省 B] 乘积最大（https://www.luogu.com.cn/problem/P8669）贪心选取 k 个数乘积最大
P8709 [蓝桥杯 2020 省 A1] 超级胶水（https://www.luogu.com.cn/problem/P8709）脑筋急转弯模拟计算
P8732 [蓝桥杯 2020 国 ABC] 答疑（https://www.luogu.com.cn/problem/P8732）经典贪心枚举两项计算优先级公式
P8887 [DMOI-R1] 柯基棋（https://www.luogu.com.cn/problem/P8887）脑筋急转弯贪心

================================CodeForces================================
https://codeforces.com/problemset/problem/1186/D（贪心取floor，再根据加和为0的特质进行补充加1成为ceil）
https://codeforces.com/contest/792/problem/C（分类进行贪心取数比较，取最长的返回结果）
https://codeforces.com/problemset/problem/166/E（思维模拟DP）
https://codeforces.com/problemset/problem/1025/C（脑筋急转弯）
https://codeforces.com/problemset/problem/1042/C（贪心分类模拟）
https://codeforces.com/problemset/problem/439/C（贪心分类讨论）
https://codeforces.com/problemset/problem/1283/E（贪心分类讨论）
https://codeforces.com/contest/1092/problem/C（脑筋急转弯思维分类题）
https://codeforces.com/problemset/problem/1280/B（脑筋急转弯思维分类题）
https://codeforces.com/problemset/problem/723/C（贪心模拟构造）
https://codeforces.com/problemset/problem/712/C（逆向思维反向模拟）
https://codeforces.com/problemset/problem/747/D（贪心模拟求解）
https://codeforces.com/problemset/problem/1148/D（贪心，自定义排序选择构造）
https://codeforces.com/contest/792/problem/C（分类进行贪心讨论）
https://codeforces.com/problemset/problem/830/A （按照影响区间排序，然后贪心分配时间）
C. Table Decorations（https://codeforces.com/problemset/problem/478/C）贪心结论题a<=b<=c则有min((a+b+c)//3, a+b)
A. Dreamoon Likes Coloring（https://codeforces.com/problemset/problem/1329/A）贪心+指针+模拟
D. Maximum Distributed Tree（https://codeforces.com/problemset/problem/1401/D）贪心dfs枚举经过边的路径计数
C. Make Palindrome（https://codeforces.com/problemset/problem/600/C）回文子串计数贪心
D. Slime（https://codeforces.com/problemset/problem/1038/D）贪心模拟，分类讨论
B. Color the Fence（https://codeforces.com/problemset/problem/349/B）贪心模拟
C. Number Game（https://codeforces.com/problemset/problem/1370/C）贪心模拟必胜态
E. Making Anti-Palindromes（https://codeforces.com/contest/1822/problem/E）贪心进行模拟计数

================================AcWing======================================
104. 货仓选址（https://www.acwing.com/problem/content/106/）中位数贪心
1536. 均分纸牌（https://www.acwing.com/problem/content/description/1538/）贪心均分纸牌
105. 七夕祭（https://www.acwing.com/problem/content/description/1538/）经典环形均分纸牌问题
110. 防晒（https://www.acwing.com/problem/content/112/）贪心匹配最多组合
123. 士兵（https://www.acwing.com/problem/content/description/125/）中位数贪心扩展问题
125. 耍杂技的牛（https://www.acwing.com/problem/content/127/）经典贪心思路，邻项交换
127. 任务（https://www.acwing.com/problem/content/description/129/）经典二维排序贪心
145. 超市（https://www.acwing.com/problem/content/147/）经典使用二叉堆贪心
122. 糖果传递（https://www.acwing.com/problem/content/124/）经典线性环形均分纸牌问题
4204. 构造矩阵（https://www.acwing.com/problem/content/description/4207/）经典构造
4307. 数字重构（https://www.acwing.com/problem/content/description/4310/）经典字典序枚举贪心
4313. 满二叉树等长路径（https://www.acwing.com/problem/content/4316/）经典满二叉树树形DP贪心（同LC2673）
4426. 整除子串（https://www.acwing.com/problem/content/4429/）思维题脑筋急转弯，等价于末尾两位数字可以被4整除
4427. 树中节点和（https://www.acwing.com/problem/content/4430/）经典树形贪心构造
4429. 无线网络（https://www.acwing.com/problem/content/description/4432/）经典计算邻项公式贪心排序，使用前后缀枚举
4430. 括号序列（https://www.acwing.com/problem/content/description/4433/）经典括号匹配枚举，前后缀遍历计算
4492. 减法操作（https://www.acwing.com/problem/content/description/4495/）脑筋急转弯分为奇数与偶数讨论
4623. 买糖果（https://www.acwing.com/problem/content/description/4626/）贪心模拟

参考：OI WiKi（xx）
"""

import math
import unittest


class BrainStorming:
    def __init__(self):
        return

    @staticmethod
    def minimal_coin_need(n, m, nums):

        nums += [m + 1]
        nums.sort()
        # 有 n 个可选取且无限的硬币，为了形成 1-m 所有组合需要的最少硬币个数
        if nums[0] != 1:
            return -1
        ans = sum_ = 0
        for i in range(n):
            nex = nums[i + 1] - 1
            nex = nex if nex < m else m
            x = math.ceil((nex - sum_) / nums[i])
            x = x if x >= 0 else 0
            ans += x
            sum_ += x * nums[i]
            if sum_ >= m:
                break
        return ans


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1038d(ac=FastIO()):
        # 模板：分类讨论贪心模拟
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
        # 模板：经典环形均分纸牌问题
        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]
        m = sum(nums)//n
        x = 0
        pre = []
        for i in range(n):
            x += m - nums[i]
            pre.append(x)
        pre.sort()
        y = pre[n//2]
        ans = sum(abs(num-y) for num in pre)
        ac.st(ans)
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

        m, n, t = ac.read_ints()
        row = [0] * m
        col = [0] * n
        for _ in range(t):
            xx, yy = ac.read_ints_minus_one()
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
        # 模板：经典中位数贪心扩展问题，连续相邻排序减去下标后再排序
        n = ac.read_int()
        lst_x = []
        lst_y = []
        for _ in range(n):
            x, y = ac.read_ints()
            lst_y.append(y)
            lst_x.append(x)
        lst_y.sort()
        mid = lst_y[n//2]
        ans = sum(abs(pos-mid) for pos in lst_y)

        lst_x.sort()
        lst_x = [lst_x[i]-i for i in range(n)]
        lst_x.sort()
        mid = lst_x[n//2]
        ans += sum(abs(pos-mid) for pos in lst_x)
        ac.st(ans)
        return

    @staticmethod
    def ac_125(ac=FastIO()):
        # 模板：经典贪心思路，邻项交换
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda it: it[0]+it[1])
        ans = -math.inf
        pre = 0
        for w, s in nums:
            ans = ac.max(ans, pre-s)
            pre += w
        ac.st(ans)
        return

    @staticmethod
    def ac_127(ac=FastIO()):
        # 模板：经典二维排序贪心
        n, m = ac.read_ints()
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
                money += 500*tm + 2*level
        ac.lst([ans, money])
        return

    @staticmethod
    def ac_145(ac=FastIO()):
        # 模板：使用二叉堆贪心计算
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
                if len(stack) == d+1:
                    heapq.heappop(stack)
            ac.st(sum(stack))
        return

    @staticmethod
    def lg_p1080(ac=FastIO()):
        # 模板：经典贪心，举例两项确定排序公式
        n = ac.read_int()
        a, b = ac.read_ints()
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
        # 模板：经典贪心，优先上对上其次下对下最后下对上
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
        # 模板：使用队列集合贪心，取空闲的，或者下一个离得最远的使用
        ans = 0
        k, n = ac.read_ints()
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
        # 模板：排序后从小到大贪心放置，使用STL维护当前积木列高度
        lst = LocalSortedList()
        ac.read_int()
        nums = ac.read_list_ints()
        nums.sort()
        for num in nums:
            i = lst.bisect_left(num)
            if (0 <= i < len(lst) and lst[i] > num) or i == len(lst):
                i -= 1
            if 0 <= i < len(lst):
                lst.add(lst.pop(i)+1)
            else:
                lst.add(1)
        ac.st(len(lst))
        return

    @staticmethod
    def lg_p3819(ac=FastIO()):
        # 模板：经典中位数贪心题
        length, n = ac.read_ints()
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
        # 模板：经典贪心血量与增幅自定义排序
        n, z = ac.read_ints()
        pos = []
        neg = []
        for i in range(n):
            d, a = ac.read_ints()
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
        # 模板：后缀最大值贪心模拟
        length, n, rf, rb = ac.read_ints()
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
        # 模拟
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
        # 模板：经典贪心队列使得连续值序列最少的分组长度最大
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
        # 模板：脑筋急转弯加状压运算
        for _ in range(ac.read_int()):
            m = ac.read_int()
            k = ac.read_int()
            dct = [set() for _ in range(m)]
            for _ in range(k):
                i, j = ac.read_ints()
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

        # 模板：看似二分使用指针贪心选取
        n = ac.read_int()
        nums1 = []
        nums2 = []
        for _ in range(n):
            x, y = ac.read_list_floats()
            nums1.append(x)
            nums2.append(y)
        nums1.sort(reverse=True)
        nums2.sort(reverse=True)
        # 双指针选择
        ans = i = j = a = b = 0
        light_a = light_b = 0
        while i < n or j < n:
            if i < n and (a - light_b < b - light_a or j == n):
                a += nums1[i]-1
                i += 1
                light_a += 1
            else:
                b += nums2[j]-1
                j += 1
                light_b += 1
            ans = ac.max(ans, ac.min(a - light_b, b - light_a))
        ac.st("%.4f" % ans)
        return

    @staticmethod
    def lg_p5093(ac=FastIO()):
        # 模板：经典脑筋急转弯使用集合确定轮数
        n, k = ac.read_ints()
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
        # 模板：看似最小生成树，实则脑筋急转弯贪心计算距离
        n, k = ac.read_ints()
        ans = (2019201913 * (k - 1) + 2019201949 * n) % 2019201997
        ac.st(ans)
        return

    @staticmethod
    def lg_p5884(ac=FastIO()):
        # 模板：脑筋急转弯
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
        # 模板：贪心使用 1 进行分段计算代价
        n = ac.read_int()
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
        # 模板：经典变换公式转为中位数贪心
        n = ac.read_int()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        for i in range(n):
            w = abs(i-n//2)
            a[i] -= w
            b[i] -= w
        a.extend(b)
        del b
        a.sort()
        x = ac.max(0, a[n])
        ac.st(sum(abs(x-num) for num in a))
        return

    @staticmethod
    def lg_p8050(ac=FastIO()):
        # 模板：脑筋急转弯黑白染色法任意操作不改变黑白元素和的差值
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
        # 模板：经典贪心枚举两项计算优先级公式
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
        # 模板：经典字典序枚举贪心
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
        # 模板：经典满二叉树树形DP贪心
        n = ac.read_int()
        m = 2**(n+1)
        dp = [0]*m
        nums = ac.read_list_ints()
        ans = 0
        for i in range(m//2-1, 0, -1):
            left = dp[i*2] + nums[i*2-2]
            right = dp[i*2+1] + nums[i*2-1]
            x = ac.max(left, right)
            dp[i] = x
            ans += x*2 - left - right
        ac.st(ans)
        return

    @staticmethod
    def ac_4426(ac=FastIO()):

        # 模板：思维题脑筋急转弯，等价于末尾两位数字可以被4整除
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
        # 模板：树形贪心构造
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
                lst = []   # 偶数位贪心取最大值
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
        # 模板：经典计算邻项公式贪心排序，使用前后缀枚举
        n, x1, y1, x2, y2 = ac.read_ints()
        pos = [ac.read_list_ints() for _ in range(n)]
        dis1 = [(x - x1) * (x - x1) + (y - y1) * (y - y1) for x, y in pos]
        dis2 = [(x - x2) * (x - x2) + (y - y2) * (y - y2) for x, y in pos]
        # 排序
        ind = list(range(n))
        ind.sort(key=lambda it: dis1[it] - dis2[it])
        # 后缀最大值
        post = [0] * (n + 1)
        ceil = 0
        for i in range(n - 1, -1, -1):
            ceil = ac.max(dis2[ind[i]], ceil)
            post[i] = ceil

        # 枚举前缀
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
        # 模板：经典括号匹配枚举，前后缀遍历计算
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
        # 模板：脑筋急转弯分为奇数与偶数讨论
        n = ac.read_int()
        if n % 2 == 0:
            ac.st(n//2)
        else:
            lst = NumberTheory().get_prime_factor(n)
            x = lst[0][0]
            ac.st(1 + (n-x)//2)
        return

    @staticmethod
    def ac_4623(ac=FastIO()):
        # 模板：贪心模拟
        n, t = ac.read_ints()
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

        # 模板：经典脑筋急转弯思维题

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
    def lc_1675(nums: List[int]) -> int:
        # 模板：脑筋急转弯思维题贪心
        lst = SortedList([num if num % 2 == 0 else num*2 for num in nums])
        ans = lst[-1] - lst[0]
        while True:
            cur = lst[-1] - lst[0]
            ans = ans if ans < cur else cur
            if lst[-1] % 2:
                break
            lst.add(lst.pop()//2)
        return ans

    @staticmethod
    def lc_1808(prime_factors: int) -> int:
        # 模板：按照模3的因子个数贪心处理，经典将和拆分成最大乘积
        mod = 10**9 + 7
        if prime_factors <= 2:
            return prime_factors
        if prime_factors % 3 == 0:
            return pow(3, prime_factors//3, mod)
        elif prime_factors % 3 == 1:
            return (4*pow(3, prime_factors//3 - 1, mod)) % mod
        else:
            return (2*pow(3, prime_factors//3, mod)) % mod

    @staticmethod
    def lc_1927(num: str) -> bool:
        # 模板：经典博弈思维题分类讨论
        def check(s):
            res = 0
            cnt = 0
            for w in s:
                if w.isnumeric():
                    res += int(w)
                else:
                    cnt += 1
            return [res, cnt]

        # 计算左右两边的数字和以及问号个数
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
        # 模板：典型贪心排序后使用双指针计算
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
        # 模板：脑筋急转弯贪心，可以根据暴力打表观察规律
        dct = set(nums)
        for i in range(34):
            if 1 << i not in dct:
                return 1 << i
        return -1


class TestGeneral(unittest.TestCase):

    def test_brain_storming(self):
        bs = BrainStorming()
        n, m = 4, 20
        nums = [1, 2, 5, 10]
        assert bs.minimal_coin_need(n, m, nums) == 5
        return


if __name__ == '__main__':
    unittest.main()
