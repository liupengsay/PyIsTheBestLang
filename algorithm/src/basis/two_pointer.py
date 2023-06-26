import math
import random
import unittest
from collections import defaultdict
from functools import reduce
from math import gcd
from operator import add
from itertools import accumulate
from typing import List
from operator import mul, add, xor, and_, or_
from algorithm.src.fast_io import FastIO

"""
算法：双指针、快慢指针、先后指针、桶计数
功能：通过相对移动，来减少计算复杂度，分为同向双指针，相反双指针，以及中心扩展法

题目：

===================================力扣===================================
259. 较小的三数之和（https://leetcode.cn/problems/3sum-smaller/）使用双指针或者计数枚举的方式
2444. 统计定界子数组的数目（https://leetcode.cn/problems/count-subarrays-with-fixed-bounds/）通向双指针的移动来根据两个指针的位置来进行计数
2398. 预算内的最多机器人数目（https://leetcode.cn/problems/maximum-number-of-robots-within-budget/）同向双指针移动的条件限制有两个需要用有序集合来维护滑动窗口过程
2302. 统计得分小于 K 的子数组数目（https://leetcode.cn/problems/count-subarrays-with-score-less-than-k/）同向双指针维护指针位置与计数
2301. 替换字符后匹配（https://leetcode.cn/problems/match-substring-after-replacement/）枚举匹配字符起点并使用双指针维护可行长度
2106. 摘水果（https://leetcode.cn/problems/maximum-fruits-harvested-after-at-most-k-steps/）巧妙利用行走距离的计算更新双指针
6293. 统计好子数组的数目（https://leetcode.cn/problems/count-the-number-of-good-subarrays/）双指针计数
16. 最接近的三数之和（https://leetcode.cn/problems/3sum-closest/）三指针确定最接近目标值的和
15. 三数之和（https://leetcode.cn/problems/3sum/）寻找三个元素和为 0 的不重复组合
2422. 使用合并操作将数组转换为回文序列（https://leetcode.cn/problems/merge-operations-to-turn-array-into-a-palindrome/）相反方向双指针贪心加和
2524. Maximum Frequency Score of a Subarray（https://leetcode.cn/problems/maximum-frequency-score-of-a-subarray/）滑动窗口维护计算数字数量与幂次取模
239. 滑动窗口最大值（https://leetcode.cn/problems/sliding-window-maximum/）滑动窗口最大值，使用滑动窗口类维护
2447. 最大公因数等于 K 的子数组数目（https://leetcode.cn/problems/number-of-subarrays-with-gcd-equal-to-k/）滑动窗口区间 gcd，使用滑动窗口类维护
6392. 使数组所有元素变成 1 的最少操作次数（https://leetcode.cn/problems/minimum-number-of-operations-to-make-all-array-elements-equal-to-1/）滑动窗口区间 gcd，使用滑动窗口类维护
1163. 按字典序排在最后的子串（https://leetcode.cn/problems/last-substring-in-lexicographical-order/）经典类似最小表示法的双指针

===================================洛谷===================================
P2381 圆圆舞蹈（https://www.luogu.com.cn/problem/P2381）环形数组，滑动窗口双指针
P3353 在你窗外闪耀的星星（https://www.luogu.com.cn/problem/P3353）滑动窗口加双指针
P3662 [USACO17FEB]Why Did the Cow Cross the Road II S（https://www.luogu.com.cn/problem/P3662）滑动子数组和
P4995 跳跳！（https://www.luogu.com.cn/problem/P4995）排序后利用贪心与双指针进行模拟
P2207 Photo（https://www.luogu.com.cn/problem/P2207）贪心加同向双指针
P7542 [COCI2009-2010#1] MALI（https://www.luogu.com.cn/problem/P7542）桶计数加双指针进行计算
P4653 [CEOI2017] Sure Bet（https://www.luogu.com.cn/problem/P4653）贪心排序后使用双指针计算
P3029 [USACO11NOV]Cow Lineup S（https://www.luogu.com.cn/problem/P3029）双指针记录包含k个不同颜色的最短连续子序列
P5583 【SWTR-01】Ethan and Sets（https://www.luogu.com.cn/problem/P5583）经典双指针
P6465 [传智杯 #2 决赛] 课程安排（https://www.luogu.com.cn/problem/P6465）滑动窗口与双指针计数


================================CodeForces================================
D. Carousel（https://codeforces.com/problemset/problem/1328/D）环形数组滑动窗口，记录变化次数并根据奇偶变换次数与环形首尾元素确定染色数量
C. Eugene and an array（https://codeforces.com/problemset/problem/1333/C）双指针，计算前缀和不重复即没有区间段和为0的个数
A2. Prefix Flip (Hard Version)（https://codeforces.com/problemset/problem/1381/A2）双指针模拟翻转匹配与贪心

参考：OI WiKi（xx）
"""

INF = int(1e64)


class SlidingWindowAggregation:
    """SlidingWindowAggregation

    Api:
    1. append value to tail,O(1).
    2. pop value from head,O(1).
    3. query aggregated value in window,O(1).
    """

    def __init__(self, e, op):
        # 模板：滑动窗口维护和查询聚合信息
        """
        Args:
            e: unit element
            op: merge function
        """
        self.stack0 = []
        self.agg0 = []
        self.stack2 = []
        self.stack3 = []
        self.e = e
        self.e0 = self.e
        self.e1 = self.e
        self.size = 0
        self.op = op

    def append(self, value) -> None:
        if not self.stack0:
            self.push0(value)
            self.transfer()
        else:
            self.push1(value)
        self.size += 1

    def popleft(self) -> None:
        if not self.size:
            return
        if not self.stack0:
            self.transfer()
        self.stack0.pop()
        self.stack2.pop()
        self.e0 = self.stack2[-1] if self.stack2 else self.e
        self.size -= 1

    def query(self):
        return self.op(self.e0, self.e1)

    def push0(self, value):
        self.stack0.append(value)
        self.e0 = self.op(value, self.e0)
        self.stack2.append(self.e0)

    def push1(self, value):
        self.agg0.append(value)
        self.e1 = self.op(self.e1, value)
        self.stack3.append(self.e1)

    def transfer(self):
        while self.agg0:
            self.push0(self.agg0.pop())
        while self.stack3:
            self.stack3.pop()
        self.e1 = self.e

    def __len__(self):
        return self.size


class TwoPointer:
    def __init__(self):
        return

    @staticmethod
    def circle_array(arr):
        # 模板：环形数组指针移动
        n = len(arr)
        ans = 0
        for i in range(n):
            ans = max(ans, arr[i] + arr[(i + n - 1) % n])
        return ans

    @staticmethod
    def fast_and_slow(head):
        # 模板：快慢指针判断链表是否存在环
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False

    @staticmethod
    def same_direction(nums):
        # 模板: 相同方向双指针（寻找最长不含重复元素的子序列）
        n = len(nums)
        ans = j = 0
        pre = set()
        for i in range(n):
            # 特别注意指针的移动情况
            while j < n and nums[j] not in pre:
                pre.add(nums[j])
                j += 1
            # 视情况更新返回值
            ans = ans if ans > j - i else j - i
            pre.discard(nums[i])
        return ans

    @staticmethod
    def opposite_direction(nums, target):
        # 模板: 相反方向双指针（寻找升序数组是否存在两个数和为target）
        n = len(nums)
        i, j = 0, n - 1
        while i < j:
            cur = nums[i] + nums[j]
            if cur > target:
                j -= 1
            elif cur < target:
                i += 1
            else:
                return True
        return False


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lg_p4653(ac=FastIO()):

        # 模板：贪心排序后使用双指针计算
        n = ac.read_int()

        nums1 = []
        nums2 = []
        for _ in range(n):
            x, y = ac.read_list_floats()
            nums1.append(x)
            nums2.append(y)
        nums1.sort(reverse=True)
        nums2.sort(reverse=True)
        nums1 = list(accumulate(nums1, add))
        nums2 = list(accumulate(nums2, add))

        ans = i = j = 0
        while i < n and j < n:
            if nums1[i] < nums2[j]:
                ans = ac.max(ans, nums1[i] - i - j - 2)
                i += 1
            else:
                ans = ac.max(ans, nums2[j] - i - j - 2)
                j += 1
        ac.st("%.4f" % ans)
        return

    @staticmethod
    def lc_16(nums, target):
        # 模板：寻找最接近目标值的三个元素和
        n = len(nums)
        nums.sort()
        ans = nums[0] + nums[1] + nums[2]
        for i in range(n - 2):
            j, k = i + 1, n - 1
            x = nums[i]
            while j < k:  # 经典遍历数组作为第一个指针，另外两个指针相向而行
                cur = x + nums[j] + nums[k]
                ans = ans if abs(target - ans) < abs(target - cur) else cur
                if cur > target:
                    k -= 1
                elif cur < target:
                    j += 1
                else:
                    return target
        return ans

    @staticmethod
    def lc_15(nums):
        # 模板：寻找三个元素和为 0 的不重复组合
        nums.sort()
        n = len(nums)
        ans = set()
        for i in range(n - 2):
            j, k = i + 1, n - 1
            x = nums[i]
            while j < k:
                cur = x + nums[j] + nums[k]
                if cur > 0:
                    k -= 1
                elif cur < 0:
                    j += 1
                else:
                    ans.add((x, nums[j], nums[k]))
                    j += 1
                    k -= 1
        return [list(a) for a in ans]

    @staticmethod
    def lc_259(nums: List[int], target: int) -> int:
        # 模板：使用相反方向的双指针统计和小于 target 的三元组数量
        nums.sort()
        n = len(nums)
        ans = 0
        for i in range(n - 2):
            x = nums[i]
            j, k = i + 1, n - 1
            while j < k:
                cur = x + nums[j] + nums[k]
                if cur < target:
                    ans += k - j
                    j += 1
                else:
                    k -= 1
        return ans

    @staticmethod
    def lc_239(nums: List[int], k: int) -> List[int]:
        # 模板：滑动窗口最大值
        n = len(nums)
        swa = SlidingWindowAggregation(-INF, max)
        ans = []
        for i in range(n):
            swa.append(nums[i])
            if i >= k - 1:
                ans.append(swa.query())
                swa.popleft()
        return ans

    @staticmethod
    def lc_6392(nums: List[int]) -> int:
        # 模板：滑动窗口维护区间 gcd 为 1 的长度信息
        if gcd(*nums) != 1:
            return -1
        if 1 in nums:
            return len(nums) - nums.count(1)

        swa = SlidingWindowAggregation(0, gcd)
        res, n = INF, len(nums)
        # 枚举右端点
        for i in range(n):
            swa.append(nums[i])
            while swa and swa.query() == 1:
                res = res if res < swa.size else swa.size
                swa.popleft()
        return res - 1 + len(nums) - 1

    @staticmethod
    def lc_1163(s: str) -> str:
        # 双指针进行计算
        i, j, n = 0, 1, len(s)
        while j < n:
            k = 0
            while j + k < n and s[i + k] == s[j + k]:
                k += 1
            if j + k < n and s[i + k] < s[j + k]:
                i, j = j, j + 1 if j + 1 > i + k + 1 else i + k + 1
            else:
                j = j + k + 1
        return s[i:]

    @staticmethod
    def lc_2447(nums: List[int], k: int) -> int:
        # 模板：滑动窗口双指针三指针维护区间 gcd 为 k 的子数组数量信息
        n = len(nums)
        e = reduce(math.lcm, nums + [k])
        e *= 2

        swa1 = SlidingWindowAggregation(e, gcd)
        ans = j1 = j2 = 0
        swa2 = SlidingWindowAggregation(e, gcd)
        for i in range(n):

            if j1 < i:
                swa1 = SlidingWindowAggregation(e, gcd)
                j1 = i
            if j2 < i:
                swa2 = SlidingWindowAggregation(e, gcd)
                j2 = i

            while j1 < n and swa1.query() > k:
                swa1.append(nums[j1])
                j1 += 1
                if swa1.query() == k:
                    break

            while j2 < n and gcd(swa2.query(), nums[j2]) >= k:
                swa2.append(nums[j2])
                j2 += 1

            if swa1.query() == k:
                ans += j2 - j1 + 1

            swa1.popleft()
            swa2.popleft()
        return ans

    @staticmethod
    def lg_p5583(ac=FastIO()):
        # 模板：双指针与变量维护区间信息
        n, m, d = ac.read_ints()
        nums = ac.read_list_ints()
        cnt = dict()
        for num in nums:
            cnt[num] = cnt.get(num, 0) + 1
        nums = [ac.read_list_ints() for _ in range(n)]
        # 动态维护的变量
        flag = 0
        ans = [-1]
        not_like = inf
        power = -inf
        cur_cnt = defaultdict(int)
        cur_power = cur_not_like = j = 0
        for i in range(n):
            # 移动右指针
            while j < n and (flag < len(cnt) or all(num in cnt for num in nums[j][2:])):
                cur_power += nums[j][0]
                for num in nums[j][2:]:
                    cur_cnt[num] += 1
                    if cur_cnt[num] == 1 and num in cnt:
                        flag += 1
                    if num not in cnt:
                        cur_not_like += 1
                j += 1
            if flag == len(cnt):
                if cur_not_like < not_like or (cur_not_like == not_like and cur_power > power):
                    not_like = cur_not_like
                    power = cur_power
                    ans = [i + 1, j]
            # 删除左指针
            cur_power -= nums[i][0]
            for num in nums[i][2:]:
                cur_cnt[num] -= 1
                if cur_cnt[num] == 0 and num in cnt:
                    flag -= 1
                if num not in cnt:
                    cur_not_like -= 1
        ac.lst(ans)
        return

    @staticmethod
    def lg_p6465(ac=FastIO()):
        # 模板：滑动窗口与双指针计数
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            nums = ac.read_list_ints()
            m = ac.max(m, 2)
            ans = j = s = 0
            cnt = dict()
            for i in range(n):
                if i and nums[i] == nums[i - 1]:
                    cnt = dict()
                    s = 0
                    j = i
                    continue
                while j <= i - m + 1:
                    cnt[nums[j]] = cnt.get(nums[j], 0) + 1
                    s += 1
                    j += 1
                ans += s - cnt.get(nums[i], 0)
            ac.st(ans)
        return


class TestGeneral(unittest.TestCase):

    def test_two_pointer(self):
        nt = TwoPointer()
        nums = [1, 2, 3, 4, 4, 3, 3, 2, 1, 6, 3]
        assert nt.same_direction(nums) == 4

        nums = [1, 2, 3, 4, 4, 5, 6, 9]
        assert nt.opposite_direction(nums, 9)
        nums = [1, 2, 3, 4, 4, 5, 6, 9]
        assert not nt.opposite_direction(nums, 16)
        return

    def test_ops_es(self):

        dct = {max: 0, min: INF, gcd: 0, or_: 0, xor: 0, add: 0, mul: 1, and_: (1 << 32)-1}
        for op in dct:
            for _ in range(1000):
                e = dct[op]
                n = 100
                nums = [random.randint(0, 10**9) for _ in range(n)]
                swa = SlidingWindowAggregation(e, op)
                k = random.randint(1, 50)
                ans = []
                res = []
                for i in range(n):
                    swa.append(nums[i])
                    if i >= k-1:
                        ans.append(swa.query())
                        swa.popleft()
                        lst = nums[i-k+1: i+1]
                        res.append(reduce(op, lst))
                assert len(res) == len(ans)
                assert res == ans



if __name__ == '__main__':
    unittest.main()
